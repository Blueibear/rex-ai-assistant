"""Tests for US-064: Workflow state persistence.

Verifies that workflow state is saved and restored across restarts and that
step progress is tracked correctly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rex.contracts import ToolCall
from rex.workflow import (
    StepResult,
    Workflow,
    WorkflowStep,
    generate_step_id,
    generate_workflow_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(description: str = "do something") -> WorkflowStep:
    return WorkflowStep(
        step_id=generate_step_id(),
        description=description,
        tool_call=ToolCall(tool="time_now", args={}),
    )


def _make_workflow(num_steps: int = 3) -> Workflow:
    return Workflow(
        workflow_id=generate_workflow_id(),
        title="Test workflow",
        steps=[_make_step(f"step {i}") for i in range(num_steps)],
    )


# ---------------------------------------------------------------------------
# Workflow state saved
# ---------------------------------------------------------------------------


class TestWorkflowStateSaved:
    """Workflow.save() persists state to disk."""

    def test_save_creates_file(self, tmp_path):
        wf = _make_workflow()
        file_path = wf.save(workflow_dir=tmp_path)
        assert file_path.exists()

    def test_saved_file_contains_workflow_id(self, tmp_path):
        wf = _make_workflow()
        file_path = wf.save(workflow_dir=tmp_path)
        content = file_path.read_text(encoding="utf-8")
        assert wf.workflow_id in content

    def test_save_persists_status(self, tmp_path):
        wf = _make_workflow()
        wf.mark_running()
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.status == "running"

    def test_save_persists_state_dict(self, tmp_path):
        wf = _make_workflow()
        wf.state["result"] = "some_value"
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.state["result"] == "some_value"

    def test_save_overwrites_existing_file(self, tmp_path):
        wf = _make_workflow()
        wf.save(workflow_dir=tmp_path)
        wf.mark_running()
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.status == "running"

    def test_save_persists_step_results(self, tmp_path):
        wf = _make_workflow(num_steps=2)
        step = wf.steps[0]
        step.result = StepResult(step_id=step.step_id, success=True, output={"key": "val"})
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.steps[0].result is not None
        assert loaded.steps[0].result.success is True


# ---------------------------------------------------------------------------
# State restored after restart
# ---------------------------------------------------------------------------


class TestStateRestoredAfterRestart:
    """Workflow.load() restores full state as if no restart occurred."""

    def test_load_returns_workflow(self, tmp_path):
        wf = _make_workflow()
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None

    def test_load_restores_title(self, tmp_path):
        wf = _make_workflow()
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.title == wf.title

    def test_load_restores_steps(self, tmp_path):
        wf = _make_workflow(num_steps=4)
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert len(loaded.steps) == 4

    def test_load_restores_state_dict(self, tmp_path):
        wf = _make_workflow()
        wf.state["key1"] = 42
        wf.state["key2"] = "hello"
        wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.state["key1"] == 42
        assert loaded.state["key2"] == "hello"

    def test_load_missing_workflow_returns_none(self, tmp_path):
        result = Workflow.load("nonexistent_wf_id", workflow_dir=tmp_path)
        assert result is None

    def test_load_from_file_restores_workflow(self, tmp_path):
        wf = _make_workflow()
        file_path = wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load_from_file(file_path)
        assert loaded.workflow_id == wf.workflow_id

    def test_load_from_file_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Workflow.load_from_file(tmp_path / "no_such_file.json")

    def test_multiple_workflows_persist_independently(self, tmp_path):
        wf1 = _make_workflow(num_steps=1)
        wf2 = _make_workflow(num_steps=2)
        wf1.save(workflow_dir=tmp_path)
        wf2.save(workflow_dir=tmp_path)
        l1 = Workflow.load(wf1.workflow_id, workflow_dir=tmp_path)
        l2 = Workflow.load(wf2.workflow_id, workflow_dir=tmp_path)
        assert l1 is not None and len(l1.steps) == 1
        assert l2 is not None and len(l2.steps) == 2


# ---------------------------------------------------------------------------
# Step progress tracked
# ---------------------------------------------------------------------------


class TestStepProgressTracked:
    """current_step_index and step results track execution progress."""

    def test_initial_step_index_is_zero(self):
        wf = _make_workflow(num_steps=3)
        assert wf.current_step_index == 0

    def test_advance_increments_step_index(self):
        wf = _make_workflow(num_steps=3)
        wf.advance()
        assert wf.current_step_index == 1

    def test_advance_to_last_step_completes_workflow(self):
        wf = _make_workflow(num_steps=2)
        wf.advance()
        wf.advance()
        assert wf.status == "completed"

    def test_current_step_returns_correct_step(self):
        wf = _make_workflow(num_steps=3)
        wf.advance()
        step = wf.current_step()
        assert step is not None
        assert step.description == "step 1"

    def test_current_step_returns_none_when_finished(self):
        wf = _make_workflow(num_steps=1)
        wf.advance()
        assert wf.current_step() is None

    def test_step_result_stored_in_step(self):
        wf = _make_workflow(num_steps=2)
        step = wf.steps[0]
        step.result = StepResult(step_id=step.step_id, success=True)
        assert step.result.success is True

    def test_step_progress_persisted_and_restored(self, tmp_path):
        wf = _make_workflow(num_steps=3)
        wf.mark_running()
        wf.advance()
        # Record a result on the first step
        wf.steps[0].result = StepResult(step_id=wf.steps[0].step_id, success=True)
        wf.save(workflow_dir=tmp_path)

        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.current_step_index == 1
        assert loaded.steps[0].result is not None
        assert loaded.steps[0].result.success is True

    def test_idempotency_keys_tracked(self):
        wf = _make_workflow(num_steps=2)
        wf.steps[0].idempotency_key = "key_a"
        wf.steps[0].result = StepResult(step_id=wf.steps[0].step_id, success=True)
        keys = wf.get_executed_idempotency_keys()
        assert "key_a" in keys

    def test_failed_step_idempotency_key_not_in_executed(self):
        wf = _make_workflow(num_steps=1)
        wf.steps[0].idempotency_key = "key_b"
        wf.steps[0].result = StepResult(step_id=wf.steps[0].step_id, success=False)
        keys = wf.get_executed_idempotency_keys()
        assert "key_b" not in keys

    def test_list_workflows_filters_by_status(self, tmp_path):
        wf_run = _make_workflow(num_steps=1)
        wf_run.mark_running()
        wf_run.save(workflow_dir=tmp_path)

        wf_done = _make_workflow(num_steps=1)
        wf_done.advance()  # completes single-step workflow
        wf_done.save(workflow_dir=tmp_path)

        running = Workflow.list_workflows(workflow_dir=tmp_path, status="running")
        completed = Workflow.list_workflows(workflow_dir=tmp_path, status="completed")

        assert any(w.workflow_id == wf_run.workflow_id for w in running)
        assert any(w.workflow_id == wf_done.workflow_id for w in completed)
