"""Tests for US-026: Workflow definitions.

Acceptance Criteria:
- workflows defined
- schema validated
- workflows stored
- Typecheck passes
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rex.contracts import ToolCall
from rex.workflow import (
    Workflow,
    WorkflowApproval,
    WorkflowStep,
    clear_condition_registry,
    generate_approval_id,
    generate_step_id,
    generate_workflow_id,
    get_condition,
    register_condition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_step(**kwargs: object) -> WorkflowStep:
    kwargs.setdefault("description", "test step")  # type: ignore[attr-defined]
    return WorkflowStep(**kwargs)  # type: ignore[arg-type]


def make_workflow(**kwargs: object) -> Workflow:
    kwargs.setdefault("title", "test workflow")  # type: ignore[attr-defined]
    return Workflow(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Workflows defined
# ---------------------------------------------------------------------------


class TestWorkflowsDefined:
    def test_workflow_instantiates(self) -> None:
        wf = make_workflow()
        assert wf.workflow_id.startswith("wf_")
        assert wf.title == "test workflow"

    def test_workflow_step_instantiates(self) -> None:
        step = make_step()
        assert step.step_id.startswith("step_")
        assert step.description == "test step"

    def test_workflow_approval_instantiates(self) -> None:
        apr = WorkflowApproval(workflow_id="wf_abc", step_id="step_001")
        assert apr.approval_id.startswith("apr_")
        assert apr.status == "pending"

    def test_generate_workflow_id_unique(self) -> None:
        ids = {generate_workflow_id() for _ in range(10)}
        assert len(ids) == 10

    def test_generate_step_id_unique(self) -> None:
        ids = {generate_step_id() for _ in range(10)}
        assert len(ids) == 10

    def test_generate_approval_id_unique(self) -> None:
        ids = {generate_approval_id() for _ in range(10)}
        assert len(ids) == 10

    def test_workflow_with_steps(self) -> None:
        step = make_step(tool_call=ToolCall(tool="time_now", args={}))
        wf = make_workflow(steps=[step])
        assert len(wf.steps) == 1
        assert wf.steps[0].tool_call is not None
        assert wf.steps[0].tool_call.tool == "time_now"

    def test_workflow_default_status_queued(self) -> None:
        wf = make_workflow()
        assert wf.status == "queued"

    def test_workflow_status_transitions(self) -> None:
        wf = make_workflow()
        wf.mark_running()
        assert wf.status == "running"
        wf.mark_blocked("apr_001")
        assert wf.status == "blocked"
        assert wf.blocking_approval_id == "apr_001"
        wf.mark_failed("something went wrong")
        assert wf.status == "failed"
        assert wf.error == "something went wrong"

    def test_workflow_is_finished(self) -> None:
        wf = make_workflow()
        assert not wf.is_finished()
        wf.status = "completed"
        assert wf.is_finished()
        wf.status = "failed"
        assert wf.is_finished()
        wf.status = "canceled"
        assert wf.is_finished()

    def test_workflow_is_blocked(self) -> None:
        wf = make_workflow()
        assert not wf.is_blocked()
        wf.mark_blocked("apr_xyz")
        assert wf.is_blocked()

    def test_workflow_advance(self) -> None:
        steps = [make_step(description=f"step {i}") for i in range(3)]
        wf = make_workflow(steps=steps)
        assert wf.current_step_index == 0
        more = wf.advance()
        assert more is True
        assert wf.current_step_index == 1
        wf.advance()
        more = wf.advance()
        assert more is False
        assert wf.status == "completed"


# ---------------------------------------------------------------------------
# Schema validated
# ---------------------------------------------------------------------------


class TestSchemaValidated:
    def test_workflow_requires_title(self) -> None:
        with pytest.raises((TypeError, ValidationError)):
            Workflow()  # type: ignore[call-arg]

    def test_workflow_step_requires_description(self) -> None:
        with pytest.raises((TypeError, ValidationError)):
            WorkflowStep()  # type: ignore[call-arg]

    def test_workflow_approval_requires_workflow_and_step_id(self) -> None:
        with pytest.raises((TypeError, ValidationError)):
            WorkflowApproval()  # type: ignore[call-arg]

    def test_workflow_status_enum_validated(self) -> None:
        with pytest.raises(ValidationError):
            make_workflow(status="invalid_status")

    def test_approval_status_enum_validated(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowApproval(
                workflow_id="wf_001",
                step_id="step_001",
                status="bad_status",  # type: ignore[arg-type]
            )

    def test_workflow_roundtrip_json(self) -> None:
        step = make_step(
            tool_call=ToolCall(tool="echo", args={"msg": "hello"}),
            idempotency_key="echo_hello",
        )
        wf = make_workflow(steps=[step], state={"key": "value"})
        json_str = wf.model_dump_json()
        restored = Workflow.model_validate_json(json_str)
        assert restored.workflow_id == wf.workflow_id
        assert restored.title == wf.title
        assert len(restored.steps) == 1
        assert restored.steps[0].idempotency_key == "echo_hello"

    def test_approval_roundtrip_json(self) -> None:
        apr = WorkflowApproval(
            workflow_id="wf_001",
            step_id="step_001",
            requested_by="runner",
            step_description="Do a thing",
        )
        restored = WorkflowApproval.model_validate_json(apr.model_dump_json())
        assert restored.approval_id == apr.approval_id
        assert restored.status == "pending"


# ---------------------------------------------------------------------------
# Workflows stored
# ---------------------------------------------------------------------------


class TestWorkflowsStored:
    def test_workflow_save_and_load(self, tmp_path: Path) -> None:
        wf = make_workflow()
        saved_path = wf.save(workflow_dir=tmp_path)
        assert saved_path.exists()
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.workflow_id == wf.workflow_id
        assert loaded.title == wf.title

    def test_workflow_save_creates_directory(self, tmp_path: Path) -> None:
        subdir = tmp_path / "nested" / "workflows"
        wf = make_workflow()
        wf.save(workflow_dir=subdir)
        assert subdir.exists()

    def test_workflow_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        result = Workflow.load("wf_doesnotexist", workflow_dir=tmp_path)
        assert result is None

    def test_workflow_load_from_file(self, tmp_path: Path) -> None:
        wf = make_workflow()
        saved_path = wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load_from_file(saved_path)
        assert loaded.workflow_id == wf.workflow_id

    def test_workflow_load_from_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Workflow.load_from_file(tmp_path / "missing.json")

    def test_workflow_load_from_file_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError):
            Workflow.load_from_file(bad_file)

    def test_workflow_list_workflows(self, tmp_path: Path) -> None:
        wf1 = make_workflow(title="first")
        wf2 = make_workflow(title="second")
        wf1.save(workflow_dir=tmp_path)
        wf2.save(workflow_dir=tmp_path)
        workflows = Workflow.list_workflows(workflow_dir=tmp_path)
        assert len(workflows) == 2

    def test_workflow_list_workflows_empty_dir(self, tmp_path: Path) -> None:
        workflows = Workflow.list_workflows(workflow_dir=tmp_path / "nonexistent")
        assert workflows == []

    def test_workflow_list_workflows_filter_by_status(self, tmp_path: Path) -> None:
        wf_queued = make_workflow(title="q")
        wf_failed = make_workflow(title="f")
        wf_failed.mark_failed("oops")
        wf_queued.save(workflow_dir=tmp_path)
        wf_failed.save(workflow_dir=tmp_path)
        queued = Workflow.list_workflows(workflow_dir=tmp_path, status="queued")
        assert len(queued) == 1
        assert queued[0].title == "q"

    def test_approval_save_and_load(self, tmp_path: Path) -> None:
        apr = WorkflowApproval(workflow_id="wf_001", step_id="step_001")
        apr.save(approval_dir=tmp_path)
        loaded = WorkflowApproval.load(apr.approval_id, approval_dir=tmp_path)
        assert loaded is not None
        assert loaded.approval_id == apr.approval_id

    def test_approval_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        result = WorkflowApproval.load("apr_missing", approval_dir=tmp_path)
        assert result is None

    def test_condition_registry(self) -> None:
        clear_condition_registry()
        register_condition("always_true", lambda state: True)
        func = get_condition("always_true")
        assert func is not None
        assert func({}) is True

    def test_get_condition_missing_returns_none(self) -> None:
        clear_condition_registry()
        assert get_condition("nonexistent") is None
