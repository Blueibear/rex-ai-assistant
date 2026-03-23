"""Tests for US-P4-034 and US-P4-035: Executor uses WorkflowBridge.

US-P4-034 acceptance criteria:
  - Executor creates WorkflowBridge (not WorkflowRunner) at execute time
  - WorkflowBridge receives hooks and context from Executor
  - Existing Executor tests still pass

US-P4-035 acceptance criteria:
  - End-to-end: Executor.execute() → WorkflowBridge.run() → RunResult
  - Completed workflow returns ExecutionResult with status "completed"
  - Blocked workflow returns ExecutionResult with blocking_approval_id
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.executor import Executor, ExecutionBudget, ExecutionResult
from rex.openclaw.workflow_bridge import WorkflowBridge
from rex.workflow import Workflow, WorkflowStep
from rex.workflow_runner import RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_workflow() -> Workflow:
    step = WorkflowStep(description="check status")
    return Workflow(title="Simple", steps=[step])


def _make_run_result(
    wf: Workflow,
    status: str = "completed",
    steps_executed: int = 1,
    blocking_approval_id: str | None = None,
) -> RunResult:
    return RunResult(
        workflow_id=wf.workflow_id,
        status=status,
        steps_executed=steps_executed,
        steps_total=len(wf.steps),
        error=None,
        blocking_approval_id=blocking_approval_id,
    )


# ---------------------------------------------------------------------------
# US-P4-034: Executor uses WorkflowBridge
# ---------------------------------------------------------------------------


class TestExecutorUsesWorkflowBridge:
    def test_executor_creates_workflow_bridge(self):
        """Executor.execute() constructs a WorkflowBridge, not WorkflowRunner."""
        wf = _simple_workflow()
        executor = Executor(wf)

        captured = {}

        original_init = WorkflowBridge.__init__

        def capture_init(self_inner, workflow, **kwargs):
            captured["bridge_created"] = True
            captured["workflow"] = workflow
            original_init(self_inner, workflow, **kwargs)

        mock_run_result = _make_run_result(wf)

        with patch.object(WorkflowBridge, "__init__", capture_init):
            with patch.object(WorkflowBridge, "run", return_value=mock_run_result):
                executor.run()

        assert captured.get("bridge_created") is True
        assert captured["workflow"] is wf

    def test_executor_forwards_hooks_to_bridge(self):
        """Executor passes its budget hooks to WorkflowBridge."""
        wf = _simple_workflow()
        executor = Executor(wf)

        kwargs_captured = {}
        original_init = WorkflowBridge.__init__

        def capture_kwargs(self_inner, workflow, **kwargs):
            kwargs_captured.update(kwargs)
            original_init(self_inner, workflow, **kwargs)

        mock_run_result = _make_run_result(wf)

        with patch.object(WorkflowBridge, "__init__", capture_kwargs):
            with patch.object(WorkflowBridge, "run", return_value=mock_run_result):
                executor.run()

        assert "pre_step_hook" in kwargs_captured
        assert "before_tool_call_hook" in kwargs_captured
        assert "after_tool_call_hook" in kwargs_captured
        assert callable(kwargs_captured["pre_step_hook"])


# ---------------------------------------------------------------------------
# US-P4-035: End-to-end autonomy execution through bridge
# ---------------------------------------------------------------------------


class TestAutonomyExecutionThroughBridge:
    def test_completed_workflow_returns_execution_result(self):
        """Executor.execute() returns ExecutionResult with status 'completed'."""
        wf = _simple_workflow()
        executor = Executor(wf)
        mock_run = _make_run_result(wf, status="completed", steps_executed=1)

        with patch.object(WorkflowBridge, "run", return_value=mock_run):
            result = executor.run()

        assert isinstance(result, ExecutionResult)
        assert result.status == "completed"

    def test_blocked_workflow_returns_blocking_approval_id(self):
        """Executor.execute() propagates blocking_approval_id from bridge."""
        step = WorkflowStep(description="send email to users", requires_approval=True)
        wf = Workflow(title="Approval Workflow", steps=[step])
        executor = Executor(wf)

        mock_run = _make_run_result(
            wf, status="blocked", steps_executed=0, blocking_approval_id="approval-xyz"
        )
        with patch.object(WorkflowBridge, "run", return_value=mock_run):
            result = executor.run()

        assert result.status == "blocked"
        assert result.blocking_approval_id == "approval-xyz"

    def test_failed_workflow_returns_failed_status(self):
        """Executor.execute() returns ExecutionResult with status 'failed' on error."""
        wf = _simple_workflow()
        executor = Executor(wf)

        mock_run = RunResult(
            workflow_id=wf.workflow_id,
            status="failed",
            steps_executed=0,
            steps_total=1,
            error="Tool call failed",
            blocking_approval_id=None,
        )
        with patch.object(WorkflowBridge, "run", return_value=mock_run):
            result = executor.run()

        assert result.status == "failed"

    def test_multi_step_execution_through_bridge(self):
        """Executor with 3-step workflow completes via bridge."""
        steps = [WorkflowStep(description=f"step {i}") for i in range(3)]
        wf = Workflow(title="3-Step", steps=steps)
        executor = Executor(wf)

        mock_run = _make_run_result(wf, status="completed", steps_executed=3)
        mock_run.steps_total = 3

        with patch.object(WorkflowBridge, "run", return_value=mock_run):
            result = executor.run()

        assert result.status == "completed"
