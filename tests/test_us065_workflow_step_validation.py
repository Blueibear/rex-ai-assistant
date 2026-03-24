"""Tests for US-065: Workflow step validation.

Verifies that step inputs are validated before execution, invalid steps are
rejected, and workflow execution is halted on validation failure.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rex.contracts import ToolCall
from rex.workflow import (
    Workflow,
    WorkflowStep,
    generate_step_id,
    generate_workflow_id,
    validate_workflow_steps,
)
from rex.workflow_runner import WorkflowRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow(steps: list[WorkflowStep], tmp_dir: Path) -> Workflow:
    return Workflow(
        workflow_id=generate_workflow_id(),
        title="Test workflow",
        steps=steps,
    )


def _make_runner(workflow: Workflow, tmp_dir: Path) -> WorkflowRunner:
    from rex.audit import AuditLogger
    from rex.policy_engine import PolicyEngine

    return WorkflowRunner(
        workflow,
        policy_engine=PolicyEngine(),
        audit_logger=AuditLogger(log_file=str(tmp_dir / "audit.log")),
        workflow_dir=tmp_dir / "workflows",
        approval_dir=tmp_dir / "approvals",
    )


# ---------------------------------------------------------------------------
# Step inputs validated — WorkflowStep.validate_inputs()
# ---------------------------------------------------------------------------


class TestStepInputsValidated:
    def test_valid_step_no_errors(self) -> None:
        step = WorkflowStep(
            step_id=generate_step_id(),
            description="Do something useful",
            tool_call=ToolCall(tool="time_now", args={}),
        )
        assert step.validate_inputs() == []

    def test_valid_step_no_tool_call(self) -> None:
        step = WorkflowStep(
            step_id=generate_step_id(),
            description="Checkpoint step",
        )
        assert step.validate_inputs() == []

    def test_empty_description_invalid(self) -> None:
        step = WorkflowStep(
            step_id="step_abc",
            description="",
        )
        errors = step.validate_inputs()
        assert len(errors) == 1
        assert "description" in errors[0]
        assert "step_abc" in errors[0]

    def test_whitespace_only_description_invalid(self) -> None:
        step = WorkflowStep(
            step_id="step_xyz",
            description="   ",
        )
        errors = step.validate_inputs()
        assert len(errors) == 1
        assert "description" in errors[0]

    def test_empty_tool_name_invalid(self) -> None:
        step = WorkflowStep(
            step_id="step_t1",
            description="Valid description",
            tool_call=ToolCall(tool="", args={}),
        )
        errors = step.validate_inputs()
        assert len(errors) == 1
        assert "tool_call.tool" in errors[0]
        assert "step_t1" in errors[0]

    def test_multiple_errors_reported(self) -> None:
        step = WorkflowStep(
            step_id="step_bad",
            description="",
            tool_call=ToolCall(tool="", args={}),
        )
        errors = step.validate_inputs()
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Invalid steps rejected — validate_workflow_steps()
# ---------------------------------------------------------------------------


class TestInvalidStepsRejected:
    def test_all_valid_steps_returns_empty(self) -> None:
        steps = [
            WorkflowStep(description="Step one", tool_call=ToolCall(tool="ping", args={})),
            WorkflowStep(description="Step two"),
        ]
        workflow = Workflow(workflow_id=generate_workflow_id(), title="T", steps=steps)
        result = validate_workflow_steps(workflow)
        assert result == {}

    def test_one_invalid_step_returned(self) -> None:
        bad_step = WorkflowStep(step_id="bad1", description="")
        good_step = WorkflowStep(step_id="good1", description="Valid")
        workflow = Workflow(
            workflow_id=generate_workflow_id(),
            title="T",
            steps=[bad_step, good_step],
        )
        result = validate_workflow_steps(workflow)
        assert "bad1" in result
        assert "good1" not in result

    def test_multiple_invalid_steps_all_returned(self) -> None:
        steps = [
            WorkflowStep(step_id="bad1", description=""),
            WorkflowStep(step_id="good1", description="OK"),
            WorkflowStep(
                step_id="bad2",
                description="Has tool",
                tool_call=ToolCall(tool="", args={}),
            ),
        ]
        workflow = Workflow(workflow_id=generate_workflow_id(), title="T", steps=steps)
        result = validate_workflow_steps(workflow)
        assert "bad1" in result
        assert "bad2" in result
        assert "good1" not in result

    def test_empty_workflow_returns_empty(self) -> None:
        workflow = Workflow(workflow_id=generate_workflow_id(), title="Empty", steps=[])
        assert validate_workflow_steps(workflow) == {}


# ---------------------------------------------------------------------------
# Workflow execution halted on failure
# ---------------------------------------------------------------------------


class TestWorkflowHaltedOnValidationFailure:
    def test_invalid_step_halts_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            bad_step = WorkflowStep(step_id="bad_s", description="")
            workflow = Workflow(
                workflow_id=generate_workflow_id(),
                title="Bad workflow",
                steps=[bad_step],
            )
            runner = _make_runner(workflow, tmp_dir)
            result = runner.run()

            assert result.status == "failed"
            assert result.steps_executed == 0
            assert result.error is not None
            assert "validation" in result.error.lower()

    def test_no_steps_executed_when_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            steps = [
                WorkflowStep(step_id="bad_s", description=""),
                WorkflowStep(
                    step_id="good_s",
                    description="Would run",
                    tool_call=ToolCall(tool="time_now", args={}),
                ),
            ]
            workflow = Workflow(
                workflow_id=generate_workflow_id(),
                title="Mixed workflow",
                steps=steps,
            )
            runner = _make_runner(workflow, tmp_dir)
            result = runner.run()

            assert result.status == "failed"
            assert result.steps_executed == 0

    def test_valid_workflow_still_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            step = WorkflowStep(
                step_id="good_s",
                description="Valid step with no tool",
            )
            workflow = Workflow(
                workflow_id=generate_workflow_id(),
                title="Valid workflow",
                steps=[step],
            )
            runner = _make_runner(workflow, tmp_dir)
            result = runner.run()

            # Workflow should not be failed due to validation
            assert result.status != "failed" or (
                result.error is not None and "validation" not in result.error.lower()
            )

    def test_workflow_marked_failed_on_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            bad_step = WorkflowStep(
                step_id="bad_tool",
                description="Has empty tool",
                tool_call=ToolCall(tool="", args={}),
            )
            workflow = Workflow(
                workflow_id=generate_workflow_id(),
                title="Empty tool workflow",
                steps=[bad_step],
            )
            runner = _make_runner(workflow, tmp_dir)
            result = runner.run()

            assert result.status == "failed"
            assert result.error is not None
            assert "tool_call.tool" in result.error or "validation" in result.error.lower()
