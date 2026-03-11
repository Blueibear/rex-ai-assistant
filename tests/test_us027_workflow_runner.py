"""Tests for US-027: Workflow runner.

Acceptance Criteria:
- workflows executed
- step transitions work
- errors handled
- Typecheck passes
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.contracts import ToolCall
from rex.policy import PolicyDecision
from rex.workflow import (
    Workflow,
    WorkflowApproval,
    WorkflowStep,
    clear_condition_registry,
    register_condition,
)
from rex.workflow_runner import (
    ApprovalBlockedError,
    RunResult,
    WorkflowRunner,
    approve_workflow,
    deny_workflow,
    list_pending_approvals,
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


def make_allowed_policy() -> MagicMock:
    engine = MagicMock()
    decision = PolicyDecision(allowed=True, reason="allowed", denied=False, requires_approval=False)
    engine.decide.return_value = decision
    return engine


def make_denied_policy() -> MagicMock:
    engine = MagicMock()
    decision = PolicyDecision(allowed=False, reason="denied", denied=True, requires_approval=False)
    engine.decide.return_value = decision
    return engine


def make_approval_policy() -> MagicMock:
    engine = MagicMock()
    decision = PolicyDecision(
        allowed=False, reason="requires approval", denied=False, requires_approval=True
    )
    engine.decide.return_value = decision
    return engine


def make_audit_logger() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Workflows executed
# ---------------------------------------------------------------------------


class TestWorkflowsExecuted:
    def test_empty_workflow_completes(self) -> None:
        wf = make_workflow(steps=[])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_allowed_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            result = runner.run()
        assert result.status == "completed"
        assert result.steps_executed == 0
        assert result.steps_total == 0

    def test_single_step_no_tool_completes(self) -> None:
        step = make_step(step_id="s1")
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_allowed_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            result = runner.run()
        assert result.status == "completed"
        assert result.steps_executed == 1

    def test_single_step_with_tool_executes(self) -> None:
        tool_call = ToolCall(tool="echo", args={"msg": "hello"})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("rex.workflow_runner.execute_tool", return_value={"result": "ok"}):
                runner = WorkflowRunner(
                    wf,
                    policy_engine=make_allowed_policy(),
                    audit_logger=make_audit_logger(),
                    workflow_dir=tmp,
                    approval_dir=tmp,
                )
                result = runner.run()
        assert result.status == "completed"
        assert result.steps_executed == 1

    def test_multi_step_workflow_completes(self) -> None:
        steps = [make_step(step_id=f"s{i}") for i in range(3)]
        wf = make_workflow(steps=steps)
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_allowed_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            result = runner.run()
        assert result.status == "completed"
        assert result.steps_executed == 3

    def test_run_result_has_correct_fields(self) -> None:
        step = make_step(step_id="s1")
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_allowed_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            result = runner.run()
        assert isinstance(result, RunResult)
        assert result.workflow_id == wf.workflow_id
        assert result.steps_total == 1
        assert result.error is None
        assert result.blocking_approval_id is None


# ---------------------------------------------------------------------------
# Step transitions work
# ---------------------------------------------------------------------------


class TestStepTransitions:
    def test_workflow_advances_through_steps(self) -> None:
        tool_call = ToolCall(tool="echo", args={})
        steps = [make_step(step_id=f"s{i}", tool_call=tool_call) for i in range(3)]
        wf = make_workflow(steps=steps)
        executed_tools: list[str] = []

        def fake_execute_tool(
            call: dict,
            context: dict,
            **kwargs: object,
        ) -> dict:
            executed_tools.append(call["tool"])
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            with patch("rex.workflow_runner.execute_tool", side_effect=fake_execute_tool):
                runner = WorkflowRunner(
                    wf,
                    policy_engine=make_allowed_policy(),
                    audit_logger=make_audit_logger(),
                    workflow_dir=tmp,
                    approval_dir=tmp,
                )
                result = runner.run()
        assert result.steps_executed == 3
        assert len(executed_tools) == 3

    def test_workflow_status_becomes_completed(self) -> None:
        step = make_step(step_id="s1")
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_allowed_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            runner.run()
        assert wf.status == "completed"

    def test_pre_step_hook_called(self) -> None:
        visited: list[str] = []
        step = make_step(step_id="s1")
        wf = make_workflow(steps=[step])

        def hook(s: WorkflowStep) -> None:
            visited.append(s.step_id)

        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_allowed_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
                pre_step_hook=hook,
            )
            runner.run()
        assert "s1" in visited

    def test_before_and_after_tool_call_hooks_called(self) -> None:
        before_calls: list[str] = []
        after_calls: list[str] = []
        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])

        def before_hook(s: WorkflowStep) -> None:
            before_calls.append(s.step_id)

        def after_hook(s: WorkflowStep, r: object) -> None:
            after_calls.append(s.step_id)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("rex.workflow_runner.execute_tool", return_value={"ok": True}):
                runner = WorkflowRunner(
                    wf,
                    policy_engine=make_allowed_policy(),
                    audit_logger=make_audit_logger(),
                    workflow_dir=tmp,
                    approval_dir=tmp,
                    before_tool_call_hook=before_hook,
                    after_tool_call_hook=after_hook,
                )
                runner.run()
        assert "s1" in before_calls
        assert "s1" in after_calls

    def test_precondition_false_skips_step(self) -> None:
        clear_condition_registry()
        register_condition("always_false", lambda state: False)

        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call, precondition="always_false")
        wf = make_workflow(steps=[step])

        executed = []

        def fake_execute_tool(call: dict, context: dict, **kwargs: object) -> dict:
            executed.append(True)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            with patch("rex.workflow_runner.execute_tool", side_effect=fake_execute_tool):
                runner = WorkflowRunner(
                    wf,
                    policy_engine=make_allowed_policy(),
                    audit_logger=make_audit_logger(),
                    workflow_dir=tmp,
                    approval_dir=tmp,
                )
                result = runner.run()

        # Step was skipped (not a failure)
        assert result.status == "completed"
        assert len(executed) == 0
        clear_condition_registry()

    def test_idempotency_key_skips_duplicate(self) -> None:
        from rex.workflow import StepResult

        tool_call = ToolCall(tool="echo", args={})
        # A "previous" step that already ran successfully with the same key
        prev_step = make_step(step_id="s0", tool_call=tool_call, idempotency_key="key_abc")
        prev_step.result = StepResult(step_id="s0", success=True)
        # The step we want to skip
        step = make_step(step_id="s1", tool_call=tool_call, idempotency_key="key_abc")
        wf = make_workflow(steps=[prev_step, step])
        # Start at step index 1 (s0 already done, s1 is next)
        wf.current_step_index = 1

        executed = []

        def fake_execute_tool(call: dict, context: dict, **kwargs: object) -> dict:
            executed.append(True)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            with patch("rex.workflow_runner.execute_tool", side_effect=fake_execute_tool):
                runner = WorkflowRunner(
                    wf,
                    policy_engine=make_allowed_policy(),
                    audit_logger=make_audit_logger(),
                    workflow_dir=tmp,
                    approval_dir=tmp,
                )
                result = runner.run()

        assert result.status == "completed"
        assert len(executed) == 0


# ---------------------------------------------------------------------------
# Errors handled
# ---------------------------------------------------------------------------


class TestErrorsHandled:
    def test_denied_step_fails_workflow(self) -> None:
        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_denied_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            result = runner.run()
        assert result.status == "failed"
        assert result.error is not None

    def test_tool_exception_fails_step(self) -> None:
        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])

        def failing_execute_tool(call: dict, context: dict, **kwargs: object) -> dict:
            raise RuntimeError("tool crashed")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("rex.workflow_runner.execute_tool", side_effect=failing_execute_tool):
                runner = WorkflowRunner(
                    wf,
                    policy_engine=make_allowed_policy(),
                    audit_logger=make_audit_logger(),
                    workflow_dir=tmp,
                    approval_dir=tmp,
                )
                result = runner.run()
        assert result.status == "failed"
        assert "tool crashed" in (result.error or "")

    def test_tool_error_result_fails_step(self) -> None:
        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])

        def error_execute_tool(call: dict, context: dict, **kwargs: object) -> dict:
            return {"error": {"message": "something went wrong"}}

        with tempfile.TemporaryDirectory() as tmp:
            with patch("rex.workflow_runner.execute_tool", side_effect=error_execute_tool):
                runner = WorkflowRunner(
                    wf,
                    policy_engine=make_allowed_policy(),
                    audit_logger=make_audit_logger(),
                    workflow_dir=tmp,
                    approval_dir=tmp,
                )
                result = runner.run()
        assert result.status == "failed"
        assert "something went wrong" in (result.error or "")

    def test_approval_required_blocks_workflow(self) -> None:
        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_approval_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            result = runner.run()
        assert result.status == "blocked"
        assert result.blocking_approval_id is not None

    def test_resume_with_approved_approval_completes(self) -> None:
        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_approval_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=Path(tmp),
                approval_dir=Path(tmp),
            )
            blocked_result = runner.run()
            assert blocked_result.status == "blocked"
            approval_id = blocked_result.blocking_approval_id
            assert approval_id is not None

            # Approve it
            approve_workflow(approval_id, approval_dir=tmp)

            # Now resume with allowed policy so execution proceeds
            runner.policy_engine = make_allowed_policy()
            with patch("rex.workflow_runner.execute_tool", return_value={"ok": True}):
                resume_result = runner.resume()
        assert resume_result.status == "completed"

    def test_resume_with_denied_approval_fails(self) -> None:
        tool_call = ToolCall(tool="echo", args={})
        step = make_step(step_id="s1", tool_call=tool_call)
        wf = make_workflow(steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_approval_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=Path(tmp),
                approval_dir=Path(tmp),
            )
            blocked_result = runner.run()
            assert blocked_result.status == "blocked"
            approval_id = blocked_result.blocking_approval_id
            assert approval_id is not None

            deny_workflow(approval_id, reason="not allowed", approval_dir=tmp)
            resume_result = runner.resume()
        assert resume_result.status == "failed"

    def test_resume_non_blocked_raises(self) -> None:
        wf = make_workflow(steps=[])
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf,
                policy_engine=make_allowed_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            runner.run()  # completes

        with pytest.raises(ValueError, match="Cannot resume"):
            runner.resume()

    def test_list_pending_approvals_returns_pending(self) -> None:
        wf = make_workflow(steps=[make_step(step_id="s1")])
        wf_blocked = make_workflow(
            steps=[make_step(step_id="s1", tool_call=ToolCall(tool="echo", args={}))]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner = WorkflowRunner(
                wf_blocked,
                policy_engine=make_approval_policy(),
                audit_logger=make_audit_logger(),
                workflow_dir=tmp,
                approval_dir=tmp,
            )
            runner.run()
            pending = list_pending_approvals(approval_dir=tmp)
        assert len(pending) == 1
        assert pending[0].status == "pending"
