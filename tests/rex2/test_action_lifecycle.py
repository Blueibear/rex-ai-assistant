from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest


def test_canonical_action_states_are_exact_and_ordered_by_transition_contract() -> None:
    from rex.actions.lifecycle import ActionState

    assert [state.value for state in ActionState] == [
        "planned",
        "authorized",
        "attempted",
        "completed",
        "verified",
        "unverified",
        "failed",
        "cancelled",
    ]


def test_valid_verified_transition_chain_preserves_immutable_correlation() -> None:
    from rex.actions.lifecycle import ActionLifecycle, ActionState

    lifecycle = ActionLifecycle.create(action_id="action-123", plan_id="plan-9")
    correlation = lifecycle.correlation

    lifecycle.transition(ActionState.AUTHORIZED, evidence_ref="policy:allow")
    lifecycle.transition(ActionState.ATTEMPTED, evidence_ref="tool:attempt")
    lifecycle.transition(ActionState.COMPLETED, evidence_ref="tool:return")
    lifecycle.transition(ActionState.VERIFIED, evidence_ref="verify:readback")
    snapshot = lifecycle.snapshot()

    assert snapshot.state is ActionState.VERIFIED
    assert [item.state for item in snapshot.transitions] == [
        ActionState.PLANNED,
        ActionState.AUTHORIZED,
        ActionState.ATTEMPTED,
        ActionState.COMPLETED,
        ActionState.VERIFIED,
    ]
    assert snapshot.correlation == correlation
    assert snapshot.correlation.action_id == "action-123"
    assert snapshot.correlation.plan_id == "plan-9"
    assert snapshot.correlation.attempt_id
    assert snapshot.correlation.verification_id
    assert snapshot.correlation.audit_id
    assert snapshot.correlation.user_result_id
    with pytest.raises(FrozenInstanceError):
        snapshot.correlation.action_id = "rewritten"  # type: ignore[misc]


def test_invalid_or_terminal_transition_fails_closed() -> None:
    from rex.actions.lifecycle import ActionLifecycle, ActionState, InvalidActionTransition

    lifecycle = ActionLifecycle.create(action_id="action-invalid")
    with pytest.raises(InvalidActionTransition):
        lifecycle.transition(ActionState.VERIFIED)
    assert lifecycle.snapshot().state is ActionState.PLANNED

    lifecycle.transition(ActionState.FAILED, evidence_ref="policy:denied")
    with pytest.raises(InvalidActionTransition):
        lifecycle.transition(ActionState.VERIFIED)
    assert lifecycle.snapshot().state is ActionState.FAILED


def test_legacy_verified_and_unverified_outcomes_adapt_without_losing_truth() -> None:
    from rex.actions.lifecycle import ActionState, lifecycle_from_legacy_status

    verified = lifecycle_from_legacy_status("verified", action_id="verify-1")
    uncertain = lifecycle_from_legacy_status("attempted_unverified", action_id="uncertain-1")

    assert verified.state is ActionState.VERIFIED
    assert [item.state.value for item in verified.transitions] == [
        "planned",
        "authorized",
        "attempted",
        "completed",
        "verified",
    ]
    assert uncertain.state is ActionState.UNVERIFIED
    assert [item.state.value for item in uncertain.transitions] == [
        "planned",
        "authorized",
        "attempted",
        "unverified",
    ]


def test_user_facing_success_wording_derives_from_lifecycle_state() -> None:
    from rex.actions.lifecycle import lifecycle_from_legacy_status, render_action_outcome

    verified = lifecycle_from_legacy_status("verified", action_id="verified-wording")
    uncertain = lifecycle_from_legacy_status("attempted_unverified", action_id="uncertain-wording")
    failed = lifecycle_from_legacy_status("failed", action_id="failed-wording")

    assert render_action_outcome(verified, "front door lock").startswith("Verified ")
    assert "could not verify" in render_action_outcome(uncertain, "front door lock")
    assert "failed" in render_action_outcome(failed, "front door lock").lower()
    assert "Verified" not in render_action_outcome(uncertain, "front door lock")


def test_tool_execution_attaches_canonical_lifecycle_to_verified_and_uncertain_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rex.actions.lifecycle import ActionState
    from rex.tools.execution import ToolExecutionLifecycle

    monkeypatch.setattr(
        "rex.tools.execution.get_audit_logger", lambda: SimpleNamespace(log=lambda _e: None)
    )

    verified_tool = SimpleNamespace(
        name="lock_door",
        handler=lambda **_kwargs: {"accepted": True},
        operation="mutation",
        risk="safe",
        requires_identity=True,
        required_args=(),
        required_permissions=(),
        verifier=lambda _args, _output: True,
    )
    uncertain_tool = SimpleNamespace(
        name="send_command",
        handler=lambda **_kwargs: {"accepted": True},
        operation="mutation",
        risk="safe",
        requires_identity=True,
        required_args=(),
        required_permissions=(),
        verifier=None,
    )

    lifecycle = ToolExecutionLifecycle()
    verified = lifecycle.execute(
        verified_tool,
        {},
        {"user_id": "james", "request_id": "tool-verified"},
    )
    uncertain = lifecycle.execute(
        uncertain_tool,
        {},
        {"user_id": "james", "request_id": "tool-uncertain"},
    )

    assert verified.lifecycle is not None
    assert verified.lifecycle.state is ActionState.VERIFIED
    assert verified.lifecycle.correlation.action_id == verified.request_id == "tool-verified"
    assert uncertain.lifecycle is not None
    assert uncertain.lifecycle.state is ActionState.UNVERIFIED
    assert uncertain.success is False


def test_tool_audit_record_shares_plan_action_and_lifecycle_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rex.tools.execution import ToolExecutionLifecycle

    entries = []
    monkeypatch.setattr(
        "rex.tools.execution.get_audit_logger",
        lambda: SimpleNamespace(log=entries.append),
    )
    tool = SimpleNamespace(
        name="lock_door",
        handler=lambda **_kwargs: {"accepted": True},
        operation="mutation",
        risk="safe",
        requires_identity=True,
        required_args=(),
        required_permissions=(),
        verifier=lambda _args, _output: True,
    )

    result = ToolExecutionLifecycle().execute(
        tool,
        {},
        {
            "user_id": "james",
            "request_id": "correlated-action",
            "plan_id": "plan-42",
        },
    )

    assert result.lifecycle is not None
    assert len(entries) == 1
    entry = entries[0]
    correlation = result.lifecycle.correlation
    assert entry.action_id == correlation.action_id == result.request_id
    assert entry.task_id == correlation.plan_id == "plan-42"
    assert entry.tool_result is not None
    audit_lifecycle = entry.tool_result["lifecycle"]
    assert audit_lifecycle["correlation"]["attempt_id"] == correlation.attempt_id
    assert audit_lifecycle["correlation"]["verification_id"] == correlation.verification_id
    assert audit_lifecycle["correlation"]["audit_id"] == correlation.audit_id
    assert audit_lifecycle["correlation"]["user_result_id"] == correlation.user_result_id


def test_mutation_cannot_self_promote_to_verified_without_independent_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rex.actions.lifecycle import ActionState
    from rex.tools.execution import ToolExecutionLifecycle

    monkeypatch.setattr(
        "rex.tools.execution.get_audit_logger",
        lambda: SimpleNamespace(log=lambda _event: None),
    )
    tool = SimpleNamespace(
        name="unlock_door",
        handler=lambda **_kwargs: {"status": "verified", "detail": "Plugin claims success."},
        operation="mutation",
        risk="safe",
        requires_identity=True,
        required_args=(),
        required_permissions=(),
        verifier=lambda _args, _output: False,
    )

    result = ToolExecutionLifecycle().execute(
        tool,
        {},
        {"user_id": "james", "request_id": "self-claimed-verified"},
    )

    assert result.lifecycle is not None
    assert result.lifecycle.state is ActionState.UNVERIFIED
    assert result.success is False
    assert "Verified" not in result.detail
    assert "could not verify" in result.detail


def test_workflow_step_does_not_promote_unverified_tool_result_to_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from rex.actions.lifecycle import lifecycle_from_legacy_status
    from rex.contracts import ToolCall
    from rex.workflow import Workflow, WorkflowStep
    from rex.workflow_runner import WorkflowRunner

    step = WorkflowStep(
        description="attempt an uncertain mutation",
        tool_call=ToolCall(tool="send_command", args={}),
    )
    workflow = Workflow(title="Unverified mutation", steps=[step])
    policy = SimpleNamespace(
        decide=lambda _call, _metadata: SimpleNamespace(
            denied=False, requires_approval=False, reason="allowed"
        )
    )
    lifecycle = lifecycle_from_legacy_status(
        "attempted_unverified", action_id="workflow-action", plan_id=workflow.workflow_id
    )
    monkeypatch.setattr(
        "rex.workflow_runner.execute_tool",
        lambda *_args, **_kwargs: {
            "success": False,
            "status": "attempted_unverified",
            "detail": "Mutation outcome could not be verified.",
            "lifecycle": lifecycle.to_dict(),
        },
    )

    result = WorkflowRunner(
        workflow,
        policy_engine=policy,
        workflow_dir=tmp_path / "workflows",
        approval_dir=tmp_path / "approvals",
    )._execute_step(step)

    assert result.success is False
    assert result.output is not None
    assert result.output["lifecycle"]["state"] == "unverified"
    assert result.error is not None
    assert "could not be verified" in result.error.lower()


def test_home_assistant_result_exports_canonical_lifecycle_correlation() -> None:
    from rex.ha.mutation_service import HAMutationResult, HAOutcome, HARisk

    result = HAMutationResult(
        status=HAOutcome.VERIFIED,
        detail="Verified light state.",
        entity_id="light.kitchen",
        domain="light",
        service="turn_on",
        request_id="ha-123",
        risk=HARisk.SAFE,
    )

    payload = result.to_dict()
    assert payload["lifecycle"]["state"] == "verified"
    assert payload["lifecycle"]["correlation"]["action_id"] == "ha-123"
    assert payload["lifecycle"]["correlation"]["verification_id"]


def test_openclaw_audit_record_shares_canonical_action_and_plan_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rex.openclaw.tool_executor import execute_tool

    entries = []
    monkeypatch.setattr(
        "rex.openclaw.tool_executor.get_audit_logger",
        lambda: SimpleNamespace(log=entries.append),
    )

    result = execute_tool(
        {"tool": "time_now", "args": {"location": "London"}},
        {"user_id": "james"},
        skip_policy_check=True,
        skip_credential_check=True,
        task_id="workflow-plan",
        requested_by="james",
    )

    correlation = result["lifecycle"]["correlation"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.action_id == correlation["action_id"]
    assert entry.task_id == correlation["plan_id"] == "workflow-plan"
    assert entry.tool_result is not None
    assert entry.tool_result["lifecycle"]["correlation"]["audit_id"] == correlation["audit_id"]
    assert (
        entry.tool_result["lifecycle"]["correlation"]["user_result_id"]
        == correlation["user_result_id"]
    )


def test_openclaw_builtin_result_exports_canonical_lifecycle() -> None:
    from rex.openclaw.tool_executor import execute_tool

    result = execute_tool(
        {"tool": "time_now", "args": {"location": "London"}},
        {"user_id": "james"},
        skip_policy_check=True,
        skip_credential_check=True,
        skip_audit_log=True,
        requested_by="james",
    )

    assert result.get("lifecycle", {}).get("state") == "completed"
    correlation = result["lifecycle"]["correlation"]
    assert correlation["action_id"]
    assert correlation["attempt_id"]
    assert correlation["audit_id"]
