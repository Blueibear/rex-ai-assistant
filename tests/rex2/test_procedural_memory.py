"""Acceptance tests for US-112 guarded procedural experience memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rex.actions.lifecycle import lifecycle_from_legacy_status
from rex.procedural_memory import (
    ProceduralMemory,
    ProcedureDefinition,
    ProcedurePromotionError,
    ProcedureRevalidationPolicy,
    ProcedureScope,
    ProcedureStatus,
)
from rex.tools.execution import ToolOperation, ToolRisk


def _definition(**overrides) -> ProcedureDefinition:
    values = {
        "name": "Check the kitchen status",
        "description": "Read the kitchen entity state and summarize it.",
        "capabilities": ("home_assistant.read_state",),
        "required_permissions": ("home_assistant.read",),
        "operation": ToolOperation.READ,
        "risk": ToolRisk.SAFE,
        "version": "1",
        "dependency_fingerprint": "ha-schema-v1",
        "steps": ("home_assistant.read_state",),
        "revalidation": ProcedureRevalidationPolicy(
            revalidate_after_seconds=3600,
            expires_after_seconds=86400,
            failure_threshold=2,
        ),
    }
    values.update(overrides)
    return ProcedureDefinition(**values)


def _verified(action_id: str = "action-1"):
    return lifecycle_from_legacy_status("verified", action_id=action_id, plan_id="plan-1")


def _execution_context(
    *, capability: str = "home_assistant.read_state", permission: str = "home_assistant.read"
) -> dict:
    return {
        "dependency_fingerprint": "ha-schema-v1",
        "version": "1",
        "available_capabilities": {capability},
        "granted_permissions": {permission},
    }


def test_only_verified_outcomes_can_create_procedure_candidates(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    for status in ("completed", "attempted_unverified", "failed", "confirmation_required"):
        outcome = lifecycle_from_legacy_status(status, action_id=f"action-{status}")
        with pytest.raises(ProcedurePromotionError, match="verified"):
            memory.learn_from_verified_outcome(
                outcome,
                _definition(),
                owner_id="james",
                scope=ProcedureScope.USER,
            )

    learned = memory.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )

    assert learned.status is ProcedureStatus.ACTIVE
    assert learned.provenance.action_id == "action-1"
    assert learned.provenance.verification_id == "verify:action-1"
    assert learned.provenance.audit_id == "audit:action-1"


def test_procedure_records_required_provenance_policy_and_counters(tmp_path) -> None:
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    memory = ProceduralMemory(base_dir=tmp_path, clock=lambda: now)

    learned = memory.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )

    assert learned.owner_id == "james"
    assert learned.scope is ProcedureScope.USER
    assert learned.capabilities == ("home_assistant.read_state",)
    assert learned.required_permissions == ("home_assistant.read",)
    assert learned.operation is ToolOperation.READ
    assert learned.risk is ToolRisk.SAFE
    assert learned.version == "1"
    assert learned.dependency_fingerprint == "ha-schema-v1"
    assert learned.success_count == 1
    assert learned.failure_count == 0
    assert learned.last_validated_at == now
    assert learned.revalidation.failure_threshold == 2
    assert learned.expires_at == now + timedelta(days=1)
    assert learned.audit_history[-1].event == "promoted_from_verified_outcome"


def test_mutation_and_elevated_risk_require_explicit_human_approval(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    definition = _definition(
        operation=ToolOperation.MUTATION,
        risk=ToolRisk.SENSITIVE,
        capabilities=("home_assistant.call_service",),
        required_permissions=("home_assistant.write",),
        steps=("home_assistant.call_service",),
    )

    learned = memory.learn_from_verified_outcome(
        _verified(), definition, owner_id="james", scope=ProcedureScope.USER
    )

    assert learned.status is ProcedureStatus.PENDING_APPROVAL
    assert learned.approval_required is True
    assert not memory.can_execute(
        learned.procedure_id,
        requester_user_id="james",
        **_execution_context(
            capability="home_assistant.call_service", permission="home_assistant.write"
        ),
    )

    with pytest.raises(PermissionError, match="explicit human approval"):
        memory.approve(
            learned.procedure_id,
            requester_user_id="james",
            approver_user_id="james",
            confirmed=False,
        )

    approved = memory.approve(
        learned.procedure_id,
        requester_user_id="james",
        approver_user_id="james",
        confirmed=True,
    )
    assert approved.status is ProcedureStatus.ACTIVE
    assert approved.approved_by == "james"
    assert memory.can_execute(
        learned.procedure_id,
        requester_user_id="james",
        **_execution_context(
            capability="home_assistant.call_service", permission="home_assistant.write"
        ),
    )


def test_prohibited_procedure_can_never_be_activated(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    definition = _definition(risk=ToolRisk.PROHIBITED)

    learned = memory.learn_from_verified_outcome(
        _verified(), definition, owner_id="james", scope=ProcedureScope.USER
    )

    assert learned.status is ProcedureStatus.REVOKED
    assert not memory.can_execute(
        learned.procedure_id,
        requester_user_id="james",
        **_execution_context(),
    )
    with pytest.raises(PermissionError, match="prohibited"):
        memory.approve(
            learned.procedure_id,
            requester_user_id="james",
            approver_user_id="james",
            confirmed=True,
        )


def test_repeated_failure_disables_without_erasing_audit_history(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )
    initial_events = len(learned.audit_history)

    first = memory.record_execution_outcome(
        learned.procedure_id,
        requester_user_id="james",
        outcome=lifecycle_from_legacy_status("failed", action_id="retry-1"),
        dependency_fingerprint="ha-schema-v1",
        version="1",
    )
    assert first.status is ProcedureStatus.ACTIVE
    second = memory.record_execution_outcome(
        learned.procedure_id,
        requester_user_id="james",
        outcome=lifecycle_from_legacy_status("failed", action_id="retry-2"),
        dependency_fingerprint="ha-schema-v1",
        version="1",
    )

    assert second.status is ProcedureStatus.DISABLED
    assert second.disabled_reason == "repeated_failure"
    assert second.failure_count == 2
    assert len(second.audit_history) >= initial_events + 2
    assert any(event.event == "disabled_repeated_failure" for event in second.audit_history)


def test_dependency_drift_disables_pending_revalidation_and_keeps_history(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )

    checked = memory.validate_for_execution(
        learned.procedure_id,
        requester_user_id="james",
        dependency_fingerprint="ha-schema-v2",
        version="1",
    )

    assert checked.status is ProcedureStatus.DISABLED
    assert checked.disabled_reason == "dependency_drift"
    assert any(event.event == "disabled_dependency_drift" for event in checked.audit_history)
    assert learned.provenance.action_id == checked.provenance.action_id


def test_expired_or_due_for_revalidation_procedure_is_not_executable(tmp_path) -> None:
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    current = [now]
    memory = ProceduralMemory(base_dir=tmp_path, clock=lambda: current[0])
    learned = memory.learn_from_verified_outcome(
        _verified(),
        _definition(
            revalidation=ProcedureRevalidationPolicy(
                revalidate_after_seconds=60,
                expires_after_seconds=120,
                failure_threshold=2,
            )
        ),
        owner_id="james",
        scope=ProcedureScope.USER,
    )
    assert memory.can_execute(
        learned.procedure_id, requester_user_id="james", **_execution_context()
    )

    current[0] = now + timedelta(seconds=61)
    checked = memory.validate_for_execution(
        learned.procedure_id,
        requester_user_id="james",
        dependency_fingerprint="ha-schema-v1",
        version="1",
    )
    assert checked.status is ProcedureStatus.DISABLED
    assert checked.disabled_reason == "revalidation_due"


def test_active_procedure_never_grants_permissions_or_capabilities(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )

    assert not memory.can_execute(
        learned.procedure_id,
        requester_user_id="james",
        dependency_fingerprint="ha-schema-v1",
        version="1",
        available_capabilities={"home_assistant.read_state"},
        granted_permissions=set(),
    )
    assert not memory.can_execute(
        learned.procedure_id,
        requester_user_id="james",
        dependency_fingerprint="ha-schema-v1",
        version="1",
        available_capabilities=set(),
        granted_permissions={"home_assistant.read"},
    )
    assert memory.can_execute(
        learned.procedure_id, requester_user_id="james", **_execution_context()
    )


def test_risky_revalidation_cannot_bypass_initial_human_approval(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        _verified(),
        _definition(operation=ToolOperation.MUTATION, risk=ToolRisk.SENSITIVE),
        owner_id="james",
        scope=ProcedureScope.USER,
    )
    assert learned.status is ProcedureStatus.PENDING_APPROVAL

    revalidated = memory.revalidate(
        learned.procedure_id,
        requester_user_id="james",
        outcome=_verified("revalidation-action"),
        dependency_fingerprint="ha-schema-v1",
        version="1",
    )

    assert revalidated.status is ProcedureStatus.PENDING_APPROVAL
    assert revalidated.approved_by is None


def test_procedures_persist_across_store_restart(tmp_path) -> None:
    first = ProceduralMemory(base_dir=tmp_path)
    learned = first.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )

    restarted = ProceduralMemory(base_dir=tmp_path)
    loaded = restarted.get(learned.procedure_id, requester_user_id="james")

    assert loaded == learned
    assert loaded.provenance.verification_id == "verify:action-1"


def test_version_drift_disables_pending_revalidation(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )

    checked = memory.validate_for_execution(
        learned.procedure_id,
        requester_user_id="james",
        dependency_fingerprint="ha-schema-v1",
        version="2",
    )

    assert checked.status is ProcedureStatus.DISABLED
    assert checked.disabled_reason == "version_drift"
    assert any(event.event == "disabled_version_drift" for event in checked.audit_history)


def test_procedure_definition_rejects_executable_payload_fields() -> None:
    from pydantic import ValidationError

    values = _definition().model_dump()
    values["python_code"] = "import os"
    with pytest.raises(ValidationError, match="python_code"):
        ProcedureDefinition.model_validate(values)


def test_disabled_procedure_requires_new_verified_evidence_for_revalidation(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        _verified(), _definition(), owner_id="james", scope=ProcedureScope.USER
    )
    disabled = memory.validate_for_execution(
        learned.procedure_id,
        requester_user_id="james",
        dependency_fingerprint="ha-schema-v2",
        version="1",
    )
    assert disabled.status is ProcedureStatus.DISABLED

    with pytest.raises(ProcedurePromotionError, match="verified"):
        memory.revalidate(
            learned.procedure_id,
            requester_user_id="james",
            outcome=lifecycle_from_legacy_status("completed", action_id="recheck"),
            dependency_fingerprint="ha-schema-v2",
            version="1",
        )

    revalidated = memory.revalidate(
        learned.procedure_id,
        requester_user_id="james",
        outcome=_verified("recheck-verified"),
        dependency_fingerprint="ha-schema-v2",
        version="1",
    )
    assert revalidated.status is ProcedureStatus.ACTIVE
    assert revalidated.dependency_fingerprint == "ha-schema-v2"
    assert revalidated.success_count == 2
    assert any(event.event == "disabled_dependency_drift" for event in revalidated.audit_history)
    assert any(event.event == "revalidated" for event in revalidated.audit_history)
