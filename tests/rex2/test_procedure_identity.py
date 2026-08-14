"""Identity/scope controls for US-112 procedural memory."""

from __future__ import annotations

import pytest

from rex.actions.lifecycle import lifecycle_from_legacy_status
from rex.procedural_memory import (
    ProceduralMemory,
    ProcedureDefinition,
    ProcedureScope,
    ProcedureStatus,
)
from rex.tools.execution import ToolOperation, ToolRisk


def _definition() -> ProcedureDefinition:
    return ProcedureDefinition(
        name="Check lights",
        description="Read light state.",
        capabilities=("home_assistant.read_state",),
        required_permissions=("home_assistant.read",),
        operation=ToolOperation.READ,
        risk=ToolRisk.SAFE,
        version="1",
        dependency_fingerprint="ha-v1",
        steps=("home_assistant.read_state",),
    )


def _learn(memory: ProceduralMemory, owner: str, action_id: str):
    return memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id=action_id),
        _definition(),
        owner_id=owner,
        scope=ProcedureScope.USER,
    )


def test_private_user_procedures_are_isolated(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    james = _learn(memory, "james", "james-action")
    cole = _learn(memory, "cole", "cole-action")

    assert [item.procedure_id for item in memory.list(requester_user_id="james")] == [
        james.procedure_id
    ]
    assert [item.procedure_id for item in memory.list(requester_user_id="cole")] == [
        cole.procedure_id
    ]
    with pytest.raises(PermissionError, match="not accessible"):
        memory.get(james.procedure_id, requester_user_id="cole")


def test_owner_can_disable_revoke_and_delete_private_procedure(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = _learn(memory, "james", "action-1")

    disabled = memory.disable(
        learned.procedure_id, requester_user_id="james", reason="user_disabled"
    )
    assert disabled.status is ProcedureStatus.DISABLED

    revoked = memory.revoke(
        learned.procedure_id, requester_user_id="james", reason="no_longer_trusted"
    )
    assert revoked.status is ProcedureStatus.REVOKED

    memory.delete(learned.procedure_id, requester_user_id="james")
    assert memory.list(requester_user_id="james") == []


def test_other_user_cannot_manage_private_procedure(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = _learn(memory, "james", "action-1")

    for operation in (
        lambda: memory.disable(learned.procedure_id, requester_user_id="cole"),
        lambda: memory.revoke(learned.procedure_id, requester_user_id="cole"),
        lambda: memory.delete(learned.procedure_id, requester_user_id="cole"),
        lambda: memory.record_execution_outcome(
            learned.procedure_id,
            requester_user_id="cole",
            outcome=lifecycle_from_legacy_status("verified", action_id="other-action"),
            dependency_fingerprint="ha-v1",
            version="1",
        ),
    ):
        with pytest.raises(PermissionError, match="not accessible"):
            operation()


def test_household_procedure_is_explicit_and_visible_to_validated_users(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id="shared-action"),
        _definition(),
        owner_id="james",
        scope=ProcedureScope.HOUSEHOLD,
    )

    assert learned.scope is ProcedureScope.HOUSEHOLD
    assert (
        memory.get(learned.procedure_id, requester_user_id="cole").procedure_id
        == learned.procedure_id
    )
    assert learned.procedure_id in {
        item.procedure_id for item in memory.list(requester_user_id="cole", include_household=True)
    }


def test_household_write_controls_stay_with_owner(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    learned = memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id="shared-action"),
        _definition(),
        owner_id="james",
        scope=ProcedureScope.HOUSEHOLD,
    )

    with pytest.raises(PermissionError, match="owner"):
        memory.disable(learned.procedure_id, requester_user_id="cole")
    assert (
        memory.get(learned.procedure_id, requester_user_id="cole").status is ProcedureStatus.ACTIVE
    )


def test_invalid_identity_fails_closed(tmp_path) -> None:
    memory = ProceduralMemory(base_dir=tmp_path)
    with pytest.raises(ValueError):
        memory.list(requester_user_id="../other")
    with pytest.raises(ValueError):
        memory.learn_from_verified_outcome(
            lifecycle_from_legacy_status("verified", action_id="action"),
            _definition(),
            owner_id="../other",
            scope=ProcedureScope.USER,
        )


def test_private_store_owner_tampering_fails_closed(tmp_path) -> None:
    import json

    from rex.procedural_memory import ProcedureStoreError

    memory = ProceduralMemory(base_dir=tmp_path)
    _learn(memory, "james", "action-1")
    path = tmp_path / "users" / "james" / "procedures.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["procedures"][0]["owner_id"] = "cole"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProcedureStoreError, match="owner"):
        memory.list(requester_user_id="james")
