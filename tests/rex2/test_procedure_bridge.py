"""Bridge-level proof for guarded procedural memory management."""

from __future__ import annotations

import importlib

import pytest

from rex.actions.lifecycle import lifecycle_from_legacy_status
from rex.memory import set_long_term_memory
from rex.procedural_memory import (
    ProceduralMemory,
    ProcedureDefinition,
    ProcedureScope,
    ProcedureStatus,
)
from rex.tools.execution import ToolOperation, ToolRisk


def _bridge():
    return importlib.import_module("rex_memories_bridge")


def _definition(*, mutation: bool = False) -> ProcedureDefinition:
    return ProcedureDefinition(
        name="Check kitchen" if not mutation else "Set kitchen light",
        description="Bridge acceptance fixture.",
        capabilities=(
            "home_assistant.read_state" if not mutation else "home_assistant.call_service",
        ),
        required_permissions=("home_assistant.read" if not mutation else "home_assistant.write",),
        operation=ToolOperation.MUTATION if mutation else ToolOperation.READ,
        risk=ToolRisk.SENSITIVE if mutation else ToolRisk.SAFE,
        version="1",
        dependency_fingerprint="ha-v1",
        steps=("home_assistant.read_state" if not mutation else "home_assistant.call_service",),
    )


@pytest.fixture
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("ASKREX_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("ASKREX_MEMORY_DIR", str(tmp_path / "Memory"))
    for user in ("james", "cole"):
        set_long_term_memory(None, user_id=user)
    yield tmp_path
    for user in ("james", "cole"):
        set_long_term_memory(None, user_id=user)


def test_normal_memory_add_cannot_create_executable_procedure(isolated_runtime) -> None:
    bridge = _bridge()
    procedures = ProceduralMemory()

    before = bridge._handle_procedures_list("james")
    assert before == {"ok": True, "procedures": []}

    added = bridge._handle_add("james", {"text": "I like dark mode", "category": "preferences"})
    assert added["ok"] is True

    after = bridge._handle_procedures_list("james")
    assert after == {"ok": True, "procedures": []}
    assert procedures.list(requester_user_id="james") == []


def test_bridge_lists_only_requester_private_procedures_plus_explicit_household(
    isolated_runtime,
) -> None:
    bridge = _bridge()
    memory = ProceduralMemory()
    james = memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id="james-private"),
        _definition(),
        owner_id="james",
        scope=ProcedureScope.USER,
    )
    cole = memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id="cole-private"),
        _definition(),
        owner_id="cole",
        scope=ProcedureScope.USER,
    )
    shared = memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id="household-shared"),
        _definition(),
        owner_id="james",
        scope=ProcedureScope.HOUSEHOLD,
    )

    james_ids = {item["id"] for item in bridge._handle_procedures_list("james")["procedures"]}
    cole_ids = {item["id"] for item in bridge._handle_procedures_list("cole")["procedures"]}

    assert james_ids == {james.procedure_id, shared.procedure_id}
    assert cole_ids == {cole.procedure_id, shared.procedure_id}
    assert james.procedure_id not in cole_ids
    assert cole.procedure_id not in james_ids


def test_direct_bridge_approval_requires_trusted_confirmation_signal(isolated_runtime) -> None:
    bridge = _bridge()
    memory = ProceduralMemory()
    learned = memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id="mutation"),
        _definition(mutation=True),
        owner_id="james",
        scope=ProcedureScope.USER,
    )
    assert learned.status is ProcedureStatus.PENDING_APPROVAL

    with pytest.raises(PermissionError, match="explicit human approval"):
        bridge._handle_procedure_action(
            "james", learned.procedure_id, "procedures-approve", confirmed=False
        )

    approved = bridge._handle_procedure_action(
        "james", learned.procedure_id, "procedures-approve", confirmed=True
    )["procedure"]
    assert approved["status"] == "active"
    assert approved["approvedBy"] == "james"


def test_bridge_manage_commands_preserve_owner_boundary(isolated_runtime) -> None:
    bridge = _bridge()
    memory = ProceduralMemory()
    learned = memory.learn_from_verified_outcome(
        lifecycle_from_legacy_status("verified", action_id="manage"),
        _definition(),
        owner_id="james",
        scope=ProcedureScope.USER,
    )

    with pytest.raises(PermissionError, match="not accessible"):
        bridge._handle_procedure_action("cole", learned.procedure_id, "procedures-disable")

    disabled = bridge._handle_procedure_action("james", learned.procedure_id, "procedures-disable")[
        "procedure"
    ]
    assert disabled["status"] == "disabled"

    revoked = bridge._handle_procedure_action("james", learned.procedure_id, "procedures-revoke")[
        "procedure"
    ]
    assert revoked["status"] == "revoked"

    assert bridge._handle_procedure_action("james", learned.procedure_id, "procedures-delete") == {
        "ok": True
    }
    assert bridge._handle_procedures_list("james") == {"ok": True, "procedures": []}
