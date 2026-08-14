"""Rex memories bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Every command operates on behalf of a resolved user identity (US-303
per-user isolation):

- An explicit ``"user"`` field in the payload takes precedence (it is
  validated; malformed IDs fail closed).
- Otherwise the active user is resolved through the standard identity
  chain (``rex identify`` session state, then ``runtime.active_user`` /
  ``runtime.user_id`` in ``config/rex_config.json``).
- If no user can be resolved, the command fails closed with an error.
  Single-user setups keep working by explicitly selecting the ``default``
  profile (``rex identify --user default`` or ``"user": "default"``).

Commands:
  {"command": "list", "user"?: "<user_id>"}
    -> {"ok": true, "memories": [...]}    (only the resolved user's memories)

  {"command": "add", "user"?: "<user_id>", "data": {text, category}}
    -> {"ok": true, "memory": {...}}      (owned by the resolved user)

  {"command": "update", "user"?: "<user_id>", "id": "<entry_id>", "data": {text, category}}
    -> {"ok": true, "memory": {...}}      (ownership enforced)

  {"command": "delete", "user"?: "<user_id>", "id": "<entry_id>"}
    -> {"ok": true}                       (ownership enforced)

Memory format (GUI):
  {id, text, category, createdAt (ISO), updatedAt (ISO)}
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

from rex.bridge_utils import bridge_error_response

_NO_USER_ERROR = (
    "No active user for memories. Set one with 'rex identify --user <id>' "
    "or include a 'user' field in the request."
)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _resolve_user(payload: dict[str, Any]) -> str | None:
    """Resolve the requesting user, failing closed on missing/invalid identity.

    Never falls back to ``"default"`` or any other profile implicitly.
    """
    from rex.identity import resolve_active_user

    if payload.get("data_scope") != "private":
        return None
    explicit = str(payload.get("user") or "").strip() or None
    try:
        config: dict[str, Any] | None
        try:
            from rex.config_manager import load_config

            config = load_config()
        except Exception:
            config = None
        return resolve_active_user(explicit, config=config)
    except ValueError:
        # Malformed explicit user ID: fail closed, no fallback.
        return None


def _entry_to_gui(entry: Any) -> dict[str, Any]:
    """Convert a LongTermMemory MemoryEntry to the GUI Memory dict format."""
    content: dict[str, Any] = entry.content or {}
    text = str(content.get("text") or "")
    updated_at = content.get("updated_at") or None

    created_at_str: str
    if hasattr(entry.created_at, "isoformat"):
        created_at_str = entry.created_at.isoformat()
    else:
        created_at_str = str(entry.created_at)

    updated_at_str: str = updated_at if isinstance(updated_at, str) else created_at_str

    return {
        "id": entry.entry_id,
        "text": text,
        "category": entry.category,
        "createdAt": created_at_str,
        "updatedAt": updated_at_str,
    }


def _procedure_to_gui(record: Any) -> dict[str, Any]:
    """Convert a guarded procedure record to a renderer-safe metadata view."""
    return {
        "id": record.procedure_id,
        "name": record.name,
        "description": record.description,
        "ownerId": record.owner_id,
        "scope": record.scope.value,
        "status": record.status.value,
        "capabilities": list(record.capabilities),
        "requiredPermissions": list(record.required_permissions),
        "operation": record.operation.value,
        "risk": record.risk.value,
        "version": record.version,
        "dependencyFingerprint": record.dependency_fingerprint,
        "successCount": record.success_count,
        "failureCount": record.failure_count,
        "lastValidatedAt": record.last_validated_at.isoformat(),
        "expiresAt": record.expires_at.isoformat() if record.expires_at else None,
        "disabledReason": record.disabled_reason,
        "approvalRequired": record.approval_required,
        "approvedBy": record.approved_by,
        "createdAt": record.created_at.isoformat(),
        "provenance": {
            "actionId": record.provenance.action_id,
            "planId": record.provenance.plan_id,
            "verificationId": record.provenance.verification_id,
            "auditId": record.provenance.audit_id,
        },
        "auditHistory": [
            {
                "timestamp": event.timestamp.isoformat(),
                "event": event.event,
                "actorUserId": event.actor_user_id,
                "reason": event.reason,
                "evidenceRef": event.evidence_ref,
            }
            for event in record.audit_history
        ],
    }


def _handle_list(user_id: str) -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    ltm = get_long_term_memory(user_id=user_id)
    entries = ltm.search()
    return {"ok": True, "memories": [_entry_to_gui(e) for e in entries]}


def _handle_add(user_id: str, data: dict[str, Any]) -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    text = str(data.get("text") or "").strip()
    category = str(data.get("category") or "general").strip() or "general"

    if not text:
        return {"ok": False, "error": "Memory text is required"}

    now = _utc_now_iso()
    ltm = get_long_term_memory(user_id=user_id)
    entry = ltm.add_entry(
        category=category,
        content={"text": text, "updated_at": now},
    )
    return {"ok": True, "memory": _entry_to_gui(entry)}


def _handle_update(user_id: str, entry_id: str, data: dict[str, Any]) -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    text = str(data.get("text") or "").strip()
    category = str(data.get("category") or "").strip()

    if not text:
        return {"ok": False, "error": "Memory text is required"}

    ltm = get_long_term_memory(user_id=user_id)
    entry = ltm.get_entry(entry_id)
    if entry is None:
        return {"ok": False, "error": f"Memory {entry_id!r} not found"}

    entry.content["text"] = text
    entry.content["updated_at"] = _utc_now_iso()
    if category:
        entry.category = category
    ltm._save()  # noqa: SLF001

    return {"ok": True, "memory": _entry_to_gui(entry)}


def _handle_delete(user_id: str, entry_id: str) -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    ltm = get_long_term_memory(user_id=user_id)
    deleted = ltm.forget(entry_id)
    if not deleted:
        return {"ok": False, "error": f"Memory {entry_id!r} not found"}
    return {"ok": True}


def _handle_procedures_list(user_id: str) -> dict[str, Any]:
    from rex.procedural_memory import ProceduralMemory

    memory = ProceduralMemory()
    records = memory.list(requester_user_id=user_id, include_household=True)
    return {"ok": True, "procedures": [_procedure_to_gui(record) for record in records]}


def _handle_procedure_action(
    user_id: str, procedure_id: str, command: str, *, confirmed: bool = False
) -> dict[str, Any]:
    from rex.procedural_memory import ProceduralMemory

    if not procedure_id:
        return {"ok": False, "error": "Procedure id is required"}
    memory = ProceduralMemory()
    if command == "procedures-approve":
        record = memory.approve(
            procedure_id,
            requester_user_id=user_id,
            approver_user_id=user_id,
            confirmed=confirmed,
        )
    elif command == "procedures-disable":
        record = memory.disable(
            procedure_id, requester_user_id=user_id, reason="user_disabled_from_gui"
        )
    elif command == "procedures-revoke":
        record = memory.revoke(
            procedure_id, requester_user_id=user_id, reason="user_revoked_from_gui"
        )
    elif command == "procedures-delete":
        memory.delete(procedure_id, requester_user_id=user_id)
        return {"ok": True}
    else:
        return {"ok": False, "error": f"Unknown procedure command: {command!r}"}
    return {"ok": True, "procedure": _procedure_to_gui(record)}


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("action") or payload.get("command") or "")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        supported = (
            "list",
            "add",
            "update",
            "delete",
            "procedures-list",
            "procedures-approve",
            "procedures-disable",
            "procedures-revoke",
            "procedures-delete",
        )
        if command in supported:
            user_id = _resolve_user(payload)
            if user_id is None:
                result: dict[str, Any] = {"ok": False, "error": _NO_USER_ERROR}
            elif command == "list":
                result = _handle_list(user_id)
            elif command == "add":
                result = _handle_add(user_id, dict(payload.get("data") or {}))
            elif command == "update":
                result = _handle_update(
                    user_id, str(payload.get("id") or ""), dict(payload.get("data") or {})
                )
            elif command == "delete":
                result = _handle_delete(user_id, str(payload.get("id") or ""))
            elif command == "procedures-list":
                result = _handle_procedures_list(user_id)
            else:
                result = _handle_procedure_action(
                    user_id,
                    str(payload.get("id") or ""),
                    command,
                    confirmed=payload.get("confirmed") is True,
                )
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = bridge_error_response(exc)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
