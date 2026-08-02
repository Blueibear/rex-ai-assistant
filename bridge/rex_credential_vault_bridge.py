"""Rex credential vault bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout,
following the same convention as every other ``bridge/rex_*_bridge.py``
script. This is the *only* code path through which Electron ever touches a
secret value - the vault's cryptography (Windows DPAPI) lives entirely in
``rex.credential_vault`` on the Python side; Electron never performs crypto
itself.

Every operation requires ``request_user_id``, an explicit scope, and exact
caller-expected integration/account/slot context. User scope is always bound
to the validated requester; no renderer-supplied owner is accepted. Set may
omit ``key`` to generate a new opaque reference and returns that reference.

  {"action": "get", "key": "...", "scope": ..., "request_user_id": ...,
   "integration": ..., "account": ..., "slot": ...}
    -> {"ok": true, "value": "<secret>" | null}

  {"action": "has", "key": "...", "scope"?: ..., "user_id"?: ...}
    -> {"ok": true, "has": true|false}

  {"action": "delete", "key": "...", "scope"?: ..., "user_id"?: ...}
    -> {"ok": true, "deleted": true|false}

  {"action": "list", "scope"?: ..., "user_id"?: ...}
    -> {"ok": true, "entries": [{"key", "integration", "account", "scope", "owner",
                                  "created_at", "updated_at"}, ...]}

Failure responses contain no traceback, exception repr, or secret-derived
output.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any


def _safe_error(exc: Exception) -> dict[str, Any]:
    """Return a categorized credential vault failure with no secret-derived content.

    Never includes ``str(exc)``, a traceback, or any submitted key/scope/user
    value - only a fixed message chosen by exception type.
    """
    from rex.bridge_utils import bridge_safe_error_response
    from rex.credential_vault import VaultCorruptedError, VaultUnavailableError

    return bridge_safe_error_response(
        exc,
        messages={
            VaultUnavailableError: "Credential vault is unavailable",
            VaultCorruptedError: "Credential vault data is invalid",
            ValueError: "Credential vault request is invalid",
            PermissionError: "Credential vault request is not permitted",
        },
        default="Credential vault operation failed",
    )


def _get_vault(payload: dict[str, Any]):
    from rex.credential_vault import get_credential_vault
    from rex.identity import validate_user_id

    requester = validate_user_id(str(payload.get("request_user_id") or ""))
    scope = str(payload.get("scope") or "")
    if scope not in {"household", "user"}:
        raise ValueError("A valid credential scope is required")
    user_id = requester if scope == "user" else None
    return get_credential_vault(scope=scope, user_id=user_id)


def _require_key(payload: dict[str, Any]) -> str:
    key = str(payload.get("key") or "").strip()
    if not key:
        raise ValueError("'key' is required")
    return key


def _require_context(payload: dict[str, Any]) -> tuple[str, str | None, str]:
    integration = str(payload.get("integration") or "").strip()
    slot = str(payload.get("slot") or "").strip()
    account_value = payload.get("account")
    account = str(account_value).strip() if isinstance(account_value, str) else None
    if not integration or not slot:
        raise ValueError("'integration' and 'slot' are required")
    return integration, account or None, slot


def _handle_set(payload: dict[str, Any]) -> dict[str, Any]:
    from rex.credential_vault import generate_credential_ref

    raw_key = str(payload.get("key") or "").strip()
    key = raw_key or generate_credential_ref()
    value = payload.get("value")
    if not isinstance(value, str) or not value:
        raise ValueError("'value' is required")
    integration, account, slot = _require_context(payload)
    vault = _get_vault(payload)
    vault.set_secret(key, value, integration=integration, account=account, slot=slot)
    try:
        verified = vault.get_secret(key, integration=integration, account=account, slot=slot)
        if verified != value:
            raise RuntimeError("Credential vault readback verification failed")
    except Exception:
        try:
            vault.delete_secret(key, integration=integration, account=account, slot=slot)
        except (
            Exception
        ):  # noqa: B110 - best-effort rollback cleanup must not hide the primary error
            pass
        raise
    return {"ok": True, "ref": key}


def _handle_get(payload: dict[str, Any]) -> dict[str, Any]:
    key = _require_key(payload)
    integration, account, slot = _require_context(payload)
    vault = _get_vault(payload)
    return {
        "ok": True,
        "value": vault.get_secret(key, integration=integration, account=account, slot=slot),
    }


def _handle_has(payload: dict[str, Any]) -> dict[str, Any]:
    key = _require_key(payload)
    integration, account, slot = _require_context(payload)
    vault = _get_vault(payload)
    return {
        "ok": True,
        "has": vault.has_secret(key, integration=integration, account=account, slot=slot),
    }


def _handle_delete(payload: dict[str, Any]) -> dict[str, Any]:
    key = _require_key(payload)
    integration, account, slot = _require_context(payload)
    vault = _get_vault(payload)
    return {
        "ok": True,
        "deleted": vault.delete_secret(key, integration=integration, account=account, slot=slot),
    }


def _handle_list(payload: dict[str, Any]) -> dict[str, Any]:
    vault = _get_vault(payload)
    entries = [asdict(entry) for entry in vault.list_entries()]
    return {"ok": True, "entries": entries}


_HANDLERS = {
    "set": _handle_set,
    "get": _handle_get,
    "has": _handle_has,
    "delete": _handle_delete,
    "list": _handle_list,
}


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        print(json.dumps(_safe_error(exc)), flush=True)
        sys.exit(1)

    action = str(payload.get("action") or "")
    handler = _HANDLERS.get(action)
    if handler is None:
        print(
            json.dumps({"ok": False, "error": f"Unknown action: {action!r}"}),
            flush=True,
        )
        sys.exit(1)

    try:
        result = handler(payload)
    except Exception as exc:
        print(json.dumps(_safe_error(exc)), flush=True)
        sys.exit(1)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
