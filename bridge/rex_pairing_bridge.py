"""Desktop-local pairing authority bridge for the Electron GUI (S5)."""

from __future__ import annotations

import json
import sys
from typing import Any

from rex.identity import validate_user_id
from rex.mobile_api.db import default_users_db_path, migrate_users_db
from rex.mobile_api.pairing import PairingAuthority, PairingError

_SAFE_ERROR = "Pairing operation failed."


def _require_desktop_context(payload: dict[str, Any]) -> tuple[str, str]:
    if payload.get("data_scope") != "private":
        raise PermissionError("Private desktop session required.")
    user_id = validate_user_id(str(payload.get("user") or ""))
    approver = str(payload.get("approver") or "").strip()
    if not approver:
        raise PermissionError("Desktop approver identity required.")
    return user_id, approver


def _grant_dict(grant: Any) -> dict[str, Any]:
    return {
        "grant_id": grant.grant_id,
        "device_id": grant.device_id,
        "desktop_id": grant.desktop_id,
        "user_id": grant.user_id,
        "version": grant.version,
        "scopes": list(grant.scopes),
        "created_at": grant.created_at,
        "expires_at": grant.expires_at,
        "revoked_at": grant.revoked_at,
    }


def dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    user_id, approver = _require_desktop_context(payload)
    db_path = default_users_db_path()
    migrate_users_db(db_path)
    authority = PairingAuthority(db_path)
    action = str(payload.get("action") or "")

    if action == "create_challenge":
        challenge = authority.create_challenge(
            user_id=user_id,
            scopes=payload.get("scopes") or ["chat.send", "chat.history.read", "voice.use"],
        )
        return {
            "ok": True,
            "challenge": {
                **challenge.qr_payload(),
                "created_at": challenge.created_at,
            },
        }
    if action == "list_pending":
        return {"ok": True, "requests": authority.list_pending()}
    if action == "approve":
        grant = authority.approve(
            str(payload.get("request_id") or ""),
            approved_by=approver,
        )
        return {"ok": True, "grant": _grant_dict(grant)}
    if action == "deny":
        authority.deny(
            str(payload.get("request_id") or ""),
            denied_by=approver,
            reason=str(payload.get("reason") or "denied_by_owner"),
        )
        return {"ok": True}
    if action == "list_devices":
        return {
            "ok": True,
            "desktop_id": authority.desktop_id(),
            "devices": authority.list_devices(),
        }
    if action == "revoke":
        revoked = authority.revoke_device(
            str(payload.get("device_id") or ""),
            revoked_by=approver,
            reason=str(payload.get("reason") or "revoked_by_owner"),
        )
        return {"ok": True, "revoked": revoked}
    raise PairingError("Unknown pairing action.")


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise PairingError("Request must be an object.")
        result = dispatch(payload)
    except (PairingError, PermissionError, ValueError):
        result = {"ok": False, "error": _SAFE_ERROR}
    except Exception:
        result = {"ok": False, "error": _SAFE_ERROR}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
