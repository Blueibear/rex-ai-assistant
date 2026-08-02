"""Rex email bridge for Electron GUI.

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

Provider routing is per-user: a ``gmail`` account assigned to the user in
``users.{id}.email_accounts`` is served with that user's own token; the
legacy global ``email.provider`` + ``GMAIL_ACCESS_TOKEN`` path applies only
to the explicit ``default`` profile.  Named users never inherit the global
credentials.  Credential references and secrets are never returned to the
renderer.

Commands:
  {"command": "list", "limit": 20, "user"?: "<user_id>"}
    -> {"ok": true, "messages": [...], "configured": bool}

Message format (GUI):
  {id, thread_id, subject, sender, recipients, body_text, received_at,
   labels, is_read, priority}

When the resolved user has no email provider configured, returns
{"ok": true, "messages": [], "configured": false} so the GUI can show an
empty-state prompt.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rex.bridge_utils import bridge_safe_error_response

OUTLOOK_EMAIL_UNSUPPORTED = (
    "Outlook email sync is not implemented yet. The current Outlook settings "
    "only store app credentials; Rex cannot read Outlook mail until Microsoft "
    "Graph OAuth token support is added."
)

_NO_USER_ERROR = (
    "No active user for email. Set one with 'rex identify --user <id>' "
    "or include a 'user' field in the request."
)


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


def _msg_to_gui(msg: Any) -> dict[str, Any]:
    return {
        "id": msg.id,
        "thread_id": msg.thread_id,
        "subject": msg.subject,
        "sender": msg.sender,
        "recipients": list(msg.recipients),
        "body_text": msg.body_text,
        "received_at": msg.received_at.isoformat(),
        "labels": list(msg.labels),
        "is_read": msg.is_read,
        "priority": msg.priority,
    }


def _handle_list(user_id: str, limit: int) -> dict[str, Any]:
    from rex.integrations.email_service import create_email_service_for_user

    svc, provider = create_email_service_for_user(user_id)
    if provider == "outlook":
        return {
            "ok": False,
            "error": OUTLOOK_EMAIL_UNSUPPORTED,
            "messages": [],
            "configured": True,
        }
    if svc is None:
        return {"ok": True, "messages": [], "configured": False}

    messages = svc.list_inbox(limit=limit)
    return {
        "ok": True,
        "messages": [_msg_to_gui(m) for m in messages],
        "configured": True,
    }


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("command") or "list")
    except Exception:
        print(json.dumps({"ok": False, "error": "Invalid email request"}), flush=True)
        sys.exit(1)

    try:
        user_id = _resolve_user(payload)
        if not user_id:
            result = {
                "ok": False,
                "error": _NO_USER_ERROR,
                "messages": [],
                "configured": False,
            }
        elif command == "list":
            limit = int(payload.get("limit") or 20)
            result = _handle_list(user_id, limit)
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except PermissionError:
        result = {
            "ok": False,
            "error": _NO_USER_ERROR,
            "messages": [],
            "configured": False,
        }
    except Exception as exc:
        result = bridge_safe_error_response(
            exc,
            messages={ValueError: "Email request is invalid"},
            default="Email request failed",
        )
        result["messages"] = []
        result["configured"] = False

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
