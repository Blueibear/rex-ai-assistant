"""Rex email bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list", "limit": 20}
    -> {"ok": true, "messages": [...], "configured": bool}

Message format (GUI):
  {id, thread_id, subject, sender, recipients, body_text, received_at,
   labels, is_read, priority}

When no email provider is configured, returns {"ok": true, "messages": [],
"configured": false} so the GUI can show an empty-state prompt.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rex.bridge_utils import bridge_error_response

OUTLOOK_EMAIL_UNSUPPORTED = (
    "Outlook email sync is not implemented yet. The current Outlook settings "
    "only store app credentials; Rex cannot read Outlook mail until Microsoft "
    "Graph OAuth token support is added."
)


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


def _handle_list(limit: int) -> dict[str, Any]:
    from rex.config import load_config
    from rex.integrations.email_service import EmailService

    cfg = load_config()
    provider = getattr(cfg, "email_provider", "none") or "none"
    if str(provider).lower() == "outlook":
        return {
            "ok": False,
            "error": OUTLOOK_EMAIL_UNSUPPORTED,
            "messages": [],
            "configured": True,
        }

    svc = EmailService(email_provider=provider)

    messages = svc.list_inbox(limit=limit)
    configured = provider != "none"
    return {
        "ok": True,
        "messages": [_msg_to_gui(m) for m in messages],
        "configured": configured,
    }


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("command") or "list")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        if command == "list":
            limit = int(payload.get("limit") or 20)
            result = _handle_list(limit)
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = bridge_error_response(exc)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
