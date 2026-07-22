"""Rex SMS bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list_threads"}
    -> {"ok": true, "threads": [...], "configured": bool}

Thread format (GUI):
  {id, contact_name, contact_number, messages: [...], last_message_at,
   unread_count}

Message format:
  {id, thread_id, direction, body, from_number, to_number, sent_at, status}

When SMS is not configured, returns {"ok": true, "threads": [],
"configured": false} so the GUI can show an empty-state prompt.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from rex.bridge_utils import bridge_error_response


def _msg_to_gui(msg: Any) -> dict[str, Any]:
    return {
        "id": msg.id,
        "thread_id": msg.thread_id,
        "direction": msg.direction,
        "body": msg.body,
        "from_number": msg.from_number,
        "to_number": msg.to_number,
        "sent_at": msg.sent_at.isoformat(),
        "status": msg.status,
    }


def _thread_to_gui(thread: Any) -> dict[str, Any]:
    return {
        "id": thread.id,
        "contact_name": thread.contact_name,
        "contact_number": thread.contact_number,
        "messages": [_msg_to_gui(m) for m in thread.messages],
        "last_message_at": thread.last_message_at.isoformat(),
        "unread_count": thread.unread_count,
    }


def _handle_list_threads() -> dict[str, Any]:
    from rex.integrations.sms_service import SMSService

    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    provider = "twilio" if (sid and token) else "none"
    svc = SMSService(sms_provider=provider)

    threads = svc.list_threads()
    configured = provider != "none"
    return {
        "ok": True,
        "threads": [_thread_to_gui(t) for t in threads],
        "configured": configured,
    }


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("command") or "list_threads")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        from rex.identity import validate_user_id

        validate_user_id(str(payload.get("user") or ""))
        if payload.get("data_scope") != "private":
            raise PermissionError("SMS requires private Electron data scope")
        if command == "list_threads":
            result = _handle_list_threads()
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = bridge_error_response(exc)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
