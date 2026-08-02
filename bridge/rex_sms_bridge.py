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
import sys
from typing import Any

from rex.bridge_utils import bridge_safe_error_response


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
    from rex.credentials import get_persisted_credential
    from rex.integrations.sms_service import SMSService

    sid = get_persisted_credential("TWILIO_ACCOUNT_SID") or ""
    token = get_persisted_credential("TWILIO_AUTH_TOKEN") or ""
    from_number = get_persisted_credential("TWILIO_FROM_NUMBER") or ""
    provider = "twilio" if (sid and token) else "none"
    svc = SMSService(
        sms_provider=provider,
        account_sid=sid,
        auth_token=token,
        from_number=from_number,
    )

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
    except Exception:
        print(json.dumps({"ok": False, "error": "Invalid SMS request"}), flush=True)
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
        result = bridge_safe_error_response(
            exc,
            messages={
                PermissionError: "SMS requires private Electron data scope",
                ValueError: "SMS request is invalid",
            },
            default="SMS request failed",
        )

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
