"""Rex calendar bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list", "start": "<ISO>", "end": "<ISO>"}
    -> {"ok": true, "events": [...], "configured": bool}

Event format (GUI):
  {id, title, start, end, location?, description?, attendees?, source,
   is_all_day}

When no calendar provider is configured, returns {"ok": true, "events": [],
"configured": false} so the GUI can show an empty-state prompt.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from rex.bridge_utils import bridge_error_response

OUTLOOK_CALENDAR_UNSUPPORTED = (
    "Outlook calendar sync is not implemented yet. The current Outlook settings "
    "only store app credentials; Rex cannot read or write Outlook events until "
    "Microsoft Graph OAuth token support is added."
)


def _event_to_gui(event: Any) -> dict[str, Any]:
    return {
        "id": event.id,
        "title": event.title,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "location": event.location,
        "description": event.description,
        "attendees": list(event.attendees),
        "source": event.source,
        "is_all_day": event.is_all_day,
    }


def _handle_list(start_str: str, end_str: str) -> dict[str, Any]:
    from rex.config import load_config
    from rex.integrations.calendar_service import CalendarService

    cfg = load_config()
    provider = getattr(cfg, "calendar_provider", "none") or "none"
    if str(provider).lower() == "outlook":
        return {
            "ok": False,
            "error": OUTLOOK_CALENDAR_UNSUPPORTED,
            "events": [],
            "configured": True,
        }

    svc = CalendarService(calendar_provider=provider)

    try:
        start = datetime.fromisoformat(start_str) if start_str else datetime.now(UTC)
        end = datetime.fromisoformat(end_str) if end_str else start + timedelta(days=30)
    except ValueError:
        start = datetime.now(UTC)
        end = start + timedelta(days=30)

    events = svc.get_events(start, end)
    configured = provider != "none"
    return {
        "ok": True,
        "events": [_event_to_gui(e) for e in events],
        "configured": configured,
    }


def _handle_create(event_data: dict[str, Any]) -> dict[str, Any]:
    from rex.config import load_config
    from rex.integrations.calendar_service import CalendarService

    cfg = load_config()
    provider = getattr(cfg, "calendar_provider", "none") or "none"
    if str(provider).lower() == "outlook":
        return {
            "ok": False,
            "error": OUTLOOK_CALENDAR_UNSUPPORTED,
            "configured": True,
        }

    svc = CalendarService(calendar_provider=provider)
    event = svc.create_event(event_data)
    return {
        "ok": True,
        "event": _event_to_gui(event),
        "configured": provider != "none",
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
            result = _handle_list(
                str(payload.get("start") or ""),
                str(payload.get("end") or ""),
            )
        elif command == "create":
            raw_event = payload.get("event")
            if not isinstance(raw_event, dict):
                result = {"ok": False, "error": "Missing event payload."}
            else:
                result = _handle_create(raw_event)
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = bridge_error_response(exc)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
