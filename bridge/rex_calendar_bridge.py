"""Rex calendar bridge for Electron GUI.

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

Provider and credential selection is per user via
``rex.calendar_accounts``: named users use only their own assigned calendar
accounts and tokens; the legacy global provider configuration serves only
the explicit ``default`` profile.

Commands:
  {"command": "list", "user"?: "<user_id>", "start": "<ISO>", "end": "<ISO>"}
    -> {"ok": true, "events": [...], "configured": bool}

  {"command": "create", "user"?: "<user_id>", "event": {...}}
    -> {"ok": true, "event": {...}, "configured": bool}

Event format (GUI):
  {id, title, start, end, location?, description?, attendees?, source,
   is_all_day}

When the resolved user has no calendar provider configured, returns
{"ok": true, "events": [], "configured": false} so the GUI can show an
empty-state prompt.
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

_NO_USER_ERROR = (
    "No active user for calendar. Set one with 'rex identify --user <id>' "
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


def _service_for_user(user_id: str) -> tuple[Any, str]:
    from rex.integrations.calendar_service import create_calendar_service_for_user

    return create_calendar_service_for_user(user_id)


def _handle_list(user_id: str, start_str: str, end_str: str) -> dict[str, Any]:
    try:
        svc, provider = _service_for_user(user_id)
    except PermissionError:
        return {"ok": False, "error": _NO_USER_ERROR, "events": [], "configured": False}

    if provider == "outlook":
        return {
            "ok": False,
            "error": OUTLOOK_CALENDAR_UNSUPPORTED,
            "events": [],
            "configured": True,
        }

    if svc is None:
        return {"ok": True, "events": [], "configured": False}

    try:
        start = datetime.fromisoformat(start_str) if start_str else datetime.now(UTC)
        end = datetime.fromisoformat(end_str) if end_str else start + timedelta(days=30)
    except ValueError:
        start = datetime.now(UTC)
        end = start + timedelta(days=30)

    events = svc.get_events(start, end)
    return {
        "ok": True,
        "events": [_event_to_gui(e) for e in events],
        "configured": True,
    }


def _handle_create(user_id: str, event_data: dict[str, Any]) -> dict[str, Any]:
    try:
        svc, provider = _service_for_user(user_id)
    except PermissionError:
        return {"ok": False, "error": _NO_USER_ERROR, "configured": False}

    if provider == "outlook":
        return {
            "ok": False,
            "error": OUTLOOK_CALENDAR_UNSUPPORTED,
            "configured": True,
        }

    if svc is None:
        return {
            "ok": False,
            "error": "No calendar account is configured for this user.",
            "configured": False,
        }

    event = svc.create_event(event_data)
    return {
        "ok": True,
        "event": _event_to_gui(event),
        "configured": True,
    }


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("command") or "list")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        user_id = _resolve_user(payload)
        if not user_id:
            result: dict[str, Any] = {
                "ok": False,
                "error": _NO_USER_ERROR,
                "events": [],
                "configured": False,
            }
        elif command == "list":
            result = _handle_list(
                user_id,
                str(payload.get("start") or ""),
                str(payload.get("end") or ""),
            )
        elif command == "create":
            raw_event = payload.get("event")
            if not isinstance(raw_event, dict):
                result = {"ok": False, "error": "Missing event payload."}
            else:
                result = _handle_create(user_id, raw_event)
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = bridge_error_response(exc)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
