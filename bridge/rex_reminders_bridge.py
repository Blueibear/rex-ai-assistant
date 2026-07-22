"""Rex reminders bridge for Electron GUI.

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
    -> {"ok": true, "reminders": [...]}   (only the resolved user's reminders)

  {"command": "save", "user"?: "<user_id>", "reminder": {id?, title, notes?, dueAt, priority, repeat}}
    -> {"ok": true, "reminder": {...}}    (owned by the resolved user)

  {"command": "delete", "user"?: "<user_id>", "id": "<reminder_id>"}
    -> {"ok": true}                       (ownership enforced)

  {"command": "complete", "user"?: "<user_id>", "id": "<reminder_id>"}
    -> {"ok": true}                       (ownership enforced)

Reminder format (GUI):
  {id, title, notes?, dueAt (ISO), priority: "low"|"medium"|"high", done, repeat?: "none"|"daily"|"weekly"|"custom"}
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

from rex.bridge_utils import bridge_error_response, repo_root, resolve_python

_PYTHON_EXE = resolve_python()  # venv-aware interpreter path for subprocess calls
_REPO_ROOT = repo_root()  # absolute repo root for resolving scripts and config

_NO_USER_ERROR = (
    "No active user for reminders. Set one with 'rex identify --user <id>' "
    "or include a 'user' field in the request."
)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


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


def _reminder_to_gui(reminder: Any) -> dict[str, Any]:
    """Convert a ReminderService Reminder to the GUI Reminder dict format."""
    meta: dict[str, Any] = getattr(reminder, "metadata", None) or {}
    due_at = reminder.remind_at
    due_at_str = due_at.isoformat() if hasattr(due_at, "isoformat") else str(due_at)
    return {
        "id": reminder.reminder_id,
        "title": reminder.title,
        "notes": meta.get("notes") or None,
        "dueAt": due_at_str,
        "priority": meta.get("priority") or "medium",
        "done": reminder.status in ("done", "canceled"),
        "repeat": meta.get("repeat") or "none",
    }


def _handle_list(user_id: str) -> dict[str, Any]:
    from rex.reminder_service import get_reminder_service  # type: ignore[import]

    service = get_reminder_service()
    reminders = service.list_reminders(user_id=user_id, status="pending")
    return {"ok": True, "reminders": [_reminder_to_gui(r) for r in reminders]}


def _handle_save(user_id: str, reminder_data: dict[str, Any]) -> dict[str, Any]:
    from rex.reminder_service import get_reminder_service  # type: ignore[import]

    service = get_reminder_service()

    reminder_id: str | None = str(reminder_data.get("id") or "").strip() or None
    title = str(reminder_data.get("title") or "").strip() or "Untitled Reminder"
    notes = reminder_data.get("notes") or None
    due_at_str = str(reminder_data.get("dueAt") or "")
    priority = str(reminder_data.get("priority") or "medium")
    repeat = str(reminder_data.get("repeat") or "none")

    try:
        remind_at = datetime.fromisoformat(due_at_str.replace("Z", "+00:00"))
    except Exception:
        remind_at = _utc_now()

    metadata: dict[str, Any] = {
        "notes": notes,
        "priority": priority,
        "repeat": repeat,
    }

    if reminder_id:
        updated = service.update_reminder(
            reminder_id,
            user_id,
            title=title,
            remind_at=remind_at,
            metadata=metadata,
        )
        if updated is not None:
            return {"ok": True, "reminder": _reminder_to_gui(updated)}

    try:
        reminder = service.create_reminder(
            user_id=user_id,
            title=title,
            remind_at=remind_at,
            metadata=metadata,
            reminder_id=reminder_id,
        )
    except ValueError:
        # Caller-supplied ID collides with a reminder it does not own (or the
        # identity is invalid). Do not overwrite; report as not found.
        return {"ok": False, "error": f"Reminder {reminder_id!r} not found"}
    return {"ok": True, "reminder": _reminder_to_gui(reminder)}


def _handle_delete(user_id: str, reminder_id: str) -> dict[str, Any]:
    from rex.reminder_service import get_reminder_service  # type: ignore[import]

    service = get_reminder_service()
    existed = service.delete_reminder(reminder_id, user_id)
    if not existed:
        return {"ok": False, "error": f"Reminder {reminder_id!r} not found"}
    return {"ok": True}


def _handle_complete(user_id: str, reminder_id: str) -> dict[str, Any]:
    from rex.reminder_service import get_reminder_service  # type: ignore[import]

    service = get_reminder_service()
    marked = service.mark_done(reminder_id, user_id)
    if not marked:
        return {"ok": False, "error": f"Reminder {reminder_id!r} not found"}
    return {"ok": True}


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("action") or payload.get("command") or "")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        if command in ("list", "save", "delete", "complete"):
            user_id = _resolve_user(payload)
            if user_id is None:
                result: dict[str, Any] = {"ok": False, "error": _NO_USER_ERROR}
            elif command == "list":
                result = _handle_list(user_id)
            elif command == "save":
                result = _handle_save(user_id, dict(payload.get("reminder") or {}))
            elif command == "delete":
                result = _handle_delete(user_id, str(payload.get("id") or ""))
            else:
                result = _handle_complete(user_id, str(payload.get("id") or ""))
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = bridge_error_response(exc)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
