"""Rex reminders bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list"}
    -> {"ok": true, "reminders": [...]}

  {"command": "save", "reminder": {id?, title, notes?, dueAt, priority, repeat}}
    -> {"ok": true, "reminder": {...}}

  {"command": "delete", "id": "<reminder_id>"}
    -> {"ok": true}

  {"command": "complete", "id": "<reminder_id>"}
    -> {"ok": true}

Reminder format (GUI):
  {id, title, notes?, dueAt (ISO), priority: "low"|"medium"|"high", done, repeat?: "none"|"daily"|"weekly"|"custom"}
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


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


def _handle_list() -> dict[str, Any]:
    from rex.reminder_service import get_reminder_service  # type: ignore[import]

    service = get_reminder_service()
    reminders = service.list_reminders(status="pending")
    return {"ok": True, "reminders": [_reminder_to_gui(r) for r in reminders]}


def _handle_save(reminder_data: dict[str, Any]) -> dict[str, Any]:
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
        existing = service.get_reminder(reminder_id)
        if existing is not None:
            existing.title = title
            existing.remind_at = remind_at
            existing.metadata = metadata
            service._save()  # noqa: SLF001
            return {"ok": True, "reminder": _reminder_to_gui(existing)}

    reminder = service.create_reminder(
        user_id="default",
        title=title,
        remind_at=remind_at,
        metadata=metadata,
        reminder_id=reminder_id,
    )
    return {"ok": True, "reminder": _reminder_to_gui(reminder)}


def _handle_delete(reminder_id: str) -> dict[str, Any]:
    from rex.reminder_service import get_reminder_service  # type: ignore[import]

    service = get_reminder_service()
    existed = service.delete_reminder(reminder_id)
    if not existed:
        return {"ok": False, "error": f"Reminder {reminder_id!r} not found"}
    return {"ok": True}


def _handle_complete(reminder_id: str) -> dict[str, Any]:
    from rex.reminder_service import get_reminder_service  # type: ignore[import]

    service = get_reminder_service()
    marked = service.mark_done(reminder_id)
    if not marked:
        return {"ok": False, "error": f"Reminder {reminder_id!r} not found"}
    return {"ok": True}


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("command") or "")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        if command == "list":
            result = _handle_list()
        elif command == "save":
            result = _handle_save(dict(payload.get("reminder") or {}))
        elif command == "delete":
            result = _handle_delete(str(payload.get("id") or ""))
        elif command == "complete":
            result = _handle_complete(str(payload.get("id") or ""))
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
