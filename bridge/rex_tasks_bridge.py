"""Rex tasks bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list"}
    -> {"ok": true, "tasks": [...]}

  {"command": "save", "task": {id?, name, prompt, schedule, active}}
    -> {"ok": true, "task": {...}}

  {"command": "delete", "id": "<task_id>"}
    -> {"ok": true}

  {"command": "set_enabled", "id": "<task_id>", "enabled": bool}
    -> {"ok": true, "task": {...}}

Task format (GUI):
  {id, name, prompt, schedule, nextRun, status: "active"|"paused"|"error"}

The bridge stores the GUI schedule string and prompt in job.metadata so they
survive round-trips through the scheduler.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from rex.bridge_utils import bridge_error_response, repo_root, resolve_python

_PYTHON_EXE = resolve_python()  # venv-aware interpreter path for subprocess calls
_REPO_ROOT = repo_root()  # absolute repo root for resolving scripts and config


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _schedule_to_interval(gui_schedule: str) -> int:
    """Convert a GUI schedule string to an interval in seconds."""
    s = gui_schedule.strip().lower()
    if s == "every hour":
        return 3600
    if s.startswith("every day at"):
        return 86400
    if s.startswith("every ") and " at " in s:
        # weekly
        return 604800
    # Custom cron or unrecognised — default to hourly
    return 3600


def _next_run_display(next_run: datetime | None) -> str:
    """Format a next_run datetime as a short human-readable string."""
    if next_run is None:
        return "Pending"
    try:
        # Make next_run tz-aware if needed
        if next_run.tzinfo is None:
            next_run = next_run.replace(tzinfo=UTC)
        diff = (next_run - _utc_now()).total_seconds()
        if diff < 0:
            return "Overdue"
        if diff < 60:
            return f"In {int(diff)}s"
        if diff < 3600:
            mins = int(diff // 60)
            return f"In {mins} min"
        if diff < 86400:
            hrs = int(diff // 3600)
            return f"In {hrs} hr{'s' if hrs != 1 else ''}"
        days = int(diff // 86400)
        return f"In {days} day{'s' if days != 1 else ''}"
    except Exception:
        return "Pending"


def _job_to_task(job: Any) -> dict[str, Any]:
    """Convert a ScheduledJob to the GUI Task dict format."""
    meta: dict[str, Any] = getattr(job, "metadata", None) or {}
    gui_schedule = meta.get("gui_schedule") or getattr(job, "schedule", "Every hour")
    prompt = meta.get("prompt") or ""
    status = "active" if getattr(job, "enabled", True) else "paused"
    next_run_str = _next_run_display(getattr(job, "next_run", None))
    return {
        "id": job.job_id,
        "name": job.name,
        "prompt": prompt,
        "schedule": gui_schedule,
        "nextRun": next_run_str,
        "status": status,
    }


def _owner(job: Any) -> str | None:
    metadata = getattr(job, "metadata", None) or {}
    owner = metadata.get("owner_user_id")
    return str(owner) if owner else None


def _handle_list(user_id: str) -> dict[str, Any]:
    from rex.scheduler import get_scheduler

    scheduler = get_scheduler()
    jobs = scheduler.list_jobs()
    tasks = [_job_to_task(job) for job in jobs if _owner(job) == user_id]
    unowned = sum(1 for job in jobs if _owner(job) is None)
    return {"ok": True, "tasks": tasks, "legacy_unowned_count": unowned}


def _handle_save(user_id: str, task: dict[str, Any]) -> dict[str, Any]:
    from rex.scheduler import get_scheduler  # type: ignore[import]

    scheduler = get_scheduler()

    task_id = str(task.get("id") or uuid.uuid4())
    name = str(task.get("name") or "").strip() or "Unnamed Task"
    prompt = str(task.get("prompt") or "")
    gui_schedule = str(task.get("schedule") or "Every hour")
    active = bool(task.get("active", True))

    interval = _schedule_to_interval(gui_schedule)
    internal_schedule = f"interval:{interval}"
    metadata: dict[str, Any] = {
        "gui_schedule": gui_schedule,
        "prompt": prompt,
        "owner_user_id": user_id,
        "data_scope": "private",
    }

    existing = scheduler.get_job(task_id)
    if existing is not None:
        if _owner(existing) != user_id:
            return {"ok": False, "error": f"Task {task_id!r} not found"}
        updated = scheduler.update_job(
            task_id,
            name=name,
            schedule=internal_schedule,
            enabled=active,
            metadata=metadata,
        )
        if updated is None:
            return {"ok": False, "error": "Failed to update task"}
        return {"ok": True, "task": _job_to_task(updated)}

    job = scheduler.add_job(
        job_id=task_id,
        name=name,
        schedule=internal_schedule,
        enabled=active,
        metadata=metadata,
    )
    return {"ok": True, "task": _job_to_task(job)}


def _handle_delete(user_id: str, task_id: str) -> dict[str, Any]:
    from rex.scheduler import get_scheduler  # type: ignore[import]

    scheduler = get_scheduler()
    existing = scheduler.get_job(task_id)
    if existing is None or _owner(existing) != user_id:
        return {"ok": False, "error": f"Task {task_id!r} not found"}
    existed = scheduler.remove_job(task_id)
    if not existed:
        return {"ok": False, "error": f"Task {task_id!r} not found"}
    return {"ok": True}


def _handle_set_enabled(user_id: str, task_id: str, enabled: bool) -> dict[str, Any]:
    from rex.scheduler import get_scheduler  # type: ignore[import]

    scheduler = get_scheduler()
    existing = scheduler.get_job(task_id)
    if existing is None or _owner(existing) != user_id:
        return {"ok": False, "error": f"Task {task_id!r} not found"}
    updated = scheduler.update_job(task_id, enabled=enabled)
    if updated is None:
        return {"ok": False, "error": f"Task {task_id!r} not found"}
    return {"ok": True, "task": _job_to_task(updated)}


def _handle_history(user_id: str, task_id: str) -> dict[str, Any]:
    """Return persisted task runs when available; never synthesize successes."""
    from rex.scheduler import get_scheduler  # type: ignore[import]

    existing = get_scheduler().get_job(task_id)
    if existing is None or _owner(existing) != user_id:
        return {"ok": False, "error": f"Task {task_id!r} not found"}
    metadata = getattr(existing, "metadata", None) or {}
    runs = metadata.get("run_history")
    return {"ok": True, "runs": runs if isinstance(runs, list) else []}


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("action") or payload.get("command") or "")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        from rex.identity import validate_user_id

        user_id = validate_user_id(str(payload.get("user") or ""))
        if payload.get("data_scope") != "private":
            raise PermissionError("Tasks require private Electron data scope")
        if command == "list":
            result = _handle_list(user_id)
        elif command == "save":
            result = _handle_save(user_id, dict(payload.get("task") or {}))
        elif command == "delete":
            result = _handle_delete(user_id, str(payload.get("id") or ""))
        elif command == "set_enabled":
            result = _handle_set_enabled(
                user_id,
                str(payload.get("id") or ""),
                bool(payload.get("enabled", True)),
            )
        elif command == "history":
            result = _handle_history(user_id, str(payload.get("id") or ""))
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = bridge_error_response(exc)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
