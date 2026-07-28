"""Two-user regression coverage for Electron-owned private and shared data."""

from __future__ import annotations

from pathlib import Path


def test_task_bridge_isolates_two_users(tmp_path: Path, monkeypatch) -> None:
    from bridge import rex_tasks_bridge
    from rex.scheduler import Scheduler

    scheduler = Scheduler(storage_path=tmp_path / "jobs.json")
    monkeypatch.setattr("rex.scheduler.get_scheduler", lambda: scheduler)

    james_task = rex_tasks_bridge._handle_save(
        "james",
        {"name": "James private task", "prompt": "private", "schedule": "Every hour"},
    )["task"]
    cole_task = rex_tasks_bridge._handle_save(
        "cole",
        {"name": "Cole private task", "prompt": "private", "schedule": "Every hour"},
    )["task"]

    assert [task["id"] for task in rex_tasks_bridge._handle_list("james")["tasks"]] == [
        james_task["id"]
    ]
    assert [task["id"] for task in rex_tasks_bridge._handle_list("cole")["tasks"]] == [
        cole_task["id"]
    ]
    assert rex_tasks_bridge._handle_delete("james", cole_task["id"])["ok"] is False
    assert scheduler.get_job(cole_task["id"]) is not None


def test_legacy_unowned_tasks_are_quarantined(tmp_path: Path, monkeypatch) -> None:
    from bridge import rex_tasks_bridge
    from rex.scheduler import Scheduler

    scheduler = Scheduler(storage_path=tmp_path / "jobs.json")
    scheduler.add_job(job_id="legacy", name="Unowned", schedule="interval:3600")
    monkeypatch.setattr("rex.scheduler.get_scheduler", lambda: scheduler)

    james = rex_tasks_bridge._handle_list("james")
    cole = rex_tasks_bridge._handle_list("cole")
    assert james["tasks"] == cole["tasks"] == []
    assert james["legacy_unowned_count"] == cole["legacy_unowned_count"] == 1


def test_shopping_payload_requires_explicit_shared_scope() -> None:
    """The main process cannot silently relabel private data as household data."""
    source = Path("bridge/rex_shopping_list_bridge.py").read_text(encoding="utf-8")
    assert 'payload.get("data_scope") != "shared_household"' in source
    assert "added_by=actor_user_id" in source
