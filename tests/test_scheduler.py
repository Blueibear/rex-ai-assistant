"""Tests for the scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rex.scheduler import Scheduler


def test_scheduler_persistence_and_manual_run(tmp_path):
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    storage = tmp_path / "jobs.json"
    calls: list[str] = []

    scheduler = Scheduler(storage_path=storage, now_func=lambda: now)
    scheduler.register_handler("test_handler", lambda job: calls.append(job.job_id))
    job = scheduler.add_job("Test job", 60, "test_handler")

    assert job.next_run_at == now + timedelta(seconds=60)
    assert scheduler.run_job(job.job_id, manual=True) is True
    assert calls == [job.job_id]
    assert job.last_run_at == now

    scheduler_reload = Scheduler(storage_path=storage, now_func=lambda: now)
    loaded = scheduler_reload.get_job(job.job_id)
    assert loaded is not None
    assert loaded.name == "Test job"
    assert loaded.last_run_at == now


def test_scheduler_run_due_jobs(tmp_path):
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    later = now + timedelta(minutes=5)
    storage = tmp_path / "jobs.json"
    calls: list[str] = []

    scheduler = Scheduler(storage_path=storage, now_func=lambda: now)
    scheduler.register_handler("test_handler", lambda job: calls.append(job.job_id))
    job = scheduler.add_job("Test due job", 60, "test_handler")

    scheduler_due = Scheduler(storage_path=storage, now_func=lambda: later)
    scheduler_due.register_handler("test_handler", lambda job: calls.append(job.job_id))
    ran = scheduler_due.run_due_jobs()

    assert job.job_id in ran
    assert calls == [job.job_id]
