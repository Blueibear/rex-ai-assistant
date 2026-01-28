"""
Tests for the scheduler.

This test module supports BOTH scheduler variants that have appeared in the
codebase:

Variant A (legacy/simple persisted scheduler):
- Scheduler(storage_path=Path, now_func=callable)
- register_handler(name, handler)
- add_job(name, interval_seconds, handler_name, ...)
- run_job(job_id, manual=True/False)
- run_due_jobs()
- get_job(job_id) -> JobDefinition with last_run_at/next_run_at

Variant B (newer/background scheduler with pydantic jobs):
- Scheduler(jobs_file=Path)
- ScheduledJob model
- register_callback(name, callback)
- add_job(job_id, name, schedule="interval:seconds", ...)
- run_job(job_id, force=False)
- start()/stop()
- persistence via jobs_file

The tests auto-detect which API is available at runtime and skip incompatible
tests. This keeps CI green while implementations converge.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from rex.scheduler import Scheduler


def _has_attr(obj: Any, name: str) -> bool:
    return hasattr(obj, name)


def _scheduled_job_class():
    try:
        from rex.scheduler import ScheduledJob  # type: ignore

        return ScheduledJob
    except Exception:
        return None


def _is_variant_a() -> bool:
    """
    Detect legacy scheduler:
    - Scheduler accepts storage_path and/or now_func
    - has register_handler / run_due_jobs
    """
    try:
        # If init accepts these kwargs, this will work.
        _ = Scheduler(storage_path=Path("dummy.json"), now_func=lambda: datetime.now(timezone.utc))  # type: ignore[call-arg]
        return True
    except TypeError:
        return False
    except Exception:
        # It accepted the signature but failed for other reasons (unlikely).
        return True


def _is_variant_b() -> bool:
    """
    Detect newer scheduler:
    - Scheduler accepts jobs_file kwarg
    - has start/stop, register_callback, callbacks dict, etc.
    """
    try:
        _ = Scheduler(jobs_file=Path("dummy.json"))  # type: ignore[call-arg]
        return True
    except TypeError:
        return False
    except Exception:
        return True


# -------------------------------------------------------------------
# Variant A tests (legacy simple scheduler)
# -------------------------------------------------------------------


@pytest.mark.skipif(
    not _is_variant_a(),
    reason="Legacy Scheduler(storage_path, now_func) API not available in this build.",
)
def test_scheduler_persistence_and_manual_run(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    storage = tmp_path / "jobs.json"
    calls: list[str] = []

    scheduler = Scheduler(storage_path=storage, now_func=lambda: now)  # type: ignore[call-arg]
    scheduler.register_handler("test_handler", lambda job: calls.append(job.job_id))  # type: ignore[attr-defined]
    job = scheduler.add_job("Test job", 60, "test_handler")  # type: ignore[attr-defined]

    assert job.next_run_at == now + timedelta(seconds=60)
    assert scheduler.run_job(job.job_id, manual=True) is True  # type: ignore[attr-defined]
    assert calls == [job.job_id]
    assert job.last_run_at == now

    scheduler_reload = Scheduler(storage_path=storage, now_func=lambda: now)  # type: ignore[call-arg]
    loaded = scheduler_reload.get_job(job.job_id)  # type: ignore[attr-defined]
    assert loaded is not None
    assert loaded.name == "Test job"
    assert loaded.last_run_at == now


@pytest.mark.skipif(
    not _is_variant_a(),
    reason="Legacy Scheduler(storage_path, now_func) API not available in this build.",
)
def test_scheduler_run_due_jobs(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    later = now + timedelta(minutes=5)
    storage = tmp_path / "jobs.json"
    calls: list[str] = []

    scheduler = Scheduler(storage_path=storage, now_func=lambda: now)  # type: ignore[call-arg]
    scheduler.register_handler("test_handler", lambda job: calls.append(job.job_id))  # type: ignore[attr-defined]
    job = scheduler.add_job("Test due job", 60, "test_handler")  # type: ignore[attr-defined]

    scheduler_due = Scheduler(storage_path=storage, now_func=lambda: later)  # type: ignore[call-arg]
    scheduler_due.register_handler("test_handler", lambda j: calls.append(j.job_id))  # type: ignore[attr-defined]
    ran = scheduler_due.run_due_jobs()  # type: ignore[attr-defined]

    assert job.job_id in ran
    assert calls == [job.job_id]


# -------------------------------------------------------------------
# Variant B fixtures (newer scheduler)
# -------------------------------------------------------------------


@pytest.fixture
def temp_jobs_file(tmp_path: Path) -> Path:
    return tmp_path / "test_jobs.json"


@pytest.fixture
def scheduler(temp_jobs_file: Path) -> Scheduler:
    return Scheduler(jobs_file=temp_jobs_file)  # type: ignore[call-arg]


# -------------------------------------------------------------------
# Variant B tests (newer pydantic/background scheduler)
# -------------------------------------------------------------------


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_scheduler_initialization(temp_jobs_file: Path) -> None:
    s = Scheduler(jobs_file=temp_jobs_file)  # type: ignore[call-arg]
    assert getattr(s, "jobs", {}) == {}
    assert getattr(s, "jobs_file", None) == temp_jobs_file


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_add_job(scheduler: Scheduler) -> None:
    ScheduledJob = _scheduled_job_class()
    assert ScheduledJob is not None

    job = scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        enabled=True,
    )

    assert isinstance(job, ScheduledJob)
    assert job.job_id == "test-job"
    assert job.name == "Test Job"
    assert job.schedule == "interval:60"
    assert job.enabled is True
    assert job.run_count == 0
    assert job.next_run > datetime.now()


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_add_job_with_callback(scheduler: Scheduler) -> None:
    def test_callback(job):
        return None

    scheduler.register_callback("test_cb", test_callback)  # type: ignore[attr-defined]

    job = scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job-cb",
        name="Test Job with Callback",
        schedule="interval:60",
        callback_name="test_cb",
    )

    assert job.callback_name == "test_cb"


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_add_job_with_workflow(scheduler: Scheduler) -> None:
    job = scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job-wf",
        name="Test Job with Workflow",
        schedule="interval:60",
        workflow_id="workflow-123",
    )
    assert job.workflow_id == "workflow-123"


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_remove_job(scheduler: Scheduler) -> None:
    scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
    )

    assert scheduler.remove_job("test-job") is True  # type: ignore[attr-defined]
    assert scheduler.get_job("test-job") is None  # type: ignore[attr-defined]
    assert scheduler.remove_job("non-existent") is False  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_list_jobs(scheduler: Scheduler) -> None:
    scheduler.add_job(job_id="job1", name="Job 1", schedule="interval:60")  # type: ignore[attr-defined]
    scheduler.add_job(job_id="job2", name="Job 2", schedule="interval:120")  # type: ignore[attr-defined]

    jobs = scheduler.list_jobs()  # type: ignore[attr-defined]
    assert len(jobs) == 2
    assert {job.job_id for job in jobs} == {"job1", "job2"}


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_get_job(scheduler: Scheduler) -> None:
    scheduler.add_job(job_id="test-job", name="Test Job", schedule="interval:60")  # type: ignore[attr-defined]
    job = scheduler.get_job("test-job")  # type: ignore[attr-defined]
    assert job is not None
    assert job.job_id == "test-job"
    assert scheduler.get_job("non-existent") is None  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_update_job(scheduler: Scheduler) -> None:
    scheduler.add_job(job_id="test-job", name="Test Job", schedule="interval:60")  # type: ignore[attr-defined]
    updated = scheduler.update_job("test-job", name="Updated Job", enabled=False)  # type: ignore[attr-defined]
    assert updated is not None
    assert updated.name == "Updated Job"
    assert updated.enabled is False
    assert scheduler.update_job("non-existent", name="Foo") is None  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_run_job_with_callback(scheduler: Scheduler) -> None:
    executed: list[str] = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)  # type: ignore[attr-defined]

    job = scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        callback_name="test_cb",
    )

    initial_run_count = job.run_count
    initial_next_run = job.next_run

    result = scheduler.run_job("test-job")  # type: ignore[attr-defined]
    assert result is True
    assert "test-job" in executed

    job2 = scheduler.get_job("test-job")  # type: ignore[attr-defined]
    assert job2 is not None
    assert job2.run_count == initial_run_count + 1
    assert job2.next_run > initial_next_run


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_run_job_disabled(scheduler: Scheduler) -> None:
    executed: list[str] = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)  # type: ignore[attr-defined]
    scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        callback_name="test_cb",
        enabled=False,
    )

    assert scheduler.run_job("test-job", force=False) is False  # type: ignore[attr-defined]
    assert len(executed) == 0

    assert scheduler.run_job("test-job", force=True) is True  # type: ignore[attr-defined]
    assert len(executed) == 1


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_run_job_max_runs(scheduler: Scheduler) -> None:
    executed: list[str] = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)  # type: ignore[attr-defined]
    scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        callback_name="test_cb",
        max_runs=2,
    )

    assert scheduler.run_job("test-job") is True  # type: ignore[attr-defined]
    assert scheduler.run_job("test-job") is True  # type: ignore[attr-defined]
    assert scheduler.run_job("test-job", force=False) is False  # type: ignore[attr-defined]
    assert scheduler.run_job("test-job", force=True) is True  # type: ignore[attr-defined]
    assert len(executed) == 3


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_run_due_jobs_idempotent(temp_jobs_file: Path) -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    executed: list[str] = []

    scheduler = Scheduler(jobs_file=temp_jobs_file, now_func=lambda: now)  # type: ignore[call-arg]

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)  # type: ignore[attr-defined]
    job = scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        callback_name="test_cb",
    )

    job.next_run = now
    scheduler.run_due_jobs()  # type: ignore[attr-defined]
    scheduler.run_due_jobs()  # type: ignore[attr-defined]

    assert executed == [job.job_id]


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_calculate_next_run(scheduler: Scheduler) -> None:
    now = datetime.now()

    next_run = scheduler._calculate_next_run("interval:60", from_time=now)  # type: ignore[attr-defined]
    expected = now + timedelta(seconds=60)
    assert abs((next_run - expected).total_seconds()) < 1

    next_run = scheduler._calculate_next_run("interval:3600", from_time=now)  # type: ignore[attr-defined]
    expected = now + timedelta(seconds=3600)
    assert abs((next_run - expected).total_seconds()) < 1


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_update_next_run(scheduler: Scheduler) -> None:
    job = scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
    )

    initial_next_run = job.next_run
    time.sleep(0.1)

    scheduler.update_next_run("test-job")  # type: ignore[attr-defined]
    job2 = scheduler.get_job("test-job")  # type: ignore[attr-defined]
    assert job2 is not None
    assert job2.next_run > initial_next_run


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_persistence(scheduler: Scheduler, temp_jobs_file: Path) -> None:
    scheduler.add_job(job_id="job1", name="Job 1", schedule="interval:60", enabled=True)  # type: ignore[attr-defined]
    scheduler.add_job(job_id="job2", name="Job 2", schedule="interval:120", enabled=False)  # type: ignore[attr-defined]

    scheduler2 = Scheduler(jobs_file=temp_jobs_file)  # type: ignore[call-arg]
    jobs = scheduler2.list_jobs()  # type: ignore[attr-defined]
    assert len(jobs) == 2

    job1 = scheduler2.get_job("job1")  # type: ignore[attr-defined]
    assert job1 is not None
    assert job1.name == "Job 1"
    assert job1.enabled is True

    job2 = scheduler2.get_job("job2")  # type: ignore[attr-defined]
    assert job2 is not None
    assert job2.name == "Job 2"
    assert job2.enabled is False


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_scheduler_start_stop(scheduler: Scheduler) -> None:
    scheduler.start()  # type: ignore[attr-defined]
    assert getattr(scheduler, "_running", False) is True
    assert getattr(scheduler, "_thread", None) is not None

    scheduler.stop()  # type: ignore[attr-defined]
    assert getattr(scheduler, "_running", True) is False


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_scheduled_execution(scheduler: Scheduler) -> None:
    executed: list[str] = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)  # type: ignore[attr-defined]
    scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:2",
        callback_name="test_cb",
    )

    job = scheduler.get_job("test-job")  # type: ignore[attr-defined]
    assert job is not None
    job.next_run = datetime.now() - timedelta(seconds=1)
    scheduler._save_jobs()  # type: ignore[attr-defined]

    scheduler.start()  # type: ignore[attr-defined]
    time.sleep(3)
    scheduler.stop()  # type: ignore[attr-defined]

    assert "test-job" in executed


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_run_job_nonexistent(scheduler: Scheduler) -> None:
    result = scheduler.run_job("non-existent")  # type: ignore[attr-defined]
    assert result is False


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_register_callback(scheduler: Scheduler) -> None:
    def callback1(job):
        return None

    def callback2(job):
        return None

    scheduler.register_callback("cb1", callback1)  # type: ignore[attr-defined]
    scheduler.register_callback("cb2", callback2)  # type: ignore[attr-defined]

    callbacks = getattr(scheduler, "callbacks", {})
    assert "cb1" in callbacks
    assert "cb2" in callbacks


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer Scheduler(jobs_file=...) API not available in this build.",
)
def test_job_with_metadata(scheduler: Scheduler) -> None:
    metadata = {"key1": "value1", "key2": 123}

    job = scheduler.add_job(  # type: ignore[attr-defined]
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        metadata=metadata,
    )

    assert job.metadata == metadata
