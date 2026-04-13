"""
US-036: Scheduler
Acceptance criteria:
- scheduler initializes
- tasks scheduled
- tasks executed
- Typecheck passes
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from rex.scheduler import ScheduledJob, Scheduler, get_scheduler, set_scheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def make_scheduler(tmp_path: Path, now_func=None) -> Scheduler:
    jobs_file = tmp_path / "jobs.json"
    return Scheduler(jobs_file=jobs_file, now_func=now_func or utc_now)


# ---------------------------------------------------------------------------
# Scheduler initializes
# ---------------------------------------------------------------------------


class TestSchedulerInitializes:
    def test_scheduler_creates_without_error(self, tmp_path):
        s = make_scheduler(tmp_path)
        assert s is not None

    def test_scheduler_starts_with_empty_job_list(self, tmp_path):
        s = make_scheduler(tmp_path)
        assert s.list_jobs() == []

    def test_scheduler_creates_jobs_file_dir(self, tmp_path):
        jobs_dir = tmp_path / "subdir" / "scheduler"
        Scheduler(jobs_file=jobs_dir / "jobs.json")
        assert jobs_dir.exists()

    def test_scheduler_not_running_on_init(self, tmp_path):
        s = make_scheduler(tmp_path)
        assert not s._running

    def test_global_scheduler_accessible(self, tmp_path):
        s = make_scheduler(tmp_path)
        set_scheduler(s)
        assert get_scheduler() is s
        set_scheduler(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tasks scheduled
# ---------------------------------------------------------------------------


class TestTasksScheduled:
    def test_add_job_returns_scheduled_job(self, tmp_path):
        s = make_scheduler(tmp_path)
        job = s.add_job(job_id="j1", name="Test Job", schedule="interval:60")
        assert isinstance(job, ScheduledJob)
        assert job.job_id == "j1"
        assert job.name == "Test Job"

    def test_add_job_appears_in_list(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.add_job(job_id="j1", name="Test Job", schedule="interval:60")
        jobs = s.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].job_id == "j1"

    def test_add_multiple_jobs(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.add_job(job_id="a", name="A", schedule="interval:60")
        s.add_job(job_id="b", name="B", schedule="interval:120")
        assert len(s.list_jobs()) == 2

    def test_get_job_by_id(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.add_job(job_id="j1", name="Test", schedule="interval:60")
        job = s.get_job("j1")
        assert job is not None
        assert job.name == "Test"

    def test_get_nonexistent_job_returns_none(self, tmp_path):
        s = make_scheduler(tmp_path)
        assert s.get_job("missing") is None

    def test_job_is_enabled_by_default(self, tmp_path):
        s = make_scheduler(tmp_path)
        job = s.add_job(job_id="j1", name="Test", schedule="interval:60")
        assert job.enabled is True

    def test_job_can_be_disabled(self, tmp_path):
        s = make_scheduler(tmp_path)
        job = s.add_job(job_id="j1", name="Test", schedule="interval:60", enabled=False)
        assert job.enabled is False

    def test_legacy_add_job_style(self, tmp_path):
        s = make_scheduler(tmp_path)
        called = []
        s.register_handler("my_handler", lambda j: called.append(j.job_id))
        job = s.add_job("Legacy Job", 300, "my_handler", job_id="legacy1")
        assert job.job_id == "legacy1"
        assert job.callback_name == "my_handler"
        assert job.interval_seconds == 300

    def test_job_persists_to_file(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.add_job(job_id="j1", name="Persistent", schedule="interval:60")
        # Load a new scheduler from same file
        s2 = Scheduler(jobs_file=s.jobs_file)
        assert s2.get_job("j1") is not None
        assert s2.get_job("j1").name == "Persistent"  # type: ignore[union-attr]

    def test_update_job_changes_schedule(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.add_job(job_id="j1", name="Test", schedule="interval:60")
        s.update_job("j1", schedule="interval:120")
        assert s.get_job("j1").interval_seconds == 120  # type: ignore[union-attr]

    def test_remove_job(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.add_job(job_id="j1", name="Test", schedule="interval:60")
        removed = s.remove_job("j1")
        assert removed is True
        assert s.get_job("j1") is None


# ---------------------------------------------------------------------------
# Tasks executed
# ---------------------------------------------------------------------------


class TestTasksExecuted:
    def test_run_job_manual_invokes_callback(self, tmp_path):
        s = make_scheduler(tmp_path)
        called = []
        s.register_callback("cb", lambda j: called.append(j.job_id))
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="cb")
        result = s.run_job("j1", manual=True)
        assert result is True
        assert called == ["j1"]

    def test_run_job_force_invokes_callback(self, tmp_path):
        s = make_scheduler(tmp_path)
        called = []
        s.register_callback("cb", lambda j: called.append(j.job_id))
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="cb")
        result = s.run_job("j1", force=True)
        assert result is True
        assert called == ["j1"]

    def test_run_due_jobs_executes_overdue_job(self, tmp_path):
        # Use a now_func that reports 1 hour in the future so jobs are immediately due
        future = utc_now() + timedelta(hours=1)
        s = Scheduler(jobs_file=tmp_path / "jobs.json", now_func=lambda: future)
        called = []
        s.register_callback("cb", lambda j: called.append(j.job_id))
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="cb")
        # Manually set next_run to the past so it's due
        job = s.get_job("j1")
        assert job is not None
        job.next_run = utc_now() - timedelta(seconds=10)
        executed = s.run_due_jobs()
        assert "j1" in executed

    def test_disabled_job_not_executed_by_run_due_jobs(self, tmp_path):
        future = utc_now() + timedelta(hours=1)
        s = Scheduler(jobs_file=tmp_path / "jobs.json", now_func=lambda: future)
        called = []
        s.register_callback("cb", lambda j: called.append(j.job_id))
        s.add_job(
            job_id="j1", name="Test", schedule="interval:60", callback_name="cb", enabled=False
        )
        job = s.get_job("j1")
        assert job is not None
        job.next_run = utc_now() - timedelta(seconds=10)
        executed = s.run_due_jobs()
        assert "j1" not in executed
        assert called == []

    def test_run_job_updates_run_count(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.register_callback("cb", lambda j: None)
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="cb")
        s.run_job("j1", manual=True)
        assert s.get_job("j1").run_count == 1  # type: ignore[union-attr]

    def test_run_job_updates_last_run_at(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.register_callback("cb", lambda j: None)
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="cb")
        s.run_job("j1", manual=True)
        assert s.get_job("j1").last_run_at is not None  # type: ignore[union-attr]

    def test_failing_callback_returns_false(self, tmp_path):
        s = make_scheduler(tmp_path)

        def bad_cb(j):
            raise RuntimeError("oops")

        s.register_callback("bad", bad_cb)
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="bad")
        result = s.run_job("j1", manual=True)
        assert result is False

    def test_failing_callback_does_not_remove_job(self, tmp_path):
        s = make_scheduler(tmp_path)

        def bad_cb(j):
            raise RuntimeError("oops")

        s.register_callback("bad", bad_cb)
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="bad")
        s.run_job("j1", manual=True)
        assert s.get_job("j1") is not None

    def test_start_stop_background_loop(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.start()
        assert s._running is True
        s.stop()
        assert s._running is False

    def test_metrics_tracked(self, tmp_path):
        s = make_scheduler(tmp_path)
        s.register_callback("cb", lambda j: None)
        s.add_job(job_id="j1", name="Test", schedule="interval:60", callback_name="cb")
        s.run_job("j1", manual=True)
        metrics = s.get_metrics()
        assert metrics["total_runs"] >= 1
        assert metrics["successful_runs"] >= 1
