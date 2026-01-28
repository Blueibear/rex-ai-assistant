"""Tests for scheduler module."""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rex.scheduler import Scheduler, ScheduledJob


@pytest.fixture
def temp_jobs_file(tmp_path):
    """Create a temporary jobs file."""
    return tmp_path / "test_jobs.json"


@pytest.fixture
def scheduler(temp_jobs_file):
    """Create a test scheduler instance."""
    return Scheduler(jobs_file=temp_jobs_file)


def test_scheduler_initialization(temp_jobs_file):
    """Test scheduler initializes correctly."""
    scheduler = Scheduler(jobs_file=temp_jobs_file)
    assert scheduler.jobs == {}
    assert scheduler.jobs_file == temp_jobs_file


def test_add_job(scheduler):
    """Test adding a job."""
    job = scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        enabled=True
    )

    assert isinstance(job, ScheduledJob)
    assert job.job_id == "test-job"
    assert job.name == "Test Job"
    assert job.schedule == "interval:60"
    assert job.enabled is True
    assert job.run_count == 0
    assert job.next_run > datetime.now()


def test_add_job_with_callback(scheduler):
    """Test adding a job with callback."""
    def test_callback(job):
        pass

    scheduler.register_callback("test_cb", test_callback)

    job = scheduler.add_job(
        job_id="test-job-cb",
        name="Test Job with Callback",
        schedule="interval:60",
        callback_name="test_cb"
    )

    assert job.callback_name == "test_cb"


def test_add_job_with_workflow(scheduler):
    """Test adding a job with workflow ID."""
    job = scheduler.add_job(
        job_id="test-job-wf",
        name="Test Job with Workflow",
        schedule="interval:60",
        workflow_id="workflow-123"
    )

    assert job.workflow_id == "workflow-123"


def test_remove_job(scheduler):
    """Test removing a job."""
    scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60"
    )

    assert scheduler.remove_job("test-job") is True
    assert scheduler.get_job("test-job") is None
    assert scheduler.remove_job("non-existent") is False


def test_list_jobs(scheduler):
    """Test listing jobs."""
    scheduler.add_job(
        job_id="job1",
        name="Job 1",
        schedule="interval:60"
    )
    scheduler.add_job(
        job_id="job2",
        name="Job 2",
        schedule="interval:120"
    )

    jobs = scheduler.list_jobs()
    assert len(jobs) == 2
    assert {job.job_id for job in jobs} == {"job1", "job2"}


def test_get_job(scheduler):
    """Test getting a specific job."""
    scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60"
    )

    job = scheduler.get_job("test-job")
    assert job is not None
    assert job.job_id == "test-job"

    assert scheduler.get_job("non-existent") is None


def test_update_job(scheduler):
    """Test updating a job."""
    scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60"
    )

    updated = scheduler.update_job("test-job", name="Updated Job", enabled=False)
    assert updated is not None
    assert updated.name == "Updated Job"
    assert updated.enabled is False

    assert scheduler.update_job("non-existent", name="Foo") is None


def test_run_job_with_callback(scheduler):
    """Test running a job with callback."""
    executed = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)

    job = scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        callback_name="test_cb"
    )

    initial_run_count = job.run_count
    initial_next_run = job.next_run

    result = scheduler.run_job("test-job")

    assert result is True
    assert "test-job" in executed

    # Check job state was updated
    job = scheduler.get_job("test-job")
    assert job.run_count == initial_run_count + 1
    assert job.next_run > initial_next_run


def test_run_job_disabled(scheduler):
    """Test running a disabled job."""
    executed = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)

    scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        callback_name="test_cb",
        enabled=False
    )

    # Without force, disabled job should not run
    result = scheduler.run_job("test-job", force=False)
    assert result is False
    assert len(executed) == 0

    # With force, disabled job should run
    result = scheduler.run_job("test-job", force=True)
    assert result is True
    assert len(executed) == 1


def test_run_job_max_runs(scheduler):
    """Test job with max_runs limit."""
    executed = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)

    scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        callback_name="test_cb",
        max_runs=2
    )

    # First run
    assert scheduler.run_job("test-job") is True
    # Second run
    assert scheduler.run_job("test-job") is True
    # Third run should fail (max_runs reached)
    assert scheduler.run_job("test-job", force=False) is False
    # But force should still work
    assert scheduler.run_job("test-job", force=True) is True

    assert len(executed) == 3


def test_calculate_next_run(scheduler):
    """Test next run calculation."""
    now = datetime.now()

    # Test interval:60 (60 seconds)
    next_run = scheduler._calculate_next_run("interval:60", from_time=now)
    expected = now + timedelta(seconds=60)
    assert abs((next_run - expected).total_seconds()) < 1

    # Test interval:3600 (1 hour)
    next_run = scheduler._calculate_next_run("interval:3600", from_time=now)
    expected = now + timedelta(seconds=3600)
    assert abs((next_run - expected).total_seconds()) < 1


def test_update_next_run(scheduler):
    """Test updating next run time."""
    job = scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60"
    )

    initial_next_run = job.next_run
    time.sleep(0.1)

    scheduler.update_next_run("test-job")

    job = scheduler.get_job("test-job")
    assert job.next_run > initial_next_run


def test_persistence(scheduler, temp_jobs_file):
    """Test job persistence to file."""
    scheduler.add_job(
        job_id="job1",
        name="Job 1",
        schedule="interval:60",
        enabled=True
    )
    scheduler.add_job(
        job_id="job2",
        name="Job 2",
        schedule="interval:120",
        enabled=False
    )

    # Create new scheduler instance with same file
    scheduler2 = Scheduler(jobs_file=temp_jobs_file)

    # Jobs should be loaded
    jobs = scheduler2.list_jobs()
    assert len(jobs) == 2

    job1 = scheduler2.get_job("job1")
    assert job1 is not None
    assert job1.name == "Job 1"
    assert job1.enabled is True

    job2 = scheduler2.get_job("job2")
    assert job2 is not None
    assert job2.name == "Job 2"
    assert job2.enabled is False


def test_scheduler_start_stop(scheduler):
    """Test starting and stopping scheduler."""
    scheduler.start()
    assert scheduler._running is True
    assert scheduler._thread is not None

    scheduler.stop()
    assert scheduler._running is False


def test_scheduled_execution(scheduler):
    """Test that jobs are executed when due."""
    executed = []

    def test_callback(job):
        executed.append(job.job_id)

    scheduler.register_callback("test_cb", test_callback)

    # Add job with very short interval (2 seconds)
    scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:2",
        callback_name="test_cb"
    )

    # Manually set next_run to the past
    job = scheduler.get_job("test-job")
    job.next_run = datetime.now() - timedelta(seconds=1)
    scheduler._save_jobs()

    # Start scheduler
    scheduler.start()

    # Wait for job to execute
    time.sleep(3)

    # Stop scheduler
    scheduler.stop()

    # Job should have been executed
    assert "test-job" in executed


def test_run_job_nonexistent(scheduler):
    """Test running a non-existent job."""
    result = scheduler.run_job("non-existent")
    assert result is False


def test_register_callback(scheduler):
    """Test registering callbacks."""
    def callback1(job):
        pass

    def callback2(job):
        pass

    scheduler.register_callback("cb1", callback1)
    scheduler.register_callback("cb2", callback2)

    assert "cb1" in scheduler.callbacks
    assert "cb2" in scheduler.callbacks


def test_job_with_metadata(scheduler):
    """Test job with additional metadata."""
    metadata = {"key1": "value1", "key2": 123}

    job = scheduler.add_job(
        job_id="test-job",
        name="Test Job",
        schedule="interval:60",
        metadata=metadata
    )

    assert job.metadata == metadata
