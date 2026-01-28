"""
Scheduler module for Rex AI Assistant.

Provides a job scheduling system with persistent storage, recurring jobs,
and integration with the event bus and workflow system.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScheduledJob(BaseModel):
    """A scheduled job definition."""

    job_id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Human-readable job name")
    schedule: str = Field(..., description="Schedule specification (e.g., 'interval:600' for 10 minutes)")
    enabled: bool = Field(default=True, description="Whether the job is enabled")
    next_run: datetime = Field(..., description="Next scheduled run time")
    max_runs: Optional[int] = Field(default=None, description="Maximum number of times to run (None for unlimited)")
    run_count: int = Field(default=0, description="Number of times this job has run")
    callback_name: Optional[str] = Field(default=None, description="Name of registered callback function")
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID to trigger")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Scheduler:
    """
    Job scheduler with persistent storage and background execution.

    Features:
    - Interval-based scheduling (e.g., every 10 minutes)
    - Persistent job definitions
    - Callback execution or workflow triggering
    - Thread-safe job management
    """

    def __init__(self, jobs_file: Optional[Path] = None):
        """
        Initialize the scheduler.

        Args:
            jobs_file: Path to jobs definition file. Defaults to data/scheduler/jobs.json
        """
        self.jobs_file = jobs_file or Path("data/scheduler/jobs.json")
        self.jobs: dict[str, ScheduledJob] = {}
        self.callbacks: dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Ensure directory exists
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing jobs
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load jobs from persistent storage."""
        if not self.jobs_file.exists():
            logger.info(f"No existing jobs file at {self.jobs_file}")
            return

        try:
            with open(self.jobs_file, 'r') as f:
                jobs_data = json.load(f)

            for job_data in jobs_data:
                # Parse datetime from ISO format
                if 'next_run' in job_data and isinstance(job_data['next_run'], str):
                    job_data['next_run'] = datetime.fromisoformat(job_data['next_run'])

                job = ScheduledJob(**job_data)
                self.jobs[job.job_id] = job

            logger.info(f"Loaded {len(self.jobs)} jobs from {self.jobs_file}")
        except Exception as e:
            logger.error(f"Failed to load jobs from {self.jobs_file}: {e}")

    def _save_jobs(self) -> None:
        """Save jobs to persistent storage."""
        try:
            jobs_data = [job.model_dump() for job in self.jobs.values()]

            # Convert datetime to ISO format for JSON serialization
            for job_data in jobs_data:
                if isinstance(job_data.get('next_run'), datetime):
                    job_data['next_run'] = job_data['next_run'].isoformat()

            with open(self.jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)

            logger.debug(f"Saved {len(self.jobs)} jobs to {self.jobs_file}")
        except Exception as e:
            logger.error(f"Failed to save jobs to {self.jobs_file}: {e}")

    def register_callback(self, name: str, callback: Callable) -> None:
        """
        Register a callback function that can be invoked by jobs.

        Args:
            name: Callback name
            callback: Callable to invoke when job runs
        """
        with self._lock:
            self.callbacks[name] = callback
            logger.debug(f"Registered callback: {name}")

    def add_job(
        self,
        job_id: str,
        name: str,
        schedule: str,
        enabled: bool = True,
        callback_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        max_runs: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> ScheduledJob:
        """
        Add a new scheduled job.

        Args:
            job_id: Unique job identifier
            name: Human-readable job name
            schedule: Schedule specification (format: 'interval:seconds')
            enabled: Whether job is enabled
            callback_name: Name of registered callback
            workflow_id: Workflow ID to trigger
            max_runs: Maximum runs (None for unlimited)
            metadata: Additional metadata

        Returns:
            Created ScheduledJob
        """
        with self._lock:
            # Calculate next run based on schedule
            next_run = self._calculate_next_run(schedule)

            job = ScheduledJob(
                job_id=job_id,
                name=name,
                schedule=schedule,
                enabled=enabled,
                next_run=next_run,
                max_runs=max_runs,
                run_count=0,
                callback_name=callback_name,
                workflow_id=workflow_id,
                metadata=metadata or {}
            )

            self.jobs[job_id] = job
            self._save_jobs()

            logger.info(f"Added job: {job_id} ({name}) - next run: {next_run}")
            return job

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a scheduled job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was removed, False if not found
        """
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self._save_jobs()
                logger.info(f"Removed job: {job_id}")
                return True
            return False

    def list_jobs(self) -> list[ScheduledJob]:
        """
        List all scheduled jobs.

        Returns:
            List of ScheduledJob instances
        """
        with self._lock:
            return list(self.jobs.values())

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """
        Get a specific job by ID.

        Args:
            job_id: Job identifier

        Returns:
            ScheduledJob if found, None otherwise
        """
        with self._lock:
            return self.jobs.get(job_id)

    def update_job(self, job_id: str, **updates) -> Optional[ScheduledJob]:
        """
        Update a job's properties.

        Args:
            job_id: Job identifier
            **updates: Properties to update

        Returns:
            Updated ScheduledJob if found, None otherwise
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None

            # Update fields
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)

            self._save_jobs()
            logger.info(f"Updated job: {job_id}")
            return job

    def _calculate_next_run(self, schedule: str, from_time: Optional[datetime] = None) -> datetime:
        """
        Calculate next run time based on schedule specification.

        Args:
            schedule: Schedule specification (format: 'interval:seconds')
            from_time: Base time for calculation (defaults to now)

        Returns:
            Next run datetime
        """
        from_time = from_time or datetime.now()

        # Parse schedule - currently supports interval format
        if schedule.startswith('interval:'):
            seconds = int(schedule.split(':')[1])
            return from_time + timedelta(seconds=seconds)

        # Default to 1 hour if format not recognized
        logger.warning(f"Unknown schedule format: {schedule}, defaulting to 1 hour")
        return from_time + timedelta(hours=1)

    def update_next_run(self, job_id: str) -> None:
        """
        Update the next_run time for a job after execution.

        Args:
            job_id: Job identifier
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return

            job.next_run = self._calculate_next_run(job.schedule, from_time=datetime.now())
            self._save_jobs()

    def run_job(self, job_id: str, force: bool = False) -> bool:
        """
        Execute a job immediately.

        Args:
            job_id: Job identifier
            force: If True, run even if disabled or max_runs exceeded

        Returns:
            True if job was executed, False otherwise
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                logger.warning(f"Job not found: {job_id}")
                return False

            # Check if job should run
            if not force:
                if not job.enabled:
                    logger.debug(f"Job {job_id} is disabled, skipping")
                    return False

                if job.max_runs is not None and job.run_count >= job.max_runs:
                    logger.debug(f"Job {job_id} has reached max_runs ({job.max_runs}), skipping")
                    return False

            logger.info(f"Running job: {job_id} ({job.name})")

            try:
                # Execute callback or trigger workflow
                if job.callback_name and job.callback_name in self.callbacks:
                    callback = self.callbacks[job.callback_name]
                    callback(job)
                elif job.workflow_id:
                    logger.info(f"Job {job_id} would trigger workflow {job.workflow_id}")
                    # Workflow triggering will be implemented when integrated
                else:
                    logger.warning(f"Job {job_id} has no callback or workflow configured")

                # Update job state
                job.run_count += 1
                self.update_next_run(job_id)

                logger.info(f"Job {job_id} completed successfully (run {job.run_count})")
                return True

            except Exception as e:
                logger.error(f"Error running job {job_id}: {e}", exc_info=True)
                # Still update next run even on error
                self.update_next_run(job_id)
                return False

    def start(self) -> None:
        """Start the scheduler background thread."""
        if self._running:
            logger.warning("Scheduler is already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="SchedulerThread")
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler background thread."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    def _run_loop(self) -> None:
        """Background thread loop that checks and runs due jobs."""
        logger.info("Scheduler loop starting")

        while self._running:
            try:
                now = datetime.now()

                # Check each job
                with self._lock:
                    jobs_to_run = [
                        job for job in self.jobs.values()
                        if job.enabled and job.next_run <= now
                    ]

                # Run due jobs
                for job in jobs_to_run:
                    self.run_job(job.job_id)

                # Sleep for a short interval before checking again
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                time.sleep(5)  # Back off on error

        logger.info("Scheduler loop exiting")


# Global scheduler instance
_scheduler: Optional[Scheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> Scheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                _scheduler = Scheduler()
    return _scheduler


def set_scheduler(scheduler: Scheduler) -> None:
    """Set the global scheduler instance (for testing)."""
    global _scheduler
    with _scheduler_lock:
        _scheduler = scheduler
