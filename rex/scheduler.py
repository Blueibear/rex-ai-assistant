"""
Scheduler module for Rex AI Assistant.

This file is a merge-resolved, compatibility-first scheduler that supports both
older "JobDefinition" style jobs and newer "ScheduledJob" style jobs.

Supported APIs (both):
- Legacy:
    scheduler.register_handler(name, handler)
    job = scheduler.add_job(name, interval_seconds, handler_name, job_id=..., enabled=..., metadata=...)
    scheduler.list_jobs() -> list[JobDefinition/ScheduledJob]
    scheduler.run_job(job_id, manual=True|False) -> bool
    scheduler.run_due_jobs() -> list[str]
    scheduler.get_job(job_id) -> JobDefinition|None

- Newer:
    scheduler.register_callback(name, callback)
    job = scheduler.add_job(job_id=..., name=..., schedule="interval:600", callback_name=..., workflow_id=..., ...)
    scheduler.run_job(job_id, force=True|False) -> bool
    scheduler.start() / scheduler.stop() for background loop (optional)
    scheduler.update_job(job_id, **updates)

Persistence:
- Defaults to: data/scheduler/jobs.json
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Handler signature used by both styles:
# We pass the job object itself to the callback.
JobCallback = Callable[["ScheduledJob"], None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_tz(value)
    if isinstance(value, str) and value.strip():
        try:
            return _ensure_tz(datetime.fromisoformat(value))
        except ValueError:
            return None
    return None


def _schedule_from_interval(interval_seconds: int) -> str:
    return f"interval:{int(interval_seconds)}"


def _interval_from_schedule(schedule: str) -> int:
    s = (schedule or "").strip().lower()
    if s.startswith("interval:"):
        try:
            return int(s.split(":", 1)[1].strip())
        except ValueError:
            return 3600
    return 3600


def _calculate_next_run(schedule: str, from_time: Optional[datetime] = None) -> datetime:
    base = _ensure_tz(from_time or _utc_now())
    seconds = _interval_from_schedule(schedule)
    return base + timedelta(seconds=seconds)


@dataclass
class ScheduledJob:
    """
    Unified job model.

    Fields cover the newer scheduler, but we provide legacy-compatible aliases:
    - interval_seconds property (derived from schedule)
    - handler_name alias for callback_name
    - next_run_at alias for next_run
    - last_run_at used by legacy systems
    """

    # Required
    job_id: str
    name: str

    # Scheduling
    schedule: str = field(default_factory=lambda: "interval:3600")
    enabled: bool = True
    next_run: datetime = field(default_factory=_utc_now)
    last_run_at: Optional[datetime] = None

    # Execution
    callback_name: Optional[str] = None
    workflow_id: Optional[str] = None

    # Limits and stats
    max_runs: Optional[int] = None
    run_count: int = 0

    # Extra
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------- Legacy compatibility aliases ----------

    @property
    def interval_seconds(self) -> int:
        return _interval_from_schedule(self.schedule)

    @property
    def handler_name(self) -> str:
        # Legacy expects a string handler_name. Prefer callback_name; fallback to empty.
        return self.callback_name or ""

    @property
    def next_run_at(self) -> datetime:
        return self.next_run

    @next_run_at.setter
    def next_run_at(self, value: datetime) -> None:
        self.next_run = _ensure_tz(value)

    # ---------- Serialization ----------

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "next_run": _ensure_tz(self.next_run).isoformat(),
            "last_run_at": _ensure_tz(self.last_run_at).isoformat() if self.last_run_at else None,
            "max_runs": self.max_runs,
            "run_count": self.run_count,
            "callback_name": self.callback_name,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledJob":
        # Accept both legacy and newer keys.
        job_id = data.get("job_id") or data.get("id") or str(uuid.uuid4())
        name = data.get("name") or "Unnamed Job"

        # Legacy: interval_seconds + handler_name + next_run_at
        interval_seconds = data.get("interval_seconds")
        handler_name = data.get("handler_name")
        next_run_at = data.get("next_run_at")

        # New: schedule + callback_name + next_run
        schedule = data.get("schedule")
        callback_name = data.get("callback_name")

        enabled = bool(data.get("enabled", True))
        metadata = dict(data.get("metadata", {}) or {})

        # Decide schedule
        if not schedule:
            if interval_seconds is not None:
                schedule = _schedule_from_interval(int(interval_seconds))
            else:
                schedule = "interval:3600"

        # Decide callback_name
        if not callback_name:
            if handler_name:
                callback_name = str(handler_name)

        # Parse times
        next_run = _parse_dt(data.get("next_run")) or _parse_dt(next_run_at)
        last_run = _parse_dt(data.get("last_run_at"))

        # If next_run missing, compute it
        if next_run is None:
            next_run = _calculate_next_run(schedule)

        return cls(
            job_id=str(job_id),
            name=str(name),
            schedule=str(schedule),
            enabled=enabled,
            next_run=_ensure_tz(next_run),
            last_run_at=_ensure_tz(last_run) if last_run else None,
            max_runs=data.get("max_runs"),
            run_count=int(data.get("run_count", 0) or 0),
            callback_name=callback_name,
            workflow_id=data.get("workflow_id"),
            metadata=metadata,
        )


# Backwards-compatible export name used in older code
JobDefinition = ScheduledJob


class Scheduler:
    """
    Job scheduler with persistent storage.

    You can use it in two ways:
    - Manually: call run_due_jobs() periodically.
    - Background: call start() to run due jobs in a daemon thread.

    Note: Workflow triggering is stubbed. If a job has workflow_id but no callback,
    it will log and return success, leaving actual workflow integration to the
    workflow runner layer.
    """

    def __init__(
        self,
        jobs_file: Optional[Path] = None,
        *,
        now_func: Optional[Callable[[], datetime]] = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.jobs_file = jobs_file or Path("data/scheduler/jobs.json")
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)

        self._now = now_func or _utc_now
        self._poll_interval = float(poll_interval_seconds)

        self._lock = threading.RLock()
        self._jobs: dict[str, ScheduledJob] = {}
        self._callbacks: dict[str, JobCallback] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._load_jobs()

    # -------------------------
    # Persistence
    # -------------------------

    def _load_jobs(self) -> None:
        if not self.jobs_file.exists():
            return
        try:
            raw = json.loads(self.jobs_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to read jobs file %s: %s", self.jobs_file, e)
            return

        # Accept either:
        # - legacy: {"jobs": [ ... ]}
        # - newer:  [ ... ]
        items: list[dict[str, Any]]
        if isinstance(raw, dict) and isinstance(raw.get("jobs"), list):
            items = raw["jobs"]
        elif isinstance(raw, list):
            items = raw
        else:
            items = []

        loaded = 0
        with self._lock:
            self._jobs.clear()
            for item in items:
                if not isinstance(item, dict):
                    continue
                job = ScheduledJob.from_dict(item)
                # Ensure next_run is sane
                if job.next_run is None:
                    job.next_run = _calculate_next_run(job.schedule, from_time=self._now())
                self._jobs[job.job_id] = job
                loaded += 1

        logger.info("Loaded %d job(s) from %s", loaded, self.jobs_file)

    def _save_jobs(self) -> None:
        with self._lock:
            payload = [job.to_dict() for job in self._jobs.values()]

        try:
            self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
            self.jobs_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save jobs to %s: %s", self.jobs_file, e)

    # -------------------------
    # Registration
    # -------------------------

    def register_handler(self, name: str, handler: JobCallback) -> None:
        """Legacy name for registering callbacks."""
        self.register_callback(name, handler)

    def register_callback(self, name: str, callback: JobCallback) -> None:
        with self._lock:
            self._callbacks[str(name)] = callback
            logger.debug("Registered scheduler callback: %s", name)

    # -------------------------
    # CRUD
    # -------------------------

    def add_job(
        self,
        *args,
        **kwargs,
    ) -> ScheduledJob:
        """
        Supports both call styles.

        Legacy style:
            add_job(name, interval_seconds, handler_name, job_id=None, enabled=True, metadata=None)

        New style:
            add_job(
                job_id="...",
                name="...",
                schedule="interval:600",
                enabled=True,
                callback_name="...",
                workflow_id=None,
                max_runs=None,
                metadata=None,
            )
        """
        # Detect legacy positional signature
        if len(args) >= 3 and isinstance(args[0], str) and isinstance(args[1], (int, float)) and isinstance(args[2], str):
            name = args[0]
            interval_seconds = int(args[1])
            handler_name = args[2]

            job_id = kwargs.get("job_id") or str(uuid.uuid4())
            enabled = bool(kwargs.get("enabled", True))
            metadata = kwargs.get("metadata") or {}

            schedule = _schedule_from_interval(interval_seconds)
            now = _ensure_tz(self._now())
            job = ScheduledJob(
                job_id=str(job_id),
                name=str(name),
                schedule=schedule,
                enabled=enabled,
                next_run=_calculate_next_run(schedule, from_time=now),
                last_run_at=None,
                callback_name=str(handler_name) if handler_name else None,
                workflow_id=None,
                max_runs=None,
                run_count=0,
                metadata=dict(metadata),
            )

            with self._lock:
                self._jobs[job.job_id] = job
            self._save_jobs()
            return job

        # New style kwargs
        job_id = str(kwargs.get("job_id") or str(uuid.uuid4()))
        name = str(kwargs.get("name") or "Unnamed Job")
        schedule = str(kwargs.get("schedule") or "interval:3600")
        enabled = bool(kwargs.get("enabled", True))
        callback_name = kwargs.get("callback_name")
        workflow_id = kwargs.get("workflow_id")
        max_runs = kwargs.get("max_runs")
        metadata = kwargs.get("metadata") or {}

        now = _ensure_tz(self._now())
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            schedule=schedule,
            enabled=enabled,
            next_run=_calculate_next_run(schedule, from_time=now),
            last_run_at=None,
            callback_name=str(callback_name) if callback_name else None,
            workflow_id=str(workflow_id) if workflow_id else None,
            max_runs=max_runs,
            run_count=0,
            metadata=dict(metadata),
        )

        with self._lock:
            self._jobs[job.job_id] = job
        self._save_jobs()
        return job

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            existed = job_id in self._jobs
            if existed:
                del self._jobs[job_id]
        if existed:
            self._save_jobs()
        return existed

    def list_jobs(self) -> list[ScheduledJob]:
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def find_job_by_name(self, name: str) -> Optional[ScheduledJob]:
        with self._lock:
            for job in self._jobs.values():
                if job.name == name:
                    return job
        return None

    def update_job(self, job_id: str, **updates: Any) -> Optional[ScheduledJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            # Apply updates conservatively to known fields
            for key, value in updates.items():
                if not hasattr(job, key):
                    continue
                if key in {"next_run", "last_run_at"} and isinstance(value, str):
                    parsed = _parse_dt(value)
                    if parsed:
                        setattr(job, key, parsed)
                    continue
                setattr(job, key, value)

            # If schedule changed, recompute next_run from now
            if "schedule" in updates:
                job.next_run = _calculate_next_run(job.schedule, from_time=self._now())

        self._save_jobs()
        return job

    # -------------------------
    # Execution
    # -------------------------

    def run_job(self, job_id: str, *, manual: bool = False, force: bool = False) -> bool:
        """
        Execute a job immediately.

        Compatibility:
        - Legacy callers pass manual=True.
        - New callers pass force=True.

        Rules:
        - If force or manual, ignore enabled and max_runs checks.
        - Otherwise, skip if disabled or max_runs reached.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning("Job not found: %s", job_id)
                return False

            bypass = bool(force or manual)

            if not bypass:
                if not job.enabled:
                    return False
                if job.max_runs is not None and job.run_count >= job.max_runs:
                    return False

            callback_name = job.callback_name

        # Execute outside lock
        ok = True
        try:
            if callback_name and callback_name in self._callbacks:
                self._callbacks[callback_name](job)
            elif job.workflow_id:
                logger.info("Job %s would trigger workflow %s (integration stub)", job.job_id, job.workflow_id)
            else:
                logger.warning("Job %s has no callback_name or workflow_id", job.job_id)
        except Exception as e:
            ok = False
            logger.error("Error running job %s: %s", job_id, e, exc_info=True)

        # Update job timing and counts
        with self._lock:
            now = _ensure_tz(self._now())
            job.last_run_at = now
            job.run_count += 1
            job.next_run = _calculate_next_run(job.schedule, from_time=now)

        self._save_jobs()
        return ok

    def run_due_jobs(self) -> list[str]:
        """Run all jobs whose next_run is due. Returns list of job_ids that were executed."""
        now = _ensure_tz(self._now())
        due: list[str] = []

        with self._lock:
            jobs_snapshot = list(self._jobs.values())

        for job in jobs_snapshot:
            if not job.enabled:
                continue
            if job.max_runs is not None and job.run_count >= job.max_runs:
                continue
            if _ensure_tz(job.next_run) <= now:
                if self.run_job(job.job_id):
                    due.append(job.job_id)

        return due

    # -------------------------
    # Background loop (optional)
    # -------------------------

    def start(self) -> None:
        if self._running:
            logger.warning("Scheduler already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="RexScheduler")
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        t = self._thread
        if t:
            t.join(timeout=5.0)
        logger.info("Scheduler stopped")

    def _run_loop(self) -> None:
        logger.info("Scheduler loop starting")
        while self._running:
            try:
                self.run_due_jobs()
                time.sleep(self._poll_interval)
            except Exception as e:
                logger.error("Scheduler loop error: %s", e, exc_info=True)
                time.sleep(max(self._poll_interval, 5.0))
        logger.info("Scheduler loop exiting")


# Global scheduler instance
_SCHEDULER: Optional[Scheduler] = None
_SCHEDULER_LOCK = threading.Lock()


def get_scheduler() -> Scheduler:
    """Get the global scheduler instance."""
    global _SCHEDULER
    if _SCHEDULER is None:
        with _SCHEDULER_LOCK:
            if _SCHEDULER is None:
                _SCHEDULER = Scheduler()
    return _SCHEDULER


def set_scheduler(scheduler: Scheduler) -> None:
    """Set the global scheduler instance (for testing)."""
    global _SCHEDULER
    with _SCHEDULER_LOCK:
        _SCHEDULER = scheduler


__all__ = ["ScheduledJob", "JobDefinition", "Scheduler", "get_scheduler", "set_scheduler"]

