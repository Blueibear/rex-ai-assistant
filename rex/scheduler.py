"""Job scheduler with persistence and manual execution."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

JobHandler = Callable[["JobDefinition"], None]

_DATA_DIR = Path("data/scheduler")


def _ensure_data_dir() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class JobDefinition:
    """Represents a scheduled job."""

    job_id: str
    name: str
    interval_seconds: int
    handler_name: str
    enabled: bool = True
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "interval_seconds": self.interval_seconds,
            "handler_name": self.handler_name,
            "enabled": self.enabled,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobDefinition":
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            interval_seconds=int(data["interval_seconds"]),
            handler_name=data["handler_name"],
            enabled=bool(data.get("enabled", True)),
            last_run_at=_parse_datetime(data.get("last_run_at")),
            next_run_at=_parse_datetime(data.get("next_run_at")),
            metadata=dict(data.get("metadata", {})),
        )


class Scheduler:
    """Scheduler that persists job definitions to disk."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        *,
        now_func: Callable[[], datetime] | None = None,
    ) -> None:
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "jobs.json"
        self.storage_path = Path(storage_path)
        self._jobs: dict[str, JobDefinition] = {}
        self._handlers: dict[str, JobHandler] = {}
        self._now = now_func or _utc_now
        self._load()

    def register_handler(self, name: str, handler: JobHandler) -> None:
        self._handlers[name] = handler

    def add_job(
        self,
        name: str,
        interval_seconds: int,
        handler_name: str,
        *,
        job_id: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> JobDefinition:
        job_id = job_id or str(uuid.uuid4())
        now = self._now()
        job = JobDefinition(
            job_id=job_id,
            name=name,
            interval_seconds=interval_seconds,
            handler_name=handler_name,
            enabled=enabled,
            last_run_at=None,
            next_run_at=now + timedelta(seconds=interval_seconds),
            metadata=metadata or {},
        )
        self._jobs[job.job_id] = job
        self._save()
        return job

    def list_jobs(self) -> list[JobDefinition]:
        return list(self._jobs.values())

    def get_job(self, job_id: str) -> JobDefinition | None:
        return self._jobs.get(job_id)

    def find_job_by_name(self, name: str) -> JobDefinition | None:
        for job in self._jobs.values():
            if job.name == name:
                return job
        return None

    def run_job(self, job_id: str, *, manual: bool = False) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if not job.enabled and not manual:
            return False
        handler = self._handlers.get(job.handler_name)
        if handler is not None:
            handler(job)
        now = self._now()
        job.last_run_at = now
        job.next_run_at = now + timedelta(seconds=job.interval_seconds)
        self._save()
        return True

    def run_due_jobs(self) -> list[str]:
        now = self._now()
        ran: list[str] = []
        for job in self._jobs.values():
            if not job.enabled or job.next_run_at is None:
                continue
            if job.next_run_at <= now:
                if self.run_job(job.job_id):
                    ran.append(job.job_id)
        return ran

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        jobs = data.get("jobs", [])
        for item in jobs:
            job = JobDefinition.from_dict(item)
            if job.next_run_at is None:
                job.next_run_at = self._now() + timedelta(seconds=job.interval_seconds)
            self._jobs[job.job_id] = job

    def _save(self) -> None:
        payload = {"jobs": [job.to_dict() for job in self._jobs.values()]}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["JobDefinition", "Scheduler"]
