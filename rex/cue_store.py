"""Cue store for conversational follow-up prompts."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data/followups")


def _ensure_data_dir() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class Cue(BaseModel):
    """Represents a conversational follow-up cue."""

    cue_id: str = Field(default_factory=lambda: f"cue_{uuid.uuid4().hex[:12]}")
    prompt: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    asked_at: datetime | None = None
    dismissed_at: datetime | None = None
    due_at: datetime | None = None
    source: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_asked(self, at: datetime | None = None) -> None:
        now = _ensure_utc(at or _utc_now())
        self.status = "asked"
        self.asked_at = now
        self.updated_at = now

    def dismiss(self, at: datetime | None = None) -> None:
        now = _ensure_utc(at or _utc_now())
        self.status = "dismissed"
        self.dismissed_at = now
        self.updated_at = now

    def is_due(self, now: datetime) -> bool:
        if self.status != "pending":
            return False
        if self.due_at is None:
            return True
        return _ensure_utc(self.due_at) <= now

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        for key in ("created_at", "updated_at", "asked_at", "dismissed_at", "due_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = _ensure_utc(value).isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cue":
        payload = dict(data)
        for key in ("created_at", "updated_at", "asked_at", "dismissed_at", "due_at"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                try:
                    payload[key] = _ensure_utc(datetime.fromisoformat(value))
                except ValueError:
                    payload[key] = None
        return cls(**payload)


class CueStore:
    """Persistent storage for conversational follow-up cues."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "cues.json"
        self.storage_path = Path(storage_path)
        self._cues: dict[str, Cue] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._cues = {}
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cue store: %s", exc)
            self._cues = {}
            return
        cues = {}
        for item in raw.get("cues", []):
            try:
                cue = Cue.from_dict(item)
                cues[cue.cue_id] = cue
            except Exception as exc:
                logger.debug("Skipping invalid cue entry: %s", exc)
        self._cues = cues

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"cues": [cue.to_dict() for cue in self._cues.values()]}
        try:
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to save cue store: %s", exc)

    def list_cues(self, status: str | None = None) -> list[Cue]:
        cues = list(self._cues.values())
        if status:
            cues = [cue for cue in cues if cue.status == status]
        return sorted(cues, key=lambda cue: cue.created_at)

    def find_by_source(self, source: str, source_id: str) -> Cue | None:
        for cue in self._cues.values():
            if cue.source == source and cue.source_id == source_id:
                return cue
        return None

    def add_cue(
        self,
        prompt: str,
        *,
        source: str | None = None,
        source_id: str | None = None,
        due_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Cue:
        if source and source_id:
            existing = self.find_by_source(source, source_id)
            if existing is not None:
                return existing
        cue = Cue(
            prompt=prompt,
            source=source,
            source_id=source_id,
            due_at=_ensure_utc(due_at) if due_at else None,
            metadata=metadata or {},
        )
        self._cues[cue.cue_id] = cue
        self._save()
        return cue

    def mark_asked(self, cue_id: str, *, at: datetime | None = None) -> bool:
        cue = self._cues.get(cue_id)
        if cue is None:
            return False
        cue.mark_asked(at=at)
        self._save()
        return True

    def dismiss(self, cue_id: str, *, at: datetime | None = None) -> bool:
        cue = self._cues.get(cue_id)
        if cue is None:
            return False
        cue.dismiss(at=at)
        self._save()
        return True

    def get_due_cues(self, now: datetime, *, limit: int | None = None) -> list[Cue]:
        now_utc = _ensure_utc(now)
        cues = [cue for cue in self._cues.values() if cue.is_due(now_utc)]
        cues.sort(key=lambda cue: (cue.due_at or cue.created_at))
        if limit is not None:
            cues = cues[: max(0, limit)]
        return cues

    def prune_expired(self, *, expire_hours: int, now: datetime | None = None) -> int:
        if expire_hours <= 0:
            return 0
        cutoff = _ensure_utc(now or _utc_now()) - timedelta(hours=expire_hours)
        to_delete = [
            cue_id
            for cue_id, cue in self._cues.items()
            if _ensure_utc(cue.created_at) < cutoff
        ]
        for cue_id in to_delete:
            del self._cues[cue_id]
        if to_delete:
            self._save()
        return len(to_delete)

    def all_cues(self) -> Iterable[Cue]:
        return list(self._cues.values())


_cue_store: CueStore | None = None


def get_cue_store() -> CueStore:
    global _cue_store
    if _cue_store is None:
        _cue_store = CueStore()
    return _cue_store


def set_cue_store(store: CueStore | None) -> None:
    global _cue_store
    _cue_store = store


__all__ = ["Cue", "CueStore", "get_cue_store", "set_cue_store"]
