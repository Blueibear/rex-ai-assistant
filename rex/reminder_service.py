"""Reminder service for manual reminders and follow-up cues."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from rex.cue_store import CueStore, get_cue_store

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


class Reminder(BaseModel):
    """Represents a reminder entry."""

    reminder_id: str = Field(default_factory=lambda: f"rem_{uuid.uuid4().hex[:12]}")
    title: str
    remind_at: datetime
    status: str = "pending"
    created_at: datetime = Field(default_factory=_utc_now)
    fired_at: datetime | None = None
    done_at: datetime | None = None
    canceled_at: datetime | None = None
    follow_up: bool = False
    follow_up_cue_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_fired(self, at: datetime | None = None) -> None:
        now = _ensure_utc(at or _utc_now())
        self.status = "fired"
        self.fired_at = now

    def mark_done(self, at: datetime | None = None) -> None:
        now = _ensure_utc(at or _utc_now())
        self.status = "done"
        self.done_at = now

    def mark_canceled(self, at: datetime | None = None) -> None:
        now = _ensure_utc(at or _utc_now())
        self.status = "canceled"
        self.canceled_at = now

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        for key in ("remind_at", "created_at", "fired_at", "done_at", "canceled_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = _ensure_utc(value).isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reminder":
        payload = dict(data)
        for key in ("remind_at", "created_at", "fired_at", "done_at", "canceled_at"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                try:
                    payload[key] = _ensure_utc(datetime.fromisoformat(value))
                except ValueError:
                    payload[key] = None
        return cls(**payload)


class ReminderService:
    """Persistent reminder management."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        *,
        cue_store: CueStore | None = None,
    ) -> None:
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "reminders.json"
        self.storage_path = Path(storage_path)
        self._reminders: dict[str, Reminder] = {}
        self._cue_store = cue_store
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._reminders = {}
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load reminder store: %s", exc)
            self._reminders = {}
            return
        reminders = {}
        for item in raw.get("reminders", []):
            try:
                reminder = Reminder.from_dict(item)
                reminders[reminder.reminder_id] = reminder
            except Exception as exc:
                logger.debug("Skipping invalid reminder entry: %s", exc)
        self._reminders = reminders

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"reminders": [reminder.to_dict() for reminder in self._reminders.values()]}
        try:
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to save reminder store: %s", exc)

    def list_reminders(self, status: str | None = None) -> list[Reminder]:
        reminders = list(self._reminders.values())
        if status:
            reminders = [rem for rem in reminders if rem.status == status]
        return sorted(reminders, key=lambda reminder: reminder.remind_at)

    def add_reminder(
        self,
        title: str,
        remind_at: datetime,
        *,
        follow_up: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Reminder:
        reminder = Reminder(
            title=title,
            remind_at=_ensure_utc(remind_at),
            follow_up=follow_up,
            metadata=metadata or {},
        )
        if follow_up:
            cue_store = self._cue_store or get_cue_store()
            cue = cue_store.add_cue(
                prompt=f"How did \"{title}\" go?",
                source="reminder",
                source_id=reminder.reminder_id,
                due_at=reminder.remind_at,
            )
            reminder.follow_up_cue_id = cue.cue_id
        self._reminders[reminder.reminder_id] = reminder
        self._save()
        return reminder

    def mark_done(self, reminder_id: str, *, at: datetime | None = None) -> bool:
        reminder = self._reminders.get(reminder_id)
        if reminder is None:
            return False
        reminder.mark_done(at=at)
        self._save()
        return True

    def cancel(self, reminder_id: str, *, at: datetime | None = None) -> bool:
        reminder = self._reminders.get(reminder_id)
        if reminder is None:
            return False
        reminder.mark_canceled(at=at)
        self._save()
        return True

    def fire_due(self, *, now: datetime | None = None) -> list[Reminder]:
        current = _ensure_utc(now or _utc_now())
        fired: list[Reminder] = []
        for reminder in self._reminders.values():
            if reminder.status != "pending":
                continue
            if _ensure_utc(reminder.remind_at) <= current:
                reminder.mark_fired(at=current)
                fired.append(reminder)
        if fired:
            self._save()
        return fired

    def get(self, reminder_id: str) -> Reminder | None:
        return self._reminders.get(reminder_id)

    def all_reminders(self) -> Iterable[Reminder]:
        return list(self._reminders.values())


_reminder_service: ReminderService | None = None


def get_reminder_service() -> ReminderService:
    global _reminder_service
    if _reminder_service is None:
        _reminder_service = ReminderService()
    return _reminder_service


def set_reminder_service(service: ReminderService | None) -> None:
    global _reminder_service
    _reminder_service = service


__all__ = ["Reminder", "ReminderService", "get_reminder_service", "set_reminder_service"]
