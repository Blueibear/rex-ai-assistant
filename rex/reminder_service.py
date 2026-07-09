"""Reminder service for Rex AI Assistant.

Provides functionality to create, list, and manage one-off reminders.
Reminders can optionally create follow-up cues after they fire.

Reminders integrate with the scheduler system and notification system.

Ownership model (US-303 per-user isolation):
- Every reminder has exactly one validated owner ``user_id``.
- Every user-facing operation (get/list/complete/cancel/delete/update)
  requires the requesting ``user_id`` and only operates on reminders owned
  by that user. Non-owned reminders behave as if they do not exist.
- Missing, blank, or malformed user IDs fail closed (``ValueError``).
- Background firing (``fire_due_reminders`` with no ``user_id``) runs in
  system context: each due reminder executes as its stored owner (the
  notification and follow-up cue are attributed to ``reminder.user_id``).
- Legacy records persisted without a ``user_id`` load as owned by the
  distinct ``default`` profile. They are not shared and are not reassigned;
  only callers explicitly operating as ``default`` can access them. Manual
  reassignment may be added later via a separate administrative tool.
- Persisted records with an invalid ``user_id`` are quarantined: preserved
  in the storage file but excluded from all operations.

Usage:
    from rex.reminder_service import get_reminder_service

    service = get_reminder_service()
    reminder = service.create_reminder(
        user_id="default",
        title="Call mom",
        remind_at=datetime.now(timezone.utc) + timedelta(hours=2),
        followup_enabled=True,
    )
    pending = service.list_reminders(user_id="default", status="pending")
    service.mark_done(reminder.reminder_id, user_id="default")
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from rex.identity import validate_user_id

logger = logging.getLogger(__name__)

# Default storage directory (new)
_DATA_DIR = Path("data/reminders")

# Legacy storage directory (old branch used this)
_LEGACY_DATA_DIR = Path("data/followups")


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware and normalized to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def _legacy_storage_path() -> Path:
    """Return legacy storage path (if it exists)."""
    return _LEGACY_DATA_DIR / "reminders.json"


ReminderStatus = Literal["pending", "fired", "done", "canceled"]


class Reminder(BaseModel):
    """A one-off reminder that fires at a specific time."""

    reminder_id: str = Field(
        default_factory=lambda: f"rem_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this reminder",
    )
    user_id: str = Field(
        default="default",
        description=(
            "User/profile ID this reminder belongs to. Legacy records without "
            "one belong to the distinct 'default' profile."
        ),
    )
    title: str = Field(
        ...,
        description="Title/description of the reminder",
    )
    remind_at: datetime = Field(
        ...,
        description="When the reminder should fire (UTC)",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the reminder was created (UTC)",
    )
    status: ReminderStatus = Field(
        default="pending",
        description="Current status of the reminder",
    )
    fired_at: datetime | None = Field(
        default=None,
        description="When the reminder was fired (if applicable)",
    )
    done_at: datetime | None = Field(
        default=None,
        description="When the reminder was marked done (if applicable)",
    )
    canceled_at: datetime | None = Field(
        default=None,
        description="When the reminder was canceled (if applicable)",
    )
    followup_enabled: bool = Field(
        default=False,
        description="Whether to create a follow-up cue after firing",
    )
    followup_prompt: str | None = Field(
        default=None,
        description="Custom prompt for the follow-up cue",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @field_validator("user_id")
    @classmethod
    def _check_user_id(cls, value: str) -> str:
        """Reject blank, malformed, or traversal-style owner IDs (fail closed)."""
        return validate_user_id(value)

    def is_due(self, now: datetime | None = None) -> bool:
        """Check if this reminder is due to fire."""
        check_time = _ensure_utc(now or _utc_now())
        return self.status == "pending" and check_time >= _ensure_utc(self.remind_at)

    def get_followup_prompt(self) -> str:
        """Get the follow-up prompt, using default if not customized."""
        if self.followup_prompt:
            return self.followup_prompt
        return f"Did you get '{self.title}' done?"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reminder_id": "rem_abc123def456",
                    "user_id": "default",
                    "title": "Call mom",
                    "remind_at": "2024-01-15T14:00:00Z",
                    "created_at": "2024-01-15T10:30:00Z",
                    "status": "pending",
                    "followup_enabled": True,
                }
            ]
        }
    }


class ReminderService:
    """Service for managing reminders.

    - Create one-off reminders (owner recorded and validated)
    - List reminders (scoped to the requesting owner)
    - Mark reminders as done/canceled (ownership enforced)
    - Fire due reminders (notifications + optional follow-up cue, each
      executing as its stored owner)
    - Optionally register a scheduler job to auto-fire

    Backward compatible aliases:
    - add_reminder(...) -> create_reminder(...)
    - cancel(...) -> cancel_reminder(...)
    - fire_due(...) -> fire_due_reminders(...)
    - get(...) -> get_reminder(...)
    - all_reminders() -> iterable of all reminders (system/maintenance
      context only; must not back a user-facing surface)
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
    ) -> None:
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "reminders.json"

        self.storage_path = Path(storage_path)
        self._reminders: dict[str, Reminder] = {}
        # Raw persisted records that failed validation (e.g. invalid owner
        # user_id). Preserved verbatim on save, excluded from all operations.
        self._quarantined: list[dict[str, Any]] = []
        self._scheduler_job_registered = False

        self._load()

    def _load(self) -> None:
        """Load reminders from disk (supports legacy path)."""
        path_to_use = self.storage_path

        # If new path doesn't exist but legacy does, load legacy.
        if not path_to_use.exists():
            legacy = _legacy_storage_path()
            if legacy.exists():
                path_to_use = legacy

        if not path_to_use.exists():
            self._reminders = {}
            self._quarantined = []
            return

        try:
            with open(path_to_use, encoding="utf-8") as f:
                data = json.load(f)

            loaded: dict[str, Reminder] = {}
            quarantined: list[dict[str, Any]] = []
            for rem_data in data.get("reminders", []):
                try:
                    reminder = Reminder.model_validate(rem_data)
                    # Normalize datetimes
                    reminder.remind_at = _ensure_utc(reminder.remind_at)
                    reminder.created_at = _ensure_utc(reminder.created_at)
                    if reminder.fired_at is not None:
                        reminder.fired_at = _ensure_utc(reminder.fired_at)
                    if reminder.done_at is not None:
                        reminder.done_at = _ensure_utc(reminder.done_at)
                    if reminder.canceled_at is not None:
                        reminder.canceled_at = _ensure_utc(reminder.canceled_at)
                    loaded[reminder.reminder_id] = reminder
                except Exception as exc:
                    # Fail closed but preserve the record: invalid entries
                    # (including invalid owner IDs) are quarantined, not
                    # dropped and not attributed to any user.
                    if isinstance(rem_data, dict):
                        quarantined.append(rem_data)
                    logger.warning(
                        "Quarantined invalid reminder entry %r: %s",
                        rem_data.get("reminder_id") if isinstance(rem_data, dict) else rem_data,
                        exc,
                    )

            self._reminders = loaded
            self._quarantined = quarantined
            logger.debug(
                "Loaded %d reminders (%d quarantined) from %s",
                len(self._reminders),
                len(self._quarantined),
                path_to_use,
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load reminders: %s", e)
            self._reminders = {}
            self._quarantined = []

        # If we loaded from legacy, write back to new path so migration happens naturally.
        if path_to_use != self.storage_path and (self._reminders or self._quarantined):
            try:
                self._save()
            except Exception:
                # Non-fatal: still usable in-memory.
                pass

    def _save(self) -> None:
        """Save reminders to disk (including quarantined records, verbatim)."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            reminders_data = [r.model_dump(mode="json") for r in self._reminders.values()]
            reminders_data.extend(self._quarantined)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"reminders": reminders_data}, f, indent=2, default=str)
        except OSError as e:
            logger.error("Failed to save reminders: %s", e)

    def _owned(self, reminder_id: str, user_id: str) -> Reminder | None:
        """Return the reminder only if it exists and is owned by *user_id*.

        Validates *user_id* (fail closed) and treats non-owned reminders as
        nonexistent so callers cannot probe other users' reminder IDs.
        """
        user_id = validate_user_id(user_id)
        reminder = self._reminders.get(reminder_id)
        if reminder is None or reminder.user_id != user_id:
            return None
        return reminder

    def create_reminder(
        self,
        user_id: str,
        title: str,
        remind_at: datetime,
        *,
        followup_enabled: bool = False,
        followup_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        reminder_id: str | None = None,
    ) -> Reminder:
        """Create a new reminder owned by *user_id*.

        Raises:
            ValueError: If *user_id* is invalid or *reminder_id* already
                exists (a caller-supplied ID must never overwrite another
                reminder, which could belong to a different user).
        """
        user_id = validate_user_id(user_id)
        remind_at_utc = _ensure_utc(remind_at)

        if reminder_id and reminder_id in self._reminders:
            raise ValueError(f"Reminder ID already exists: {reminder_id!r}")

        reminder = Reminder(
            reminder_id=reminder_id or f"rem_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            title=title,
            remind_at=remind_at_utc,
            created_at=_utc_now(),
            status="pending",
            followup_enabled=followup_enabled,
            followup_prompt=followup_prompt,
            metadata=metadata or {},
        )

        self._reminders[reminder.reminder_id] = reminder
        self._save()
        logger.info("Created reminder %s for user %s: %s", reminder.reminder_id, user_id, title)

        # Try to register scheduler job if available
        self._ensure_scheduler_job()

        return reminder

    def get_reminder(self, reminder_id: str, user_id: str) -> Reminder | None:
        """Get a reminder by ID, only if owned by *user_id*."""
        return self._owned(reminder_id, user_id)

    def list_reminders(
        self,
        user_id: str,
        status: ReminderStatus | None = None,
        include_past: bool = True,
    ) -> list[Reminder]:
        """List reminders owned by *user_id*, with optional filtering."""
        user_id = validate_user_id(user_id)
        reminders = [r for r in self._reminders.values() if r.user_id == user_id]
        if status is not None:
            reminders = [r for r in reminders if r.status == status]
        if not include_past:
            reminders = [r for r in reminders if r.status == "pending"]

        reminders.sort(key=lambda r: r.remind_at)
        return reminders

    def update_reminder(
        self,
        reminder_id: str,
        user_id: str,
        *,
        title: str | None = None,
        remind_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Reminder | None:
        """Update fields on a reminder owned by *user_id*.

        Returns the updated reminder, or None if not found / not owned.
        """
        reminder = self._owned(reminder_id, user_id)
        if reminder is None:
            return None

        if title is not None:
            reminder.title = title
        if remind_at is not None:
            reminder.remind_at = _ensure_utc(remind_at)
        if metadata is not None:
            reminder.metadata = metadata
        self._save()
        logger.info("Updated reminder %s for user %s", reminder_id, user_id)
        return reminder

    def mark_done(self, reminder_id: str, user_id: str) -> bool:
        """Mark a reminder as done (ownership enforced)."""
        reminder = self._owned(reminder_id, user_id)
        if reminder is None:
            return False

        reminder.status = "done"
        reminder.done_at = _utc_now()
        self._save()
        logger.info("Marked reminder %s as done for user %s", reminder_id, user_id)
        return True

    def cancel_reminder(self, reminder_id: str, user_id: str) -> bool:
        """Cancel a reminder (ownership enforced)."""
        reminder = self._owned(reminder_id, user_id)
        if reminder is None:
            return False

        reminder.status = "canceled"
        reminder.canceled_at = _utc_now()
        self._save()
        logger.info("Canceled reminder %s for user %s", reminder_id, user_id)
        return True

    def delete_reminder(self, reminder_id: str, user_id: str) -> bool:
        """Delete a reminder (ownership enforced)."""
        if self._owned(reminder_id, user_id) is None:
            return False
        del self._reminders[reminder_id]
        self._save()
        logger.debug("Deleted reminder %s for user %s", reminder_id, user_id)
        return True

    def fire_due_reminders(
        self,
        now: datetime | None = None,
        *,
        user_id: str | None = None,
    ) -> list[Reminder]:
        """Fire due reminders (notifications + optional follow-up cues).

        With ``user_id=None`` this runs in system context (the background
        scheduler job): every user's due reminders fire, each executing as
        its stored owner. A user-initiated call must pass ``user_id`` and
        only fires that user's reminders.
        """
        if user_id is not None:
            user_id = validate_user_id(user_id)
        check_time = _ensure_utc(now or _utc_now())
        fired: list[Reminder] = []

        for reminder in list(self._reminders.values()):
            if user_id is not None and reminder.user_id != user_id:
                continue
            if reminder.is_due(check_time):
                self._fire_reminder(reminder, now=check_time)
                fired.append(reminder)

        return fired

    def _fire_reminder(self, reminder: Reminder, *, now: datetime | None = None) -> None:
        """Fire a single reminder as its stored owner."""
        fire_time = _ensure_utc(now or _utc_now())
        logger.info(
            "Firing reminder %s for user %s: %s",
            reminder.reminder_id,
            reminder.user_id,
            reminder.title,
        )

        self._send_notification(reminder)

        if reminder.followup_enabled:
            self._create_followup_cue(reminder, fired_at=fire_time)

        reminder.status = "fired"
        reminder.fired_at = fire_time
        self._save()

    def _send_notification(self, reminder: Reminder) -> None:
        """Send a notification for a reminder (best-effort)."""
        try:
            from rex.notification import NotificationRequest, get_notifier

            notifier = get_notifier()
            notification = NotificationRequest(
                priority="normal",
                title="Reminder",
                body=reminder.title,
                channel_preferences=["dashboard", "ha_tts"],
                metadata={"reminder_id": reminder.reminder_id, "user_id": reminder.user_id},
            )
            notifier.send(notification)
            logger.debug("Sent notification for reminder %s", reminder.reminder_id)
        except Exception as e:
            logger.warning("Failed to send notification for reminder: %s", e)

    def _create_followup_cue(self, reminder: Reminder, *, fired_at: datetime | None = None) -> None:
        """Create a follow-up cue for a reminder (best-effort)."""
        try:
            from rex.cue_store import get_cue_store

            store = get_cue_store()

            # Default expire hours from config, fallback to 168 hours (7 days)
            expire_hours = 168
            try:
                from rex.config_manager import load_config

                cfg = load_config()
                expire_hours = int(
                    cfg.get("conversation", {}).get("followups", {}).get("expire_hours", 168)
                )
            except Exception:
                pass

            fired_time = _ensure_utc(fired_at or _utc_now())
            expires_at = fired_time + timedelta(hours=max(1, expire_hours))

            store.add_cue(
                user_id=reminder.user_id,
                source_type="reminder",
                source_id=reminder.reminder_id,
                title=reminder.title,
                prompt=reminder.get_followup_prompt(),
                eligible_after=fired_time,
                expires_at=expires_at,
                metadata={"reminder_id": reminder.reminder_id},
            )
            logger.debug("Created follow-up cue for reminder %s", reminder.reminder_id)
        except Exception as e:
            logger.warning("Failed to create follow-up cue for reminder: %s", e)

    def _ensure_scheduler_job(self) -> None:
        """Ensure the scheduler job for firing reminders is registered (best-effort)."""
        if self._scheduler_job_registered:
            return

        try:
            from rex.scheduler import get_scheduler

            scheduler = get_scheduler()

            existing = scheduler.get_job("reminder_check")
            if existing is not None:
                self._scheduler_job_registered = True
                return

            def reminder_check_callback(job):
                # System context: fires every user's due reminders, each
                # executing as its stored owner.
                service = get_reminder_service()
                fired = service.fire_due_reminders()
                if fired:
                    logger.info("Fired %d due reminder(s)", len(fired))

            scheduler.register_callback("reminder_check", reminder_check_callback)

            scheduler.add_job(
                job_id="reminder_check",
                name="Check Due Reminders",
                schedule="interval:60",
                callback_name="reminder_check",
                metadata={"service": "reminder_service"},
            )

            self._scheduler_job_registered = True
            logger.info("Registered reminder check scheduler job")
        except Exception as e:
            logger.warning("Failed to register scheduler job for reminders: %s", e)

    def __len__(self) -> int:
        return len(self._reminders)

    def stats(self) -> dict[str, Any]:
        by_status = {"pending": 0, "fired": 0, "done": 0, "canceled": 0}
        followup_count = 0

        for reminder in self._reminders.values():
            by_status[reminder.status] = by_status.get(reminder.status, 0) + 1
            if reminder.followup_enabled:
                followup_count += 1

        return {
            "total": len(self._reminders),
            "by_status": by_status,
            "with_followup": followup_count,
            "quarantined": len(self._quarantined),
        }

    # Backward compatible aliases (codex branch API)

    def add_reminder(
        self,
        title: str,
        remind_at: datetime,
        *,
        follow_up: bool = False,
        metadata: dict[str, Any] | None = None,
        user_id: str,
    ) -> Reminder:
        """Alias for older code: add_reminder(title, remind_at, follow_up=...).

        ``user_id`` is required: reminders must record their real owner and
        never silently fall back to the ``default`` profile.
        """
        return self.create_reminder(
            user_id=user_id,
            title=title,
            remind_at=remind_at,
            followup_enabled=follow_up,
            metadata=metadata,
        )

    def cancel(self, reminder_id: str, user_id: str, *, at: datetime | None = None) -> bool:
        """Alias for older code: cancel(reminder_id) (ownership enforced)."""
        _ = at  # kept for signature compatibility
        return self.cancel_reminder(reminder_id, user_id)

    def fire_due(self, *, now: datetime | None = None) -> list[Reminder]:
        """Alias for older code: fire_due(now=...) (system context)."""
        return self.fire_due_reminders(now=now)

    def get(self, reminder_id: str, user_id: str) -> Reminder | None:
        """Alias for older code: get(reminder_id) (ownership enforced)."""
        return self.get_reminder(reminder_id, user_id)

    def all_reminders(self) -> Iterable[Reminder]:
        """Iterable of all reminders, across every owner.

        System/maintenance context only (e.g. diagnostics and tests). Must
        never back a user-facing list surface — those go through
        :meth:`list_reminders` with the requesting user's ID.
        """
        return list(self._reminders.values())


# Global instance
_reminder_service: ReminderService | None = None


def get_reminder_service() -> ReminderService:
    """Get the global reminder service instance."""
    global _reminder_service
    if _reminder_service is None:
        _reminder_service = ReminderService()
    return _reminder_service


def set_reminder_service(service: ReminderService | None) -> None:
    """Set the global reminder service instance (for testing)."""
    global _reminder_service
    _reminder_service = service


__all__ = [
    "Reminder",
    "ReminderService",
    "ReminderStatus",
    "get_reminder_service",
    "set_reminder_service",
]
