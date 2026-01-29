"""Reminder service for Rex AI Assistant.

Provides functionality to create, list, and manage one-off reminders.
Reminders can optionally create follow-up cues after they fire.

Reminders integrate with the scheduler system and notification system.

Usage:
    from rex.reminder_service import get_reminder_service, Reminder

    service = get_reminder_service()
    reminder = service.create_reminder(
        user_id="default",
        title="Call mom",
        remind_at=datetime.now() + timedelta(hours=2),
        followup_enabled=True,
    )
    pending = service.list_reminders(user_id="default")
    service.mark_done(reminder.reminder_id)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default storage directory
_DATA_DIR = Path("data/reminders")


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


ReminderStatus = Literal["pending", "fired", "done", "canceled"]


class Reminder(BaseModel):
    """A one-off reminder that fires at a specific time.

    Attributes:
        reminder_id: Unique identifier for this reminder.
        user_id: User/profile ID this reminder belongs to.
        title: Title/description of the reminder.
        remind_at: When the reminder should fire (UTC).
        created_at: When the reminder was created (UTC).
        status: Current status of the reminder.
        fired_at: When the reminder was fired (if applicable).
        done_at: When the reminder was marked done (if applicable).
        followup_enabled: Whether to create a follow-up cue after firing.
        followup_prompt: Custom prompt for the follow-up cue.
        metadata: Additional metadata.
    """

    reminder_id: str = Field(
        default_factory=lambda: f"rem_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this reminder",
    )
    user_id: str = Field(
        default="default",
        description="User/profile ID this reminder belongs to",
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
    fired_at: Optional[datetime] = Field(
        default=None,
        description="When the reminder was fired (if applicable)",
    )
    done_at: Optional[datetime] = Field(
        default=None,
        description="When the reminder was marked done (if applicable)",
    )
    followup_enabled: bool = Field(
        default=False,
        description="Whether to create a follow-up cue after firing",
    )
    followup_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for the follow-up cue",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def is_due(self, now: Optional[datetime] = None) -> bool:
        """Check if this reminder is due to fire."""
        check_time = now or _utc_now()
        return self.status == "pending" and check_time >= self.remind_at

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

    Provides functionality to:
    - Create one-off reminders
    - List reminders for a user
    - Mark reminders as done
    - Fire due reminders (with notification and optional follow-up cue)
    - Integrate with scheduler for automatic firing

    Example:
        service = ReminderService()
        reminder = service.create_reminder(
            user_id="default",
            title="Call mom",
            remind_at=datetime.now() + timedelta(hours=2),
        )
        service.fire_due_reminders()  # Called by scheduler
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
    ) -> None:
        """Initialize the reminder service.

        Args:
            storage_path: Path to the persistence file. Defaults to
                data/reminders/reminders.json
        """
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "reminders.json"

        self.storage_path = Path(storage_path)
        self._reminders: dict[str, Reminder] = {}
        self._scheduler_job_registered = False
        self._load()

    def _load(self) -> None:
        """Load reminders from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for rem_data in data.get("reminders", []):
                        reminder = Reminder.model_validate(rem_data)
                        self._reminders[reminder.reminder_id] = reminder
                    logger.debug(f"Loaded {len(self._reminders)} reminders from disk")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load reminders: {e}")
                self._reminders = {}

    def _save(self) -> None:
        """Save reminders to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            reminders_data = [
                rem.model_dump(mode="json") for rem in self._reminders.values()
            ]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"reminders": reminders_data}, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save reminders: {e}")

    def create_reminder(
        self,
        user_id: str,
        title: str,
        remind_at: datetime,
        *,
        followup_enabled: bool = False,
        followup_prompt: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        reminder_id: Optional[str] = None,
    ) -> Reminder:
        """Create a new reminder.

        Args:
            user_id: User/profile ID.
            title: Title/description of the reminder.
            remind_at: When the reminder should fire.
            followup_enabled: Whether to create a follow-up cue after firing.
            followup_prompt: Custom prompt for the follow-up cue.
            metadata: Additional metadata.
            reminder_id: Optional custom reminder ID.

        Returns:
            The created Reminder.
        """
        # Ensure remind_at is timezone-aware UTC
        if remind_at.tzinfo is None:
            remind_at = remind_at.replace(tzinfo=timezone.utc)
        else:
            remind_at = remind_at.astimezone(timezone.utc)

        reminder = Reminder(
            reminder_id=reminder_id or f"rem_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            title=title,
            remind_at=remind_at,
            created_at=_utc_now(),
            status="pending",
            followup_enabled=followup_enabled,
            followup_prompt=followup_prompt,
            metadata=metadata or {},
        )

        self._reminders[reminder.reminder_id] = reminder
        self._save()
        logger.info(f"Created reminder {reminder.reminder_id}: {title}")

        # Try to register scheduler job if not already done
        self._ensure_scheduler_job()

        return reminder

    def get_reminder(self, reminder_id: str) -> Optional[Reminder]:
        """Get a specific reminder by ID.

        Args:
            reminder_id: The reminder ID to look up.

        Returns:
            The Reminder if found, else None.
        """
        return self._reminders.get(reminder_id)

    def list_reminders(
        self,
        user_id: Optional[str] = None,
        status: Optional[ReminderStatus] = None,
        include_past: bool = True,
    ) -> list[Reminder]:
        """List reminders with optional filtering.

        Args:
            user_id: Filter by user ID.
            status: Filter by status.
            include_past: Whether to include reminders that have fired.

        Returns:
            List of reminders matching the filters, sorted by remind_at.
        """
        reminders = list(self._reminders.values())
        if user_id is not None:
            reminders = [r for r in reminders if r.user_id == user_id]
        if status is not None:
            reminders = [r for r in reminders if r.status == status]
        if not include_past:
            reminders = [r for r in reminders if r.status == "pending"]

        reminders.sort(key=lambda r: r.remind_at)
        return reminders

    def mark_done(self, reminder_id: str) -> bool:
        """Mark a reminder as done.

        Args:
            reminder_id: The reminder ID to mark.

        Returns:
            True if the reminder was found and marked, False otherwise.
        """
        reminder = self._reminders.get(reminder_id)
        if reminder is None:
            return False

        reminder.status = "done"
        reminder.done_at = _utc_now()
        self._save()
        logger.info(f"Marked reminder {reminder_id} as done")
        return True

    def cancel_reminder(self, reminder_id: str) -> bool:
        """Cancel a reminder.

        Args:
            reminder_id: The reminder ID to cancel.

        Returns:
            True if the reminder was found and canceled, False otherwise.
        """
        reminder = self._reminders.get(reminder_id)
        if reminder is None:
            return False

        reminder.status = "canceled"
        self._save()
        logger.info(f"Canceled reminder {reminder_id}")
        return True

    def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder from the store.

        Args:
            reminder_id: The reminder ID to delete.

        Returns:
            True if the reminder was found and deleted, False otherwise.
        """
        if reminder_id in self._reminders:
            del self._reminders[reminder_id]
            self._save()
            logger.debug(f"Deleted reminder {reminder_id}")
            return True
        return False

    def fire_due_reminders(self, now: Optional[datetime] = None) -> list[Reminder]:
        """Fire all due reminders.

        This method:
        1. Finds all pending reminders that are due
        2. Sends notifications for each
        3. Creates follow-up cues if enabled
        4. Marks them as fired

        Args:
            now: Current time (defaults to UTC now).

        Returns:
            List of reminders that were fired.
        """
        check_time = now or _utc_now()
        fired = []

        for reminder in list(self._reminders.values()):
            if reminder.is_due(check_time):
                self._fire_reminder(reminder)
                fired.append(reminder)

        return fired

    def _fire_reminder(self, reminder: Reminder) -> None:
        """Fire a single reminder.

        Args:
            reminder: The reminder to fire.
        """
        logger.info(f"Firing reminder {reminder.reminder_id}: {reminder.title}")

        # Send notification
        self._send_notification(reminder)

        # Create follow-up cue if enabled
        if reminder.followup_enabled:
            self._create_followup_cue(reminder)

        # Update status
        reminder.status = "fired"
        reminder.fired_at = _utc_now()
        self._save()

    def _send_notification(self, reminder: Reminder) -> None:
        """Send a notification for a reminder.

        Args:
            reminder: The reminder to notify about.
        """
        try:
            from rex.notification import NotificationRequest, get_notifier

            notifier = get_notifier()
            notification = NotificationRequest(
                priority="normal",
                title="Reminder",
                body=reminder.title,
                channel_preferences=["dashboard", "ha_tts"],
                metadata={
                    "reminder_id": reminder.reminder_id,
                    "user_id": reminder.user_id,
                },
            )
            notifier.send(notification)
            logger.debug(f"Sent notification for reminder {reminder.reminder_id}")
        except Exception as e:
            logger.warning(f"Failed to send notification for reminder: {e}")

    def _create_followup_cue(self, reminder: Reminder) -> None:
        """Create a follow-up cue for a reminder.

        Args:
            reminder: The reminder to create a cue for.
        """
        try:
            from rex.cue_store import get_cue_store

            store = get_cue_store()

            # Get default expire hours from config, or use 168 (7 days)
            expire_hours = 168
            try:
                from rex.config_manager import load_config

                config = load_config()
                expire_hours = config.get("conversation", {}).get(
                    "followups", {}
                ).get("expire_hours", 168)
            except Exception:
                pass

            store.add_cue(
                user_id=reminder.user_id,
                source_type="reminder",
                source_id=reminder.reminder_id,
                title=reminder.title,
                prompt=reminder.get_followup_prompt(),
                expires_in=timedelta(hours=expire_hours),
                metadata={"reminder_id": reminder.reminder_id},
            )
            logger.debug(f"Created follow-up cue for reminder {reminder.reminder_id}")
        except Exception as e:
            logger.warning(f"Failed to create follow-up cue for reminder: {e}")

    def _ensure_scheduler_job(self) -> None:
        """Ensure the scheduler job for firing reminders is registered."""
        if self._scheduler_job_registered:
            return

        try:
            from rex.scheduler import get_scheduler

            scheduler = get_scheduler()

            # Check if job already exists
            existing = scheduler.get_job("reminder_check")
            if existing is not None:
                self._scheduler_job_registered = True
                return

            # Register callback
            def reminder_check_callback(job):
                service = get_reminder_service()
                fired = service.fire_due_reminders()
                if fired:
                    logger.info(f"Fired {len(fired)} due reminder(s)")

            scheduler.register_callback("reminder_check", reminder_check_callback)

            # Add job - check every 60 seconds
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
            logger.warning(f"Failed to register scheduler job for reminders: {e}")

    def __len__(self) -> int:
        """Return the number of reminders."""
        return len(self._reminders)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the reminder service."""
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
        }


# Global instance
_reminder_service: Optional[ReminderService] = None


def get_reminder_service() -> ReminderService:
    """Get the global reminder service instance."""
    global _reminder_service
    if _reminder_service is None:
        _reminder_service = ReminderService()
    return _reminder_service


def set_reminder_service(service: ReminderService) -> None:
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
