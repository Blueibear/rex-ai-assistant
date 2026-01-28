"""Multi-channel notification system with priority routing and escalation.

This module provides:
- Notification model for representing alerts and messages
- Notifier class for dispatching notifications to multiple channels
- EscalationManager for handling quiet hours and escalation rules
- Integration with event bus for automatic notification triggers

The system supports multiple channels (dashboard, email, SMS, Home Assistant TTS)
with priority-based routing and digest mode for batching low-priority alerts.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from rex.contracts.core import NotificationChannel, NotificationPriority

logger = logging.getLogger(__name__)

# Global instances
_notifier: Optional["Notifier"] = None
_escalation_manager: Optional["EscalationManager"] = None


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


# --- Models ---


class NotificationRequest(BaseModel):
    """A notification request to be delivered to users.

    This differs from the Notification model in rex.contracts.core in that
    it supports multiple channel preferences and is specific to the
    notification service.
    """

    id: str = Field(
        default_factory=lambda: f"notif_{uuid.uuid4().hex[:16]}",
        description="Unique notification identifier",
    )
    priority: Literal["urgent", "normal", "digest"] = Field(
        default="normal",
        description="Priority level affecting delivery timing and channels",
    )
    title: str = Field(
        ...,
        description="Notification title/subject",
    )
    body: str = Field(
        ...,
        description="Notification body content",
    )
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="When the notification was created (UTC)",
    )
    channel_preferences: list[str] = Field(
        default_factory=lambda: ["dashboard"],
        description="Ordered list of preferred delivery channels",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata for the notification",
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="When the notification was acknowledged (for escalation)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "notif_abc123",
                    "priority": "urgent",
                    "title": "High Priority Email",
                    "body": "You have a high importance email from your manager.",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "channel_preferences": ["sms", "email", "dashboard"],
                }
            ]
        }
    }


@dataclass
class DigestQueue:
    """Queue for batching digest notifications."""

    channel: str
    notifications: list[NotificationRequest] = field(default_factory=list)
    last_flush_at: Optional[datetime] = None


# --- Notifier ---


class Notifier:
    """Multi-channel notification dispatcher.

    The Notifier routes notifications to appropriate channels based on
    priority and channel preferences. It supports:
    - Urgent: Send immediately to all preferred channels
    - Normal: Send to first available channel
    - Digest: Queue and send summary after configured interval
    """

    def __init__(
        self,
        digest_interval_seconds: int = 3600,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the notifier.

        Args:
            digest_interval_seconds: How often to flush digest queues
            storage_path: Path to store digest queue data
        """
        self.digest_interval = digest_interval_seconds
        self.storage_path = storage_path or Path("data/notifications")
        self.digest_file = self.storage_path / "digests.json"
        self.digest_queues: dict[str, DigestQueue] = {}
        self._load_digests()

    def _load_digests(self) -> None:
        """Load digest queues from disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if self.digest_file.exists():
            try:
                with open(self.digest_file) as f:
                    data = json.load(f)
                for channel, queue_data in data.items():
                    notifications = [
                        NotificationRequest.model_validate(n)
                        for n in queue_data.get("notifications", [])
                    ]
                    last_flush_str = queue_data.get("last_flush_at")
                    last_flush = (
                        datetime.fromisoformat(last_flush_str)
                        if last_flush_str
                        else None
                    )
                    self.digest_queues[channel] = DigestQueue(
                        channel=channel,
                        notifications=notifications,
                        last_flush_at=last_flush,
                    )
                logger.info(f"Loaded {len(self.digest_queues)} digest queues from disk")
            except Exception as e:
                logger.warning(f"Failed to load digest queues: {e}")
                self.digest_queues = {}

    def _save_digests(self) -> None:
        """Save digest queues to disk."""
        try:
            data = {}
            for channel, queue in self.digest_queues.items():
                data[channel] = {
                    "notifications": [
                        n.model_dump(mode="json") for n in queue.notifications
                    ],
                    "last_flush_at": queue.last_flush_at.isoformat()
                    if queue.last_flush_at
                    else None,
                }
            with open(self.digest_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save digest queues: {e}")

    def send(self, notification: NotificationRequest) -> None:
        """Send a notification based on its priority.

        - Urgent: Send immediately to all preferred channels
        - Normal: Send to first available channel
        - Digest: Add to queue for later batch sending

        Args:
            notification: The notification to send
        """
        logger.info(
            f"Sending {notification.priority} notification: {notification.title}"
        )

        if notification.priority == "urgent":
            self._send_urgent(notification)
        elif notification.priority == "normal":
            self._send_normal(notification)
        elif notification.priority == "digest":
            self._queue_digest(notification)
        else:
            logger.warning(f"Unknown priority: {notification.priority}")

    def _send_urgent(self, notification: NotificationRequest) -> None:
        """Send urgent notification to all preferred channels immediately."""
        for channel in notification.channel_preferences:
            try:
                self._dispatch_to_channel(channel, notification)
            except Exception as e:
                logger.error(f"Failed to send urgent notification to {channel}: {e}")

    def _send_normal(self, notification: NotificationRequest) -> None:
        """Send normal notification to first available channel."""
        for channel in notification.channel_preferences:
            try:
                self._dispatch_to_channel(channel, notification)
                logger.debug(
                    f"Sent normal notification to {channel}, "
                    f"skipping remaining channels"
                )
                break
            except Exception as e:
                logger.warning(
                    f"Failed to send to {channel}, trying next channel: {e}"
                )
                continue

    def _queue_digest(self, notification: NotificationRequest) -> None:
        """Add notification to digest queue."""
        for channel in notification.channel_preferences:
            if channel not in self.digest_queues:
                self.digest_queues[channel] = DigestQueue(channel=channel)
            self.digest_queues[channel].notifications.append(notification)
            logger.debug(f"Queued digest notification for {channel}")

        self._save_digests()

    def _dispatch_to_channel(
        self, channel: str, notification: NotificationRequest
    ) -> None:
        """Dispatch notification to a specific channel.

        Args:
            channel: The channel to send to
            notification: The notification to send

        Raises:
            Exception: If dispatch fails
        """
        if channel == "dashboard":
            self._send_to_dashboard(notification)
        elif channel == "email":
            self._send_to_email(notification)
        elif channel == "sms":
            self._send_to_sms(notification)
        elif channel == "ha_tts":
            self._send_to_ha_tts(notification)
        else:
            logger.warning(f"Unknown channel: {channel}")

    def _send_to_dashboard(self, notification: NotificationRequest) -> None:
        """Send notification to dashboard (placeholder)."""
        logger.info(f"[DASHBOARD] {notification.title}: {notification.body}")
        # In a real implementation, this would write to a dashboard API or database

    def _send_to_email(self, notification: NotificationRequest) -> None:
        """Send notification via email."""
        try:
            from rex.email_service import get_email_service

            email_service = get_email_service()
            logger.info(
                f"[EMAIL] Would send: {notification.title}\n{notification.body}"
            )
            # In a real implementation:
            # email_service.send(
            #     to=user_email,
            #     subject=notification.title,
            #     body=notification.body
            # )
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")
            raise

    def _send_to_sms(self, notification: NotificationRequest) -> None:
        """Send notification via SMS."""
        try:
            from rex.messaging_service import Message, get_sms_service

            sms_service = get_sms_service()

            # Get recipient from metadata or use default
            to_number = notification.metadata.get("to_number", "+15555551234")

            message = Message(
                channel="sms",
                to=to_number,
                from_=sms_service.from_number,
                body=f"{notification.title}: {notification.body}",
            )

            sms_service.send(message)
            logger.info(f"[SMS] Sent to {to_number}: {notification.title}")
        except Exception as e:
            logger.warning(f"SMS notification failed: {e}")
            raise

    def _send_to_ha_tts(self, notification: NotificationRequest) -> None:
        """Send notification to Home Assistant TTS (stub)."""
        logger.info(f"[HA_TTS] Would announce: {notification.title}")
        # In a real implementation, this would call Home Assistant TTS API

    def flush_digests(self, channel: Optional[str] = None) -> int:
        """Flush digest queues and send summaries.

        Args:
            channel: Specific channel to flush, or None for all channels

        Returns:
            Number of digest summaries sent
        """
        channels_to_flush = (
            [channel] if channel else list(self.digest_queues.keys())
        )
        count = 0

        for ch in channels_to_flush:
            if ch not in self.digest_queues:
                continue

            queue = self.digest_queues[ch]
            if not queue.notifications:
                continue

            # Create digest summary
            summary = self._create_digest_summary(queue.notifications)
            digest_notification = NotificationRequest(
                priority="normal",
                title=f"Digest Summary ({len(queue.notifications)} items)",
                body=summary,
                channel_preferences=[ch],
            )

            # Send the digest
            self._dispatch_to_channel(ch, digest_notification)

            # Clear queue and update flush time
            queue.notifications.clear()
            queue.last_flush_at = _utc_now()
            count += 1

        self._save_digests()
        logger.info(f"Flushed {count} digest queue(s)")
        return count

    def _create_digest_summary(
        self, notifications: list[NotificationRequest]
    ) -> str:
        """Create a summary of digest notifications."""
        lines = ["Notification Digest:\n"]
        for i, notif in enumerate(notifications, 1):
            lines.append(f"{i}. {notif.title}")
            if notif.body:
                # Truncate long bodies
                body_preview = (
                    notif.body[:80] + "..."
                    if len(notif.body) > 80
                    else notif.body
                )
                lines.append(f"   {body_preview}")
        return "\n".join(lines)

    def list_digests(self) -> dict[str, list[dict]]:
        """List all queued digest notifications.

        Returns:
            Dictionary mapping channel names to lists of notification dicts
        """
        result = {}
        for channel, queue in self.digest_queues.items():
            result[channel] = [
                {
                    "id": n.id,
                    "title": n.title,
                    "body": n.body,
                    "timestamp": n.timestamp.isoformat(),
                }
                for n in queue.notifications
            ]
        return result

    def setup_event_subscriptions(self) -> None:
        """Subscribe to event bus events to create notifications automatically."""
        try:
            from rex.event_bus import get_event_bus

            bus = get_event_bus()

            # Subscribe to email events
            bus.subscribe("email.unread", self._on_email_unread)

            # Subscribe to calendar events
            bus.subscribe("calendar.update", self._on_calendar_update)

            logger.info("Notification system subscribed to event bus events")
        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    def _on_email_unread(self, event) -> None:
        """Handle email.unread events."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            emails = payload.get("emails", [])

            # Check for high-importance emails
            for email in emails:
                importance = email.get("importance_score", 0.5)
                if importance >= 0.8:
                    notification = NotificationRequest(
                        priority="urgent",
                        title="High Importance Email",
                        body=f"From: {email.get('from_addr', 'Unknown')}\n"
                        f"Subject: {email.get('subject', 'No subject')}",
                        channel_preferences=["sms", "email", "dashboard"],
                        metadata={"email_id": email.get("id")},
                    )
                    self.send(notification)
        except Exception as e:
            logger.error(f"Error handling email.unread event: {e}")

    def _on_calendar_update(self, event) -> None:
        """Handle calendar.update events."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            events = payload.get("events", [])

            # Notify about upcoming events (within 15 minutes)
            now = _utc_now()
            for cal_event in events:
                start_time_str = cal_event.get("start_time")
                if not start_time_str:
                    continue

                start_time = datetime.fromisoformat(start_time_str)
                time_until = (start_time - now).total_seconds()

                if 0 < time_until <= 900:  # Within 15 minutes
                    notification = NotificationRequest(
                        priority="normal",
                        title="Upcoming Calendar Event",
                        body=f"{cal_event.get('title', 'Event')} "
                        f"starts at {start_time.strftime('%I:%M %p')}",
                        channel_preferences=["dashboard", "ha_tts"],
                        metadata={"event_id": cal_event.get("event_id")},
                    )
                    self.send(notification)
        except Exception as e:
            logger.error(f"Error handling calendar.update event: {e}")


# --- Escalation Manager ---


@dataclass
class EscalationRule:
    """Rule for escalating unacknowledged notifications."""

    notification_id: str
    sent_at: datetime
    escalation_delay_minutes: int
    next_channel: str
    escalated: bool = False


class EscalationManager:
    """Manages notification escalation and quiet hours.

    The EscalationManager:
    - Defines quiet hours when non-urgent notifications are suppressed
    - Tracks unacknowledged urgent notifications
    - Escalates to alternative channels if not acknowledged within threshold
    """

    def __init__(
        self,
        quiet_hours_start: time = time(22, 0),
        quiet_hours_end: time = time(7, 0),
        escalation_delay_minutes: int = 5,
    ):
        """Initialize the escalation manager.

        Args:
            quiet_hours_start: Start of quiet hours (default 22:00)
            quiet_hours_end: End of quiet hours (default 07:00)
            escalation_delay_minutes: Minutes to wait before escalating
        """
        self.quiet_hours_start = quiet_hours_start
        self.quiet_hours_end = quiet_hours_end
        self.escalation_delay = escalation_delay_minutes
        self.dnd_enabled = False
        self.pending_escalations: dict[str, EscalationRule] = {}

    def is_quiet_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if the given time is within quiet hours.

        Args:
            dt: Datetime to check (defaults to now)

        Returns:
            True if within quiet hours
        """
        if dt is None:
            dt = _utc_now()

        current_time = dt.time()

        # Handle quiet hours that span midnight
        if self.quiet_hours_start < self.quiet_hours_end:
            return self.quiet_hours_start <= current_time < self.quiet_hours_end
        else:
            return current_time >= self.quiet_hours_start or current_time < self.quiet_hours_end

    def should_suppress(self, notification: NotificationRequest) -> bool:
        """Check if a notification should be suppressed.

        Args:
            notification: The notification to check

        Returns:
            True if the notification should be suppressed
        """
        # Urgent notifications always go through
        if notification.priority == "urgent":
            return False

        # DND mode blocks all non-urgent
        if self.dnd_enabled:
            return True

        # Quiet hours blocks normal and digest notifications
        if self.is_quiet_hours():
            return True

        return False

    def track_notification(
        self,
        notification: NotificationRequest,
        next_channel: str = "email",
    ) -> None:
        """Track a notification for potential escalation.

        Args:
            notification: The notification to track
            next_channel: Channel to escalate to if not acknowledged
        """
        if notification.priority != "urgent":
            return

        rule = EscalationRule(
            notification_id=notification.id,
            sent_at=notification.timestamp,
            escalation_delay_minutes=self.escalation_delay,
            next_channel=next_channel,
        )
        self.pending_escalations[notification.id] = rule
        logger.debug(f"Tracking notification {notification.id} for escalation")

    def acknowledge(self, notification_id: str) -> bool:
        """Mark a notification as acknowledged.

        Args:
            notification_id: The notification to acknowledge

        Returns:
            True if the notification was found and acknowledged
        """
        if notification_id in self.pending_escalations:
            del self.pending_escalations[notification_id]
            logger.info(f"Acknowledged notification {notification_id}")
            return True
        return False

    def check_escalations(self) -> list[tuple[str, str]]:
        """Check for notifications that need escalation.

        Returns:
            List of (notification_id, next_channel) tuples to escalate
        """
        now = _utc_now()
        to_escalate = []

        for notif_id, rule in list(self.pending_escalations.items()):
            if rule.escalated:
                continue

            elapsed = (now - rule.sent_at).total_seconds() / 60
            if elapsed >= rule.escalation_delay_minutes:
                to_escalate.append((notif_id, rule.next_channel))
                rule.escalated = True
                logger.info(
                    f"Escalating notification {notif_id} to {rule.next_channel}"
                )

        return to_escalate

    def set_dnd(self, enabled: bool) -> None:
        """Enable or disable do-not-disturb mode.

        Args:
            enabled: True to enable DND, False to disable
        """
        self.dnd_enabled = enabled
        logger.info(f"Do-not-disturb mode: {'enabled' if enabled else 'disabled'}")


# --- Global Service Accessors ---


def get_notifier() -> Notifier:
    """Get the global notifier instance.

    Returns:
        The global notifier, creating it if needed
    """
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier


def set_notifier(notifier: Notifier) -> None:
    """Set the global notifier instance.

    Args:
        notifier: The notifier to use
    """
    global _notifier
    _notifier = notifier


def get_escalation_manager() -> EscalationManager:
    """Get the global escalation manager instance.

    Returns:
        The global escalation manager, creating it if needed
    """
    global _escalation_manager
    if _escalation_manager is None:
        _escalation_manager = EscalationManager()
    return _escalation_manager


def set_escalation_manager(manager: EscalationManager) -> None:
    """Set the global escalation manager instance.

    Args:
        manager: The escalation manager to use
    """
    global _escalation_manager
    _escalation_manager = manager
