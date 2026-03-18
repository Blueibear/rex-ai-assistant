"""Auto-escalation job for unacknowledged high-priority notifications (US-092).

Tracks high-priority notifications and escalates them after a configurable
timeout when they remain unacknowledged.  Escalation stops after a
configurable maximum attempt count.

In beta, escalation events are logged without making real deliveries (stub
behaviour).

Usage example::

    from rex.escalation_job import EscalationJob, EscalationConfig
    from rex.priority_notification_router import RoutableNotification
    from rex.notification_priority import NotificationPriority

    config = EscalationConfig(
        timeout_minutes_by_priority={NotificationPriority.HIGH: 15},
        max_attempts=3,
    )
    job = EscalationJob(config=config)

    notif = RoutableNotification(
        id="n1",
        priority=NotificationPriority.HIGH,
        title="Disk almost full",
        body="90% used",
    )
    job.track(notif)

    # Later, after enough time passes:
    result = job.run()
    print(result.escalated)       # list of EscalationAttempt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from rex.notification_priority import NotificationPriority
from rex.priority_notification_router import RoutableNotification

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MINUTES: int = 15
_DEFAULT_MAX_ATTEMPTS: int = 3

# Priorities that are subject to escalation by default.
_DEFAULT_ESCALATION_PRIORITIES: frozenset[NotificationPriority] = frozenset(
    {NotificationPriority.HIGH}
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EscalationConfig:
    """Runtime configuration for :class:`EscalationJob`.

    Attributes:
        timeout_minutes_by_priority: Maps a priority level to the number of
            minutes after creation (or last escalation) before the first/next
            escalation attempt fires.  Defaults to ``{HIGH: 15}``.
        max_attempts: Maximum number of escalation attempts per notification
            before escalation is silently stopped.  Defaults to ``3``.
    """

    timeout_minutes_by_priority: dict[NotificationPriority, int] = field(
        default_factory=lambda: {NotificationPriority.HIGH: _DEFAULT_TIMEOUT_MINUTES}
    )
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationConfig:
        """Build config from a plain dict (e.g. loaded from JSON).

        Accepted keys:

        - ``"timeout_minutes_by_priority"``: dict mapping priority strings to
          integer minute values, e.g. ``{"high": 15}``.
        - ``"max_attempts"``: integer maximum escalation attempts.

        Args:
            data: Plain config dict.

        Returns:
            An :class:`EscalationConfig` instance.
        """
        raw_timeouts: dict[str, Any] = data.get("timeout_minutes_by_priority", {})
        timeout_map: dict[NotificationPriority, int] = {}
        if raw_timeouts:
            for k, v in raw_timeouts.items():
                priority = NotificationPriority.from_str(str(k))
                timeout_map[priority] = int(v)
        else:
            timeout_map = {NotificationPriority.HIGH: _DEFAULT_TIMEOUT_MINUTES}

        max_attempts = int(data.get("max_attempts", _DEFAULT_MAX_ATTEMPTS))
        return cls(
            timeout_minutes_by_priority=timeout_map,
            max_attempts=max_attempts,
        )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrackedNotification:
    """Internal record tracking escalation state for one notification.

    Attributes:
        notification_id: Unique ID of the notification.
        priority: Priority level.
        title: Notification title (for log messages).
        body: Notification body.
        created_at: UTC timestamp when the notification was first tracked.
        attempt_count: Number of escalation attempts made so far.
        acknowledged: ``True`` once :meth:`EscalationJob.acknowledge` is called.
        last_escalated_at: UTC timestamp of the most recent escalation attempt,
            or ``None`` if not yet escalated.
    """

    notification_id: str
    priority: NotificationPriority
    title: str
    body: str
    created_at: datetime
    attempt_count: int = 0
    acknowledged: bool = False
    last_escalated_at: datetime | None = None


@dataclass
class EscalationAttempt:
    """A single escalation event that was logged or delivered.

    Attributes:
        notification_id: ID of the escalated notification.
        attempt_number: 1-based attempt counter for this notification.
        escalated_at: UTC timestamp when the escalation was attempted.
        priority: Priority level of the notification.
        title: Title of the escalated notification.
    """

    notification_id: str
    attempt_number: int
    escalated_at: datetime
    priority: NotificationPriority
    title: str


@dataclass
class EscalationResult:
    """Outcome of a single :meth:`EscalationJob.run` call.

    Attributes:
        ran_at: UTC timestamp when the job ran.
        escalated: List of :class:`EscalationAttempt` records produced this run.
        max_reached_ids: IDs of notifications whose attempt count hit
            ``max_attempts`` this run (escalation halted for these).
    """

    ran_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    escalated: list[EscalationAttempt] = field(default_factory=list)
    max_reached_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EscalationJob
# ---------------------------------------------------------------------------


class EscalationJob:
    """Escalates unacknowledged high-priority notifications after a timeout.

    Notifications must first be registered via :meth:`track`.  The job does
    not subscribe to the notification router automatically; callers should
    call :meth:`track` when a relevant notification is created, and call
    :meth:`run` on a schedule (e.g. every minute).

    Escalation logic (per notification, per :meth:`run` call):

    1. Skip if ``acknowledged`` is ``True``.
    2. Skip if ``attempt_count >= max_attempts``.
    3. Check the timeout: compare ``now`` against
       ``last_escalated_at`` (or ``created_at`` if never escalated yet).
       If the elapsed time >= the configured timeout for this priority,
       escalate.
    4. Log the attempt (stub behaviour — no real delivery in beta).
    5. Increment ``attempt_count`` and update ``last_escalated_at``.

    Args:
        config: Escalation configuration.  Defaults to
            :class:`EscalationConfig` with HIGH timeout=15 minutes and
            max_attempts=3.
    """

    def __init__(self, config: EscalationConfig | None = None) -> None:
        self._config: EscalationConfig = config or EscalationConfig()
        self._tracked: dict[str, TrackedNotification] = {}
        self._all_attempts: list[EscalationAttempt] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> EscalationConfig:
        """Current escalation configuration."""
        return self._config

    def configure(self, config: EscalationConfig) -> None:
        """Replace escalation config at runtime (no code changes required)."""
        self._config = config

    def configure_from_dict(self, data: dict[str, Any]) -> None:
        """Load and apply escalation config from a plain dict."""
        self._config = EscalationConfig.from_dict(data)

    def track(self, notification: RoutableNotification) -> None:
        """Register a notification for escalation tracking.

        Only notifications whose priority appears in
        ``config.timeout_minutes_by_priority`` will ever be escalated; others
        are stored but never trigger an attempt.

        If a notification with the same ``id`` is already tracked, this call
        is silently ignored.

        Args:
            notification: The notification to track.
        """
        if notification.id in self._tracked:
            logger.debug(
                "[EscalationJob] Notification %r already tracked — ignoring duplicate.",
                notification.id,
            )
            return

        self._tracked[notification.id] = TrackedNotification(
            notification_id=notification.id,
            priority=notification.priority,
            title=notification.title,
            body=notification.body,
            created_at=notification.created_at,
        )
        logger.debug(
            "[EscalationJob] Tracking notification id=%r priority=%s.",
            notification.id,
            notification.priority.value,
        )

    def acknowledge(self, notification_id: str) -> None:
        """Mark a tracked notification as acknowledged, stopping escalation.

        If the notification is not tracked, the call is silently ignored.

        Args:
            notification_id: ID of the notification to acknowledge.
        """
        record = self._tracked.get(notification_id)
        if record is None:
            logger.debug(
                "[EscalationJob] acknowledge() called for unknown id=%r — ignoring.",
                notification_id,
            )
            return
        record.acknowledged = True
        logger.info(
            "[EscalationJob] Notification id=%r acknowledged — escalation halted.",
            notification_id,
        )

    def untrack(self, notification_id: str) -> None:
        """Remove a notification from tracking entirely.

        Args:
            notification_id: ID to remove.
        """
        self._tracked.pop(notification_id, None)

    @property
    def tracked(self) -> list[TrackedNotification]:
        """Read-only list of all currently tracked notifications."""
        return list(self._tracked.values())

    @property
    def attempts(self) -> list[EscalationAttempt]:
        """Cumulative list of all escalation attempts made across all runs."""
        return list(self._all_attempts)

    def run(self) -> EscalationResult:
        """Check tracked notifications and escalate those past their timeout.

        For each unacknowledged notification whose priority has a configured
        timeout, this method:

        - Skips it if it has already hit ``max_attempts``.
        - Escalates it if the time since creation / last escalation >= the
          timeout for its priority.
        - Logs each escalation attempt with timestamp, notification ID, and
          attempt number (stub behaviour — no real delivery in beta).

        Returns:
            An :class:`EscalationResult` with this run's escalation attempts
            and any IDs that hit the maximum attempt count.
        """
        ran_at = datetime.now(timezone.utc)
        result = EscalationResult(ran_at=ran_at)

        for record in self._tracked.values():
            if record.acknowledged:
                continue

            timeout_minutes = self._config.timeout_minutes_by_priority.get(record.priority)
            if timeout_minutes is None:
                # This priority is not configured for escalation.
                continue

            if record.attempt_count >= self._config.max_attempts:
                # Already hit the cap — nothing to do.
                continue

            # Determine when the escalation window opens.
            reference = record.last_escalated_at or record.created_at
            elapsed = ran_at - reference
            threshold = timedelta(minutes=timeout_minutes)

            if elapsed < threshold:
                continue

            # Escalate.
            record.attempt_count += 1
            record.last_escalated_at = ran_at

            attempt = EscalationAttempt(
                notification_id=record.notification_id,
                attempt_number=record.attempt_count,
                escalated_at=ran_at,
                priority=record.priority,
                title=record.title,
            )
            result.escalated.append(attempt)
            self._all_attempts.append(attempt)

            logger.warning(
                "[EscalationJob] ESCALATION STUB — id=%r attempt=%d/%d "
                "priority=%s title=%r escalated_at=%s",
                record.notification_id,
                record.attempt_count,
                self._config.max_attempts,
                record.priority.value,
                record.title,
                ran_at.isoformat(),
            )

            if record.attempt_count >= self._config.max_attempts:
                result.max_reached_ids.append(record.notification_id)
                logger.warning(
                    "[EscalationJob] Max attempts (%d) reached for id=%r — " "escalation stopped.",
                    self._config.max_attempts,
                    record.notification_id,
                )

        if not result.escalated:
            logger.debug(
                "[EscalationJob] Run at %s: no notifications eligible for escalation.",
                ran_at.isoformat(),
            )

        return result


__all__ = [
    "EscalationAttempt",
    "EscalationConfig",
    "EscalationJob",
    "EscalationResult",
    "TrackedNotification",
]
