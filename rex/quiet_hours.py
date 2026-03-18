"""Quiet hours enforcement for notifications (US-091).

Non-critical (medium, low) notifications generated during quiet hours are held
in a queue.  Critical notifications bypass quiet hours and are never held.
When quiet hours end the held notifications are released.

Configuration is loaded from user config — no code changes required to update
the schedule.

Usage example::

    from datetime import time
    from rex.quiet_hours import QuietHoursConfig, QuietHoursGuard
    from rex.notification_priority import NotificationPriority
    from rex.priority_notification_router import (
        PriorityNotificationRouter,
        RoutableNotification,
    )

    router = PriorityNotificationRouter()
    config = QuietHoursConfig(start=time(22, 0), end=time(7, 0))
    guard = QuietHoursGuard(router=router, config=config)

    notif = RoutableNotification(
        id="n1",
        priority=NotificationPriority.MEDIUM,
        title="New email",
        body="",
    )
    result = guard.submit(notif, now=datetime(2026, 3, 11, 23, 0, tzinfo=timezone.utc))
    assert result.held is True

    # Later, when quiet hours end:
    released = guard.release_held(now=datetime(2026, 3, 12, 7, 0, tzinfo=timezone.utc))
    assert released == 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any

from rex.notification_priority import NotificationPriority
from rex.priority_notification_router import (
    PriorityNotificationRouter,
    RoutableNotification,
    RoutingResult,
)

logger = logging.getLogger(__name__)

# Priorities that are NEVER held during quiet hours.
_BYPASS_PRIORITIES: frozenset[NotificationPriority] = frozenset({NotificationPriority.CRITICAL})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class QuietHoursConfig:
    """Quiet hours schedule.

    Attributes:
        start: Local time when quiet hours begin (e.g. ``time(22, 0)``).
        end: Local time when quiet hours end (e.g. ``time(7, 0)``).
        enabled: Master toggle; when ``False`` quiet hours are never active.

    The range is interpreted as follows:

    * If ``start < end`` the quiet period falls within a single calendar day
      (e.g. 08:00–12:00).
    * If ``start >= end`` the quiet period crosses midnight (e.g. 22:00–07:00
      means active from 22:00 through 06:59 of the following day).
    """

    start: time = field(default_factory=lambda: time(22, 0))
    end: time = field(default_factory=lambda: time(7, 0))
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuietHoursConfig:
        """Build config from a plain dict (e.g. loaded from JSON).

        Accepts ``"start"`` and ``"end"`` as ``"HH:MM"`` strings and
        ``"enabled"`` as a bool.

        Args:
            data: E.g. ``{"start": "22:00", "end": "07:00", "enabled": true}``.

        Returns:
            A :class:`QuietHoursConfig` instance.
        """
        start = _parse_time(data.get("start", "22:00"))
        end = _parse_time(data.get("end", "07:00"))
        enabled = bool(data.get("enabled", True))
        return cls(start=start, end=end, enabled=enabled)

    def is_quiet(self, now: datetime) -> bool:
        """Return ``True`` when *now* falls within the quiet hours window.

        Args:
            now: Timezone-aware or naive datetime to check.

        Returns:
            ``True`` if quiet hours are enabled and *now* is within the window.
        """
        if not self.enabled:
            return False
        t = now.time().replace(second=0, microsecond=0)
        if self.start < self.end:
            # Same-day window (e.g. 08:00–12:00)
            return self.start <= t < self.end
        else:
            # Overnight window (e.g. 22:00–07:00)
            return t >= self.start or t < self.end


# ---------------------------------------------------------------------------
# Submit result
# ---------------------------------------------------------------------------


@dataclass
class SubmitResult:
    """Outcome of :meth:`QuietHoursGuard.submit`.

    Attributes:
        notification_id: ID of the submitted notification.
        held: ``True`` when the notification was held (quiet hours active).
        routing_result: Result from the underlying router (when not held).
    """

    notification_id: str
    held: bool = False
    routing_result: RoutingResult | None = None


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class QuietHoursGuard:
    """Holds non-critical notifications during quiet hours.

    Wraps a :class:`~rex.priority_notification_router.PriorityNotificationRouter`
    and intercepts notifications when quiet hours are active.  Critical
    notifications always bypass the hold and are routed immediately.

    Args:
        router: Underlying router used for immediate and released delivery.
        config: Quiet hours schedule.  Defaults to 22:00–07:00 enabled.
    """

    def __init__(
        self,
        router: PriorityNotificationRouter,
        config: QuietHoursConfig | None = None,
    ) -> None:
        self._router = router
        self._config = config or QuietHoursConfig()
        self._held: list[RoutableNotification] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> QuietHoursConfig:
        """Current quiet hours configuration."""
        return self._config

    def configure(self, config: QuietHoursConfig) -> None:
        """Replace config at runtime (no code changes required)."""
        self._config = config

    def configure_from_dict(self, data: dict[str, Any]) -> None:
        """Load and apply config from a plain dict."""
        self._config = QuietHoursConfig.from_dict(data)

    @property
    def held_notifications(self) -> list[RoutableNotification]:
        """Read-only copy of currently held notifications."""
        return list(self._held)

    def is_quiet(self, now: datetime | None = None) -> bool:
        """Return ``True`` when quiet hours are currently active.

        Args:
            now: Timestamp to check (defaults to current UTC time).
        """
        ts = now or datetime.now(timezone.utc)
        return self._config.is_quiet(ts)

    def submit(
        self,
        notification: RoutableNotification,
        now: datetime | None = None,
    ) -> SubmitResult:
        """Submit a notification, holding it if quiet hours are active.

        Critical notifications always bypass quiet hours and are routed
        immediately regardless of the current time.

        Args:
            notification: The notification to submit.
            now: Override for the current time (defaults to UTC now).

        Returns:
            A :class:`SubmitResult` describing what happened.
        """
        ts = now or datetime.now(timezone.utc)

        # Critical always bypasses.
        if notification.priority in _BYPASS_PRIORITIES:
            result = self._router.route(notification)
            logger.info(
                "[QuietHoursGuard] Critical notification %s bypassed quiet hours.",
                notification.id,
            )
            return SubmitResult(
                notification_id=notification.id,
                held=False,
                routing_result=result,
            )

        if self._config.is_quiet(ts):
            self._held.append(notification)
            logger.info(
                "[QuietHoursGuard] Notification %s held during quiet hours "
                "(priority=%s, held_count=%d).",
                notification.id,
                notification.priority.value,
                len(self._held),
            )
            return SubmitResult(notification_id=notification.id, held=True)

        # Outside quiet hours — route normally.
        result = self._router.route(notification)
        return SubmitResult(
            notification_id=notification.id,
            held=False,
            routing_result=result,
        )

    def release_held(self, now: datetime | None = None) -> int:
        """Release all held notifications by routing them through the router.

        This should be called when quiet hours end.  All held notifications
        are routed in the order they were received and the hold queue is
        cleared.

        Args:
            now: Override for the current time used for logging.

        Returns:
            The number of notifications released.
        """
        if not self._held:
            logger.info("[QuietHoursGuard] release_held called; no held notifications.")
            return 0

        count = len(self._held)
        to_release = list(self._held)
        self._held.clear()

        for notif in to_release:
            try:
                self._router.route(notif)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[QuietHoursGuard] Error routing held notification %s: %s",
                    notif.id,
                    exc,
                )

        ts = now or datetime.now(timezone.utc)
        logger.info(
            "[QuietHoursGuard] Released %d held notification(s) at %s.",
            count,
            ts.isoformat(),
        )
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_time(value: str) -> time:
    """Parse an ``"HH:MM"`` string into a :class:`~datetime.time`.

    Args:
        value: Time string in ``"HH:MM"`` format.

    Returns:
        Parsed :class:`~datetime.time`.

    Raises:
        ValueError: If the string is not in ``"HH:MM"`` format.
    """
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM format, got: {value!r}")
    return time(int(parts[0]), int(parts[1]))


__all__ = [
    "QuietHoursConfig",
    "QuietHoursGuard",
    "SubmitResult",
]
