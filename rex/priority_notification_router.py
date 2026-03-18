"""Priority-based notification router (US-089).

Routes outbound notifications to immediate delivery or a digest queue based
on priority level.  Critical and high priority notifications are dispatched
immediately; medium and low are placed in the digest queue.

Routing rules are stored in :class:`PriorityRoutingConfig` and can be changed
at runtime or loaded from a config dict without any code changes.

Usage example::

    from rex.priority_notification_router import (
        PriorityNotificationRouter,
        PriorityRoutingConfig,
        RoutableNotification,
    )
    from rex.notification_priority import NotificationPriority

    router = PriorityNotificationRouter()

    notif = RoutableNotification(
        id="n1",
        priority=NotificationPriority.CRITICAL,
        title="Server Down",
        body="Database unreachable",
    )
    result = router.route(notif)
    assert result.dispatched_immediately is True
    assert result.queued_for_digest is False
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from rex.notification_priority import NotificationPriority

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default rule sets (configurable)
# ---------------------------------------------------------------------------

_DEFAULT_IMMEDIATE: frozenset[NotificationPriority] = frozenset(
    {NotificationPriority.CRITICAL, NotificationPriority.HIGH}
)
_DEFAULT_DIGEST: frozenset[NotificationPriority] = frozenset(
    {NotificationPriority.MEDIUM, NotificationPriority.LOW}
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RoutableNotification:
    """A notification that can be routed by :class:`PriorityNotificationRouter`.

    Attributes:
        id: Unique notification identifier.
        priority: Priority level determining routing behaviour.
        title: Notification subject / title.
        body: Notification body text.
        channels: Ordered preferred delivery channels.
        metadata: Arbitrary key-value pairs forwarded to dispatchers.
        created_at: Creation timestamp (UTC).
    """

    id: str
    priority: NotificationPriority
    title: str
    body: str = ""
    channels: list[str] = field(default_factory=lambda: ["dashboard"])
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RoutingResult:
    """Outcome of a :meth:`PriorityNotificationRouter.route` call.

    Attributes:
        notification_id: The routed notification's ID.
        priority: Priority level that was used for the decision.
        dispatched_immediately: ``True`` when the notification was dispatched
            immediately (critical / high by default).
        queued_for_digest: ``True`` when the notification was placed in the
            digest queue (medium / low by default).
        channels_dispatched: Channels that received immediate delivery.
        error: Human-readable error if delivery raised an exception.
    """

    notification_id: str
    priority: NotificationPriority
    dispatched_immediately: bool = False
    queued_for_digest: bool = False
    channels_dispatched: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class DigestEntry:
    """A single entry in the digest queue."""

    notification: RoutableNotification
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PriorityRoutingConfig:
    """Runtime configuration for :class:`PriorityNotificationRouter`.

    Changing any field takes effect immediately on the next :meth:`route` call
    — no code changes or restarts required.

    Attributes:
        immediate_priorities: Priorities that trigger immediate dispatch.
            Defaults to ``{critical, high}``.
        digest_priorities: Priorities that are queued for digest delivery.
            Defaults to ``{medium, low}``.
    """

    immediate_priorities: frozenset[NotificationPriority] = field(
        default_factory=lambda: _DEFAULT_IMMEDIATE
    )
    digest_priorities: frozenset[NotificationPriority] = field(
        default_factory=lambda: _DEFAULT_DIGEST
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorityRoutingConfig:
        """Build config from a plain dict (e.g. loaded from JSON).

        Accepts lists of priority strings under ``"immediate_priorities"`` and
        ``"digest_priorities"`` keys.  Unrecognised strings default to
        ``NotificationPriority.MEDIUM`` via :meth:`~NotificationPriority.from_str`.

        Args:
            data: Config dict, e.g. ``{"immediate_priorities": ["critical", "high"]}``.

        Returns:
            A :class:`PriorityRoutingConfig` instance.
        """
        immediate = frozenset(
            NotificationPriority.from_str(v)
            for v in data.get("immediate_priorities", ["critical", "high"])
        )
        digest = frozenset(
            NotificationPriority.from_str(v)
            for v in data.get("digest_priorities", ["medium", "low"])
        )
        return cls(immediate_priorities=immediate, digest_priorities=digest)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Type alias for an immediate-delivery callback.
ImmediateDispatcher = Callable[[RoutableNotification, str], None]


class PriorityNotificationRouter:
    """Routes notifications to immediate delivery or the digest queue.

    Routing decision:
    - If ``notification.priority`` is in ``config.immediate_priorities`` →
      call each registered :attr:`immediate_dispatcher` and mark dispatched.
    - If ``notification.priority`` is in ``config.digest_priorities`` →
      append to the in-memory digest queue.
    - If a priority falls in neither set (edge case) → log a warning and
      place in digest queue as the safe fallback.

    Args:
        config: Routing rules.  Defaults to :class:`PriorityRoutingConfig`
            with ``{critical, high}`` as immediate and ``{medium, low}`` as
            digest.
        immediate_dispatcher: Optional callable invoked for each channel when
            dispatching immediately.  Signature:
            ``(notification: RoutableNotification, channel: str) -> None``.
            When ``None`` the router logs dispatch actions only (stub behaviour).
    """

    def __init__(
        self,
        config: PriorityRoutingConfig | None = None,
        immediate_dispatcher: ImmediateDispatcher | None = None,
    ) -> None:
        self._config = config or PriorityRoutingConfig()
        self._dispatcher = immediate_dispatcher
        self._digest_queue: list[DigestEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> PriorityRoutingConfig:
        """Current routing configuration."""
        return self._config

    def configure(self, config: PriorityRoutingConfig) -> None:
        """Replace routing config at runtime (no code changes required)."""
        self._config = config

    def configure_from_dict(self, data: dict[str, Any]) -> None:
        """Load and apply routing config from a plain dict."""
        self._config = PriorityRoutingConfig.from_dict(data)

    @property
    def digest_queue(self) -> list[DigestEntry]:
        """Read-only view of the current digest queue entries."""
        return list(self._digest_queue)

    def drain_digest_queue(self) -> list[DigestEntry]:
        """Remove and return all entries currently in the digest queue."""
        entries = list(self._digest_queue)
        self._digest_queue.clear()
        return entries

    def route(self, notification: RoutableNotification) -> RoutingResult:
        """Route *notification* according to current priority rules.

        Args:
            notification: The notification to route.

        Returns:
            A :class:`RoutingResult` describing the routing outcome.
        """
        priority = notification.priority

        if priority in self._config.immediate_priorities:
            return self._dispatch_immediately(notification)

        if priority in self._config.digest_priorities:
            return self._enqueue_digest(notification)

        # Fallback: unknown priority → digest (safe default).
        logger.warning(
            "[PriorityNotificationRouter] Priority %r not in any rule set; "
            "falling back to digest queue.",
            priority.value,
        )
        return self._enqueue_digest(notification)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_immediately(self, notification: RoutableNotification) -> RoutingResult:
        channels_ok: list[str] = []
        error_msg: str | None = None

        for channel in notification.channels:
            try:
                if self._dispatcher is not None:
                    self._dispatcher(notification, channel)
                else:
                    logger.info(
                        "[PriorityNotificationRouter] Immediate dispatch to %r: %s",
                        channel,
                        notification.title,
                    )
                channels_ok.append(channel)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[PriorityNotificationRouter] Dispatch to %r failed: %s",
                    channel,
                    exc,
                )
                error_msg = str(exc)

        logger.info(
            "[PriorityNotificationRouter] Priority=%s id=%s dispatched immediately "
            "to %d channel(s).",
            notification.priority.value,
            notification.id,
            len(channels_ok),
        )
        return RoutingResult(
            notification_id=notification.id,
            priority=notification.priority,
            dispatched_immediately=True,
            queued_for_digest=False,
            channels_dispatched=channels_ok,
            error=error_msg,
        )

    def _enqueue_digest(self, notification: RoutableNotification) -> RoutingResult:
        self._digest_queue.append(DigestEntry(notification=notification))
        logger.info(
            "[PriorityNotificationRouter] Priority=%s id=%s queued for digest " "(queue size=%d).",
            notification.priority.value,
            notification.id,
            len(self._digest_queue),
        )
        return RoutingResult(
            notification_id=notification.id,
            priority=notification.priority,
            dispatched_immediately=False,
            queued_for_digest=True,
        )


__all__ = [
    "DigestEntry",
    "ImmediateDispatcher",
    "PriorityNotificationRouter",
    "PriorityRoutingConfig",
    "RoutableNotification",
    "RoutingResult",
]
