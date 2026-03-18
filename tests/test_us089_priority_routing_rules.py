"""Tests for US-089: Priority routing rules.

Acceptance criteria:
- critical and high priority notifications dispatched to configured delivery
  channels immediately on creation
- medium and low priority notifications placed in the digest queue instead of
  immediate delivery
- routing rules configurable without code changes
- unit test confirms a critical notification bypasses the digest queue
- unit test confirms a low notification is placed in the digest queue
- Typecheck passes
"""

from __future__ import annotations

from rex.notification_priority import NotificationPriority
from rex.priority_notification_router import (
    PriorityNotificationRouter,
    PriorityRoutingConfig,
    RoutableNotification,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_notif(
    priority: NotificationPriority,
    *,
    nid: str | None = None,
    channels: list[str] | None = None,
) -> RoutableNotification:
    return RoutableNotification(
        id=nid or f"n_{priority.value}",
        priority=priority,
        title=f"Test {priority.label}",
        body="body text",
        channels=channels or ["dashboard"],
    )


def make_router(dispatcher=None) -> PriorityNotificationRouter:
    return PriorityNotificationRouter(immediate_dispatcher=dispatcher)


# ---------------------------------------------------------------------------
# RoutableNotification
# ---------------------------------------------------------------------------


class TestRoutableNotification:
    def test_minimal_construction(self) -> None:
        n = RoutableNotification(id="x", priority=NotificationPriority.HIGH, title="Hi")
        assert n.id == "x"
        assert n.priority == NotificationPriority.HIGH
        assert n.channels == ["dashboard"]
        assert n.metadata == {}

    def test_full_construction(self) -> None:
        n = RoutableNotification(
            id="y",
            priority=NotificationPriority.LOW,
            title="Lo",
            body="b",
            channels=["email", "sms"],
            metadata={"k": "v"},
        )
        assert n.channels == ["email", "sms"]
        assert n.metadata["k"] == "v"


# ---------------------------------------------------------------------------
# PriorityRoutingConfig
# ---------------------------------------------------------------------------


class TestPriorityRoutingConfig:
    def test_default_immediate_priorities(self) -> None:
        cfg = PriorityRoutingConfig()
        assert NotificationPriority.CRITICAL in cfg.immediate_priorities
        assert NotificationPriority.HIGH in cfg.immediate_priorities

    def test_default_digest_priorities(self) -> None:
        cfg = PriorityRoutingConfig()
        assert NotificationPriority.MEDIUM in cfg.digest_priorities
        assert NotificationPriority.LOW in cfg.digest_priorities

    def test_from_dict_default(self) -> None:
        cfg = PriorityRoutingConfig.from_dict({})
        assert NotificationPriority.CRITICAL in cfg.immediate_priorities
        assert NotificationPriority.LOW in cfg.digest_priorities

    def test_from_dict_custom(self) -> None:
        cfg = PriorityRoutingConfig.from_dict(
            {"immediate_priorities": ["critical"], "digest_priorities": ["high", "medium", "low"]}
        )
        assert cfg.immediate_priorities == frozenset({NotificationPriority.CRITICAL})
        assert NotificationPriority.HIGH in cfg.digest_priorities

    def test_from_dict_unknown_string_becomes_medium(self) -> None:
        cfg = PriorityRoutingConfig.from_dict({"immediate_priorities": ["super_urgent"]})
        # "super_urgent" normalises to MEDIUM via from_str
        assert NotificationPriority.MEDIUM in cfg.immediate_priorities


# ---------------------------------------------------------------------------
# AC: critical notification dispatched immediately — bypasses digest queue
# ---------------------------------------------------------------------------


class TestCriticalBypassesDigestQueue:
    def test_critical_dispatched_immediately(self) -> None:
        """Critical notification dispatched_immediately=True."""
        router = make_router()
        result = router.route(make_notif(NotificationPriority.CRITICAL))
        assert result.dispatched_immediately is True

    def test_critical_not_queued_for_digest(self) -> None:
        """Critical notification is NOT placed in the digest queue."""
        router = make_router()
        router.route(make_notif(NotificationPriority.CRITICAL))
        assert len(router.digest_queue) == 0

    def test_critical_digest_queue_remains_empty(self) -> None:
        """After routing a critical notification, digest queue stays empty."""
        router = make_router()
        for _ in range(3):
            router.route(make_notif(NotificationPriority.CRITICAL))
        assert len(router.digest_queue) == 0

    def test_critical_result_has_correct_fields(self) -> None:
        router = make_router()
        result = router.route(make_notif(NotificationPriority.CRITICAL, nid="crit-1"))
        assert result.notification_id == "crit-1"
        assert result.priority == NotificationPriority.CRITICAL
        assert result.dispatched_immediately is True
        assert result.queued_for_digest is False


# ---------------------------------------------------------------------------
# AC: high notification dispatched immediately
# ---------------------------------------------------------------------------


class TestHighDispatchedImmediately:
    def test_high_dispatched_immediately(self) -> None:
        router = make_router()
        result = router.route(make_notif(NotificationPriority.HIGH))
        assert result.dispatched_immediately is True
        assert result.queued_for_digest is False

    def test_high_not_in_digest_queue(self) -> None:
        router = make_router()
        router.route(make_notif(NotificationPriority.HIGH))
        assert len(router.digest_queue) == 0


# ---------------------------------------------------------------------------
# AC: low notification placed in digest queue
# ---------------------------------------------------------------------------


class TestLowPlacedInDigestQueue:
    def test_low_queued_for_digest(self) -> None:
        """Low notification placed in digest queue, not dispatched immediately."""
        router = make_router()
        result = router.route(make_notif(NotificationPriority.LOW))
        assert result.queued_for_digest is True

    def test_low_not_dispatched_immediately(self) -> None:
        router = make_router()
        result = router.route(make_notif(NotificationPriority.LOW))
        assert result.dispatched_immediately is False

    def test_low_appears_in_digest_queue(self) -> None:
        router = make_router()
        router.route(make_notif(NotificationPriority.LOW, nid="low-1"))
        assert len(router.digest_queue) == 1
        assert router.digest_queue[0].notification.id == "low-1"

    def test_low_result_has_correct_fields(self) -> None:
        router = make_router()
        result = router.route(make_notif(NotificationPriority.LOW, nid="low-2"))
        assert result.notification_id == "low-2"
        assert result.priority == NotificationPriority.LOW
        assert result.queued_for_digest is True


# ---------------------------------------------------------------------------
# AC: medium notification placed in digest queue
# ---------------------------------------------------------------------------


class TestMediumPlacedInDigestQueue:
    def test_medium_queued_for_digest(self) -> None:
        router = make_router()
        result = router.route(make_notif(NotificationPriority.MEDIUM))
        assert result.queued_for_digest is True
        assert result.dispatched_immediately is False

    def test_medium_appears_in_digest_queue(self) -> None:
        router = make_router()
        router.route(make_notif(NotificationPriority.MEDIUM, nid="med-1"))
        assert any(e.notification.id == "med-1" for e in router.digest_queue)


# ---------------------------------------------------------------------------
# AC: immediate dispatcher called for immediate priorities
# ---------------------------------------------------------------------------


class TestImmediateDispatcherCalled:
    def test_dispatcher_called_for_critical(self) -> None:
        calls: list[tuple[str, str]] = []

        def dispatcher(notif: RoutableNotification, channel: str) -> None:
            calls.append((notif.id, channel))

        router = PriorityNotificationRouter(immediate_dispatcher=dispatcher)
        router.route(make_notif(NotificationPriority.CRITICAL, nid="c1"))
        assert ("c1", "dashboard") in calls

    def test_dispatcher_called_for_high(self) -> None:
        calls: list[tuple[str, str]] = []

        def dispatcher(notif: RoutableNotification, channel: str) -> None:
            calls.append((notif.id, channel))

        router = PriorityNotificationRouter(immediate_dispatcher=dispatcher)
        router.route(make_notif(NotificationPriority.HIGH, nid="h1"))
        assert ("h1", "dashboard") in calls

    def test_dispatcher_not_called_for_low(self) -> None:
        calls: list[tuple[str, str]] = []

        def dispatcher(notif: RoutableNotification, channel: str) -> None:
            calls.append((notif.id, channel))

        router = PriorityNotificationRouter(immediate_dispatcher=dispatcher)
        router.route(make_notif(NotificationPriority.LOW, nid="l1"))
        assert len(calls) == 0

    def test_dispatcher_called_for_each_channel(self) -> None:
        channels: list[str] = []

        def dispatcher(notif: RoutableNotification, channel: str) -> None:
            channels.append(channel)

        router = PriorityNotificationRouter(immediate_dispatcher=dispatcher)
        n = RoutableNotification(
            id="multi",
            priority=NotificationPriority.CRITICAL,
            title="Multi",
            channels=["dashboard", "sms", "email"],
        )
        result = router.route(n)
        assert set(channels) == {"dashboard", "sms", "email"}
        assert result.channels_dispatched == ["dashboard", "sms", "email"]

    def test_dispatcher_error_recorded_in_result(self) -> None:
        def bad_dispatcher(notif: RoutableNotification, channel: str) -> None:
            raise RuntimeError("channel down")

        router = PriorityNotificationRouter(immediate_dispatcher=bad_dispatcher)
        result = router.route(make_notif(NotificationPriority.HIGH))
        assert result.error is not None
        assert "channel down" in result.error


# ---------------------------------------------------------------------------
# AC: routing rules configurable without code changes
# ---------------------------------------------------------------------------


class TestRoutingRulesConfigurable:
    def test_configure_method_changes_rules(self) -> None:
        """Calling configure() changes which priorities go to digest queue."""
        router = make_router()
        # By default HIGH is immediate; reconfigure to make HIGH a digest priority
        new_cfg = PriorityRoutingConfig(
            immediate_priorities=frozenset({NotificationPriority.CRITICAL}),
            digest_priorities=frozenset(
                {NotificationPriority.HIGH, NotificationPriority.MEDIUM, NotificationPriority.LOW}
            ),
        )
        router.configure(new_cfg)
        result = router.route(make_notif(NotificationPriority.HIGH))
        assert result.queued_for_digest is True

    def test_configure_from_dict(self) -> None:
        """configure_from_dict() loads rules from a plain dict."""
        router = make_router()
        router.configure_from_dict(
            {
                "immediate_priorities": ["critical", "high"],
                "digest_priorities": ["medium", "low"],
            }
        )
        assert router.route(make_notif(NotificationPriority.CRITICAL)).dispatched_immediately
        assert router.route(make_notif(NotificationPriority.LOW)).queued_for_digest

    def test_rules_take_effect_without_restart(self) -> None:
        """Routing decision changes immediately after configure_from_dict call."""
        router = make_router()
        # Initially HIGH is immediate
        r1 = router.route(make_notif(NotificationPriority.HIGH, nid="a"))
        assert r1.dispatched_immediately is True

        # Reconfigure so HIGH goes to digest
        router.configure_from_dict(
            {
                "immediate_priorities": ["critical"],
                "digest_priorities": ["high", "medium", "low"],
            }
        )
        r2 = router.route(make_notif(NotificationPriority.HIGH, nid="b"))
        assert r2.queued_for_digest is True

    def test_all_critical_immediate_regardless_of_count(self) -> None:
        router = make_router()
        results = [
            router.route(make_notif(NotificationPriority.CRITICAL, nid=f"c{i}")) for i in range(5)
        ]
        assert all(r.dispatched_immediately for r in results)
        assert len(router.digest_queue) == 0

    def test_all_low_to_digest_regardless_of_count(self) -> None:
        router = make_router()
        for i in range(5):
            router.route(make_notif(NotificationPriority.LOW, nid=f"l{i}"))
        assert len(router.digest_queue) == 5


# ---------------------------------------------------------------------------
# Digest queue management
# ---------------------------------------------------------------------------


class TestDigestQueueManagement:
    def test_drain_digest_queue_returns_entries(self) -> None:
        router = make_router()
        router.route(make_notif(NotificationPriority.LOW, nid="l1"))
        router.route(make_notif(NotificationPriority.MEDIUM, nid="m1"))
        entries = router.drain_digest_queue()
        assert len(entries) == 2

    def test_drain_clears_the_queue(self) -> None:
        router = make_router()
        router.route(make_notif(NotificationPriority.LOW))
        router.drain_digest_queue()
        assert len(router.digest_queue) == 0

    def test_digest_queue_property_is_a_copy(self) -> None:
        """Mutating the returned list does not affect internal state."""
        router = make_router()
        router.route(make_notif(NotificationPriority.MEDIUM, nid="x"))
        snapshot = router.digest_queue
        snapshot.clear()
        assert len(router.digest_queue) == 1

    def test_mixed_routing_populates_only_digest_queue_for_low(self) -> None:
        router = make_router()
        router.route(make_notif(NotificationPriority.CRITICAL))
        router.route(make_notif(NotificationPriority.LOW))
        router.route(make_notif(NotificationPriority.HIGH))
        router.route(make_notif(NotificationPriority.MEDIUM))
        assert len(router.digest_queue) == 2
