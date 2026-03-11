"""Tests for US-091: Quiet hours.

Acceptance criteria:
- quiet hours configured as start time and end time in user config
- non-critical (medium, low) notifications generated during quiet hours are held
- critical notifications bypass quiet hours and deliver immediately
- held notifications released when quiet hours end
- Typecheck passes
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timezone

import pytest

from rex.notification_priority import NotificationPriority
from rex.priority_notification_router import (
    PriorityNotificationRouter,
    RoutableNotification,
)
from rex.quiet_hours import QuietHoursConfig, QuietHoursGuard, SubmitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt(hour: int, minute: int = 0) -> datetime:
    """UTC datetime on a fixed date at the given hour:minute."""
    return datetime(2026, 3, 11, hour, minute, tzinfo=timezone.utc)


def _notif(
    nid: str,
    priority: NotificationPriority,
    title: str = "T",
) -> RoutableNotification:
    return RoutableNotification(id=nid, priority=priority, title=title)


# Default overnight config: quiet from 22:00 to 07:00
_OVERNIGHT = QuietHoursConfig(start=time(22, 0), end=time(7, 0))
# Same-day config: quiet from 08:00 to 12:00
_SAMEDAY = QuietHoursConfig(start=time(8, 0), end=time(12, 0))


# ---------------------------------------------------------------------------
# QuietHoursConfig tests
# ---------------------------------------------------------------------------


class TestQuietHoursConfig:
    def test_default_start_end(self) -> None:
        cfg = QuietHoursConfig()
        assert cfg.start == time(22, 0)
        assert cfg.end == time(7, 0)

    def test_default_enabled(self) -> None:
        cfg = QuietHoursConfig()
        assert cfg.enabled is True

    def test_from_dict_basic(self) -> None:
        cfg = QuietHoursConfig.from_dict({"start": "23:00", "end": "06:00"})
        assert cfg.start == time(23, 0)
        assert cfg.end == time(6, 0)
        assert cfg.enabled is True

    def test_from_dict_disabled(self) -> None:
        cfg = QuietHoursConfig.from_dict({"enabled": False})
        assert cfg.enabled is False

    def test_from_dict_defaults_when_empty(self) -> None:
        cfg = QuietHoursConfig.from_dict({})
        assert cfg.start == time(22, 0)
        assert cfg.end == time(7, 0)
        assert cfg.enabled is True

    # Overnight window is_quiet tests
    def test_overnight_during_quiet(self) -> None:
        # 23:30 is inside 22:00–07:00
        assert _OVERNIGHT.is_quiet(_dt(23, 30)) is True

    def test_overnight_at_start(self) -> None:
        # Exactly at start is inside
        assert _OVERNIGHT.is_quiet(_dt(22, 0)) is True

    def test_overnight_early_morning(self) -> None:
        # 03:00 is inside 22:00–07:00
        assert _OVERNIGHT.is_quiet(_dt(3, 0)) is True

    def test_overnight_at_end_excluded(self) -> None:
        # Exactly at end (07:00) is outside
        assert _OVERNIGHT.is_quiet(_dt(7, 0)) is False

    def test_overnight_outside_afternoon(self) -> None:
        # 14:00 is outside 22:00–07:00
        assert _OVERNIGHT.is_quiet(_dt(14, 0)) is False

    def test_overnight_just_before_start(self) -> None:
        assert _OVERNIGHT.is_quiet(_dt(21, 59)) is False

    # Same-day window is_quiet tests
    def test_sameday_during_quiet(self) -> None:
        assert _SAMEDAY.is_quiet(_dt(10, 0)) is True

    def test_sameday_at_start(self) -> None:
        assert _SAMEDAY.is_quiet(_dt(8, 0)) is True

    def test_sameday_at_end_excluded(self) -> None:
        assert _SAMEDAY.is_quiet(_dt(12, 0)) is False

    def test_sameday_before_start(self) -> None:
        assert _SAMEDAY.is_quiet(_dt(7, 0)) is False

    def test_sameday_after_end(self) -> None:
        assert _SAMEDAY.is_quiet(_dt(13, 0)) is False

    # Disabled
    def test_disabled_never_quiet(self) -> None:
        cfg = QuietHoursConfig(start=time(0, 0), end=time(23, 59), enabled=False)
        assert cfg.is_quiet(_dt(12, 0)) is False


# ---------------------------------------------------------------------------
# QuietHoursGuard construction / config
# ---------------------------------------------------------------------------


class TestQuietHoursGuardInit:
    def test_default_config(self) -> None:
        guard = QuietHoursGuard(router=PriorityNotificationRouter())
        assert guard.config.start == time(22, 0)
        assert guard.config.end == time(7, 0)
        assert guard.config.enabled is True

    def test_custom_config(self) -> None:
        cfg = QuietHoursConfig(start=time(20, 0), end=time(8, 0))
        guard = QuietHoursGuard(router=PriorityNotificationRouter(), config=cfg)
        assert guard.config.start == time(20, 0)

    def test_held_initially_empty(self) -> None:
        guard = QuietHoursGuard(router=PriorityNotificationRouter())
        assert guard.held_notifications == []

    def test_configure_replaces_config(self) -> None:
        guard = QuietHoursGuard(router=PriorityNotificationRouter())
        new_cfg = QuietHoursConfig(start=time(21, 0), end=time(6, 0))
        guard.configure(new_cfg)
        assert guard.config.start == time(21, 0)

    def test_configure_from_dict(self) -> None:
        guard = QuietHoursGuard(router=PriorityNotificationRouter())
        guard.configure_from_dict({"start": "20:00", "end": "06:00"})
        assert guard.config.start == time(20, 0)
        assert guard.config.end == time(6, 0)


# ---------------------------------------------------------------------------
# Holding non-critical during quiet hours (AC: medium/low held)
# ---------------------------------------------------------------------------


class TestHoldDuringQuietHours:
    def test_medium_held_during_quiet(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        result = guard.submit(
            _notif("m1", NotificationPriority.MEDIUM), now=_dt(23, 0)
        )
        assert result.held is True
        assert len(guard.held_notifications) == 1

    def test_low_held_during_quiet(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        result = guard.submit(
            _notif("l1", NotificationPriority.LOW), now=_dt(2, 0)
        )
        assert result.held is True

    def test_high_held_during_quiet(self) -> None:
        # High is non-critical — should be held
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        result = guard.submit(
            _notif("h1", NotificationPriority.HIGH), now=_dt(23, 0)
        )
        assert result.held is True

    def test_held_count_increases(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        for i in range(5):
            guard.submit(_notif(f"m{i}", NotificationPriority.MEDIUM), now=_dt(23, 0))
        assert len(guard.held_notifications) == 5

    def test_held_notifications_returns_copy(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        guard.submit(_notif("m1", NotificationPriority.MEDIUM), now=_dt(23, 0))
        copy = guard.held_notifications
        copy.clear()
        assert len(guard.held_notifications) == 1

    def test_not_held_outside_quiet_hours(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        result = guard.submit(
            _notif("m1", NotificationPriority.MEDIUM), now=_dt(14, 0)
        )
        assert result.held is False
        assert len(guard.held_notifications) == 0


# ---------------------------------------------------------------------------
# Critical bypasses quiet hours (AC: critical delivers immediately)
# ---------------------------------------------------------------------------


class TestCriticalBypassesQuietHours:
    def test_critical_not_held(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        result = guard.submit(
            _notif("c1", NotificationPriority.CRITICAL), now=_dt(23, 0)
        )
        assert result.held is False

    def test_critical_routing_result_present(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        result = guard.submit(
            _notif("c1", NotificationPriority.CRITICAL), now=_dt(23, 0)
        )
        assert result.routing_result is not None
        assert result.routing_result.dispatched_immediately is True

    def test_critical_does_not_accumulate_in_held(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        for i in range(3):
            guard.submit(
                _notif(f"c{i}", NotificationPriority.CRITICAL), now=_dt(23, 0)
            )
        assert len(guard.held_notifications) == 0

    def test_critical_logged_as_bypass(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        with caplog.at_level(logging.INFO, logger="rex.quiet_hours"):
            guard.submit(
                _notif("c1", NotificationPriority.CRITICAL), now=_dt(23, 0)
            )
        assert any("bypassed quiet hours" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Release held notifications when quiet hours end
# ---------------------------------------------------------------------------


class TestReleaseHeld:
    def test_release_returns_count(self) -> None:
        router = PriorityNotificationRouter()
        guard = QuietHoursGuard(router=router, config=_OVERNIGHT)
        for i in range(4):
            guard.submit(_notif(f"m{i}", NotificationPriority.MEDIUM), now=_dt(23, 0))

        released = guard.release_held(now=_dt(7, 0))
        assert released == 4

    def test_held_cleared_after_release(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        for i in range(3):
            guard.submit(_notif(f"l{i}", NotificationPriority.LOW), now=_dt(23, 0))

        guard.release_held(now=_dt(7, 0))
        assert guard.held_notifications == []

    def test_released_notifications_routed(self) -> None:
        router = PriorityNotificationRouter()
        guard = QuietHoursGuard(router=router, config=_OVERNIGHT)

        for i in range(3):
            guard.submit(
                _notif(f"m{i}", NotificationPriority.MEDIUM, title=f"M{i}"),
                now=_dt(23, 0),
            )

        # Digest queue should be empty before release (held by guard, not router)
        assert len(router.digest_queue) == 0

        guard.release_held(now=_dt(7, 0))

        # After release, notifications routed through router → digest queue
        assert len(router.digest_queue) == 3

    def test_release_empty_returns_zero(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        assert guard.release_held() == 0

    def test_release_logs_count(self, caplog: pytest.LogCaptureFixture) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        guard.submit(_notif("m1", NotificationPriority.MEDIUM), now=_dt(23, 0))

        with caplog.at_level(logging.INFO, logger="rex.quiet_hours"):
            guard.release_held(now=_dt(7, 0))

        assert any("Released" in r.message for r in caplog.records)

    def test_release_preserves_order(self) -> None:
        router = PriorityNotificationRouter()
        guard = QuietHoursGuard(router=router, config=_OVERNIGHT)

        ids = ["first", "second", "third"]
        for nid in ids:
            guard.submit(
                _notif(nid, NotificationPriority.LOW, title=nid), now=_dt(23, 0)
            )

        guard.release_held(now=_dt(7, 0))

        queued_ids = [e.notification.id for e in router.digest_queue]
        assert queued_ids == ids


# ---------------------------------------------------------------------------
# Disabled guard
# ---------------------------------------------------------------------------


class TestQuietHoursDisabled:
    def test_disabled_does_not_hold(self) -> None:
        cfg = QuietHoursConfig(start=time(22, 0), end=time(7, 0), enabled=False)
        guard = QuietHoursGuard(router=PriorityNotificationRouter(), config=cfg)
        result = guard.submit(
            _notif("m1", NotificationPriority.MEDIUM), now=_dt(23, 0)
        )
        assert result.held is False

    def test_disabled_routes_normally(self) -> None:
        router = PriorityNotificationRouter()
        cfg = QuietHoursConfig(enabled=False)
        guard = QuietHoursGuard(router=router, config=cfg)
        result = guard.submit(
            _notif("l1", NotificationPriority.LOW), now=_dt(23, 0)
        )
        assert result.routing_result is not None
        assert result.routing_result.queued_for_digest is True


# ---------------------------------------------------------------------------
# is_quiet helper
# ---------------------------------------------------------------------------


class TestIsQuiet:
    def test_is_quiet_delegates_to_config(self) -> None:
        guard = QuietHoursGuard(
            router=PriorityNotificationRouter(), config=_OVERNIGHT
        )
        assert guard.is_quiet(_dt(23, 0)) is True
        assert guard.is_quiet(_dt(14, 0)) is False


# ---------------------------------------------------------------------------
# SubmitResult dataclass
# ---------------------------------------------------------------------------


class TestSubmitResult:
    def test_defaults(self) -> None:
        r = SubmitResult(notification_id="n1")
        assert r.held is False
        assert r.routing_result is None

    def test_held_true(self) -> None:
        r = SubmitResult(notification_id="n1", held=True)
        assert r.held is True
