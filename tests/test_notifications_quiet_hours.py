"""Unit tests for rex.notifications.quiet_hours — QuietHoursGate."""

from __future__ import annotations

from datetime import time

from rex.notifications.models import Notification
from rex.notifications.quiet_hours import QuietHoursGate, _is_in_quiet_window, _parse_time

# ---------------------------------------------------------------------------
# _parse_time helpers
# ---------------------------------------------------------------------------


class TestParseTime:
    def test_valid_hhmm(self) -> None:
        assert _parse_time("07:30") == time(7, 30)

    def test_valid_midnight(self) -> None:
        assert _parse_time("00:00") == time(0, 0)

    def test_valid_23_59(self) -> None:
        assert _parse_time("23:59") == time(23, 59)

    def test_invalid_returns_none(self) -> None:
        assert _parse_time("bad") is None
        assert _parse_time("25:00") is None
        assert _parse_time("") is None
        assert _parse_time("7") is None


# ---------------------------------------------------------------------------
# _is_in_quiet_window
# ---------------------------------------------------------------------------


class TestIsInQuietWindow:
    # Same-day window 22:00 – 23:30
    def test_same_day_inside(self) -> None:
        assert _is_in_quiet_window(time(22, 30), time(22, 0), time(23, 30))

    def test_same_day_before(self) -> None:
        assert not _is_in_quiet_window(time(21, 59), time(22, 0), time(23, 30))

    def test_same_day_at_start(self) -> None:
        assert _is_in_quiet_window(time(22, 0), time(22, 0), time(23, 30))

    def test_same_day_at_end(self) -> None:
        # end is exclusive
        assert not _is_in_quiet_window(time(23, 30), time(22, 0), time(23, 30))

    def test_same_day_after(self) -> None:
        assert not _is_in_quiet_window(time(23, 31), time(22, 0), time(23, 30))

    # Overnight window 23:00 – 07:00
    def test_overnight_before_midnight(self) -> None:
        assert _is_in_quiet_window(time(23, 30), time(23, 0), time(7, 0))

    def test_overnight_at_midnight(self) -> None:
        assert _is_in_quiet_window(time(0, 0), time(23, 0), time(7, 0))

    def test_overnight_after_midnight(self) -> None:
        assert _is_in_quiet_window(time(3, 0), time(23, 0), time(7, 0))

    def test_overnight_just_before_end(self) -> None:
        assert _is_in_quiet_window(time(6, 59), time(23, 0), time(7, 0))

    def test_overnight_at_end(self) -> None:
        # end is exclusive
        assert not _is_in_quiet_window(time(7, 0), time(23, 0), time(7, 0))

    def test_overnight_after_end(self) -> None:
        assert not _is_in_quiet_window(time(12, 0), time(23, 0), time(7, 0))

    def test_overnight_at_start(self) -> None:
        assert _is_in_quiet_window(time(23, 0), time(23, 0), time(7, 0))

    def test_day_between_start_and_midnight(self) -> None:
        # 14:00 is in the middle of the day — not in overnight window
        assert not _is_in_quiet_window(time(14, 0), time(23, 0), time(7, 0))

    # Equal start/end → full-day suppression
    def test_equal_start_end_always_quiet(self) -> None:
        for h in (0, 6, 12, 18, 23):
            assert _is_in_quiet_window(time(h, 0), time(12, 0), time(12, 0))


# ---------------------------------------------------------------------------
# QuietHoursGate
# ---------------------------------------------------------------------------


def _gate(start: str, end: str, current: time) -> QuietHoursGate:
    """Build a QuietHoursGate with fixed config and a fixed clock."""
    config: dict[str, object] = {
        "notifications_quiet_hours_start": start,
        "notifications_quiet_hours_end": end,
    }
    return QuietHoursGate(config=config, clock=lambda: current)


def _notification(quiet_hours_exempt: bool = False) -> Notification:
    return Notification(
        title="Test",
        body="Body",
        source="test",
        quiet_hours_exempt=quiet_hours_exempt,
    )


class TestIsQuietNow:
    def test_quiet_during_window(self) -> None:
        gate = _gate("22:00", "23:00", time(22, 30))
        assert gate.is_quiet_now() is True

    def test_not_quiet_outside_window(self) -> None:
        gate = _gate("22:00", "23:00", time(20, 0))
        assert gate.is_quiet_now() is False

    def test_overnight_window_before_midnight(self) -> None:
        gate = _gate("23:00", "07:00", time(23, 30))
        assert gate.is_quiet_now() is True

    def test_overnight_window_after_midnight(self) -> None:
        gate = _gate("23:00", "07:00", time(2, 0))
        assert gate.is_quiet_now() is True

    def test_overnight_window_outside(self) -> None:
        gate = _gate("23:00", "07:00", time(12, 0))
        assert gate.is_quiet_now() is False

    def test_missing_config_returns_false(self) -> None:
        gate = QuietHoursGate(config={}, clock=lambda: time(3, 0))
        assert gate.is_quiet_now() is False

    def test_malformed_start_returns_false(self) -> None:
        gate = QuietHoursGate(
            config={
                "notifications_quiet_hours_start": "bad",
                "notifications_quiet_hours_end": "07:00",
            },
            clock=lambda: time(3, 0),
        )
        assert gate.is_quiet_now() is False

    def test_malformed_end_returns_false(self) -> None:
        gate = QuietHoursGate(
            config={
                "notifications_quiet_hours_start": "23:00",
                "notifications_quiet_hours_end": "",
            },
            clock=lambda: time(3, 0),
        )
        assert gate.is_quiet_now() is False


class TestShouldSuppress:
    def test_suppresses_when_quiet_and_not_exempt(self) -> None:
        gate = _gate("22:00", "23:00", time(22, 30))
        assert gate.should_suppress(_notification(quiet_hours_exempt=False)) is True

    def test_does_not_suppress_when_quiet_but_exempt(self) -> None:
        gate = _gate("22:00", "23:00", time(22, 30))
        assert gate.should_suppress(_notification(quiet_hours_exempt=True)) is False

    def test_does_not_suppress_outside_quiet_hours(self) -> None:
        gate = _gate("22:00", "23:00", time(10, 0))
        assert gate.should_suppress(_notification(quiet_hours_exempt=False)) is False

    def test_does_not_suppress_outside_quiet_exempt(self) -> None:
        gate = _gate("22:00", "23:00", time(10, 0))
        assert gate.should_suppress(_notification(quiet_hours_exempt=True)) is False
