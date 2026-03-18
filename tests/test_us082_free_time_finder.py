"""Tests for US-082: Free time finder.

Acceptance criteria:
  AC1 - given a date range and meeting duration, returns a list of available
        time slots
  AC2 - overlapping calendar events are excluded from returned slots
  AC3 - returns at least three candidate slots when the calendar is not fully
        booked
  AC4 - works correctly against stub/mock calendar data in beta
  AC5 - Typecheck passes
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rex.calendar_backends import CalendarStub, FreeBusyBlock
from rex.calendar_backends.free_time_finder import TimeSlot, find_free_slots

# ---------------------------------------------------------------------------
# Constants — stable anchor dates (from CalendarStub.DEFAULT_ANCHOR = 2026-03-09)
# ---------------------------------------------------------------------------

MONDAY = datetime(2026, 3, 9, tzinfo=timezone.utc)
TUESDAY = datetime(2026, 3, 10, tzinfo=timezone.utc)
WEDNESDAY = datetime(2026, 3, 11, tzinfo=timezone.utc)
THURSDAY = datetime(2026, 3, 12, tzinfo=timezone.utc)
FRIDAY = datetime(2026, 3, 13, tzinfo=timezone.utc)


def _utc(d: datetime, hour: int, minute: int = 0) -> datetime:
    return d.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_stub_busy(start: datetime, end: datetime) -> list[FreeBusyBlock]:
    stub = CalendarStub()
    return stub.get_free_busy(start, end)


# ---------------------------------------------------------------------------
# AC1 — returns a list of TimeSlot objects for a given date range + duration
# ---------------------------------------------------------------------------


class TestReturnsSlotList:
    def test_returns_list(self) -> None:
        busy = get_stub_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 30)
        assert isinstance(result, list)

    def test_each_item_is_time_slot(self) -> None:
        busy = get_stub_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 30)
        for slot in result:
            assert isinstance(slot, TimeSlot)

    def test_slot_end_equals_start_plus_duration(self) -> None:
        busy = get_stub_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 30)
        for slot in result:
            assert slot.end - slot.start == timedelta(minutes=30)

    def test_slots_ordered_by_start(self) -> None:
        busy = get_stub_busy(MONDAY, FRIDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, FRIDAY + timedelta(days=1), 30)
        starts = [s.start for s in result]
        assert starts == sorted(starts)

    def test_empty_busy_returns_slots(self) -> None:
        result = find_free_slots([], MONDAY, MONDAY + timedelta(days=1), 30)
        assert len(result) > 0

    def test_duration_minutes_property(self) -> None:
        busy = get_stub_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 45)
        for slot in result:
            assert slot.duration_minutes == 45

    def test_60_minute_meeting(self) -> None:
        busy = get_stub_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 60)
        assert isinstance(result, list)
        for slot in result:
            assert slot.end - slot.start == timedelta(minutes=60)

    def test_returns_empty_list_when_no_free_time(self) -> None:
        # Single busy block covering entire working day
        busy = [
            FreeBusyBlock(
                start=_utc(MONDAY, 9),
                end=_utc(MONDAY, 18),
                title="All day",
                event_id="x",
            )
        ]
        result = find_free_slots(busy, _utc(MONDAY, 9), _utc(MONDAY, 18), 30)
        assert result == []


# ---------------------------------------------------------------------------
# AC2 — overlapping events are excluded
# ---------------------------------------------------------------------------


class TestOverlappingEventsExcluded:
    def test_slot_does_not_overlap_any_busy_block(self) -> None:
        busy = get_stub_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 30)
        for slot in result:
            for block in busy:
                # Slot must not overlap [block.start, block.end)
                assert not (
                    slot.start < block.end and slot.end > block.start
                ), f"{slot} overlaps {block}"

    def test_injected_busy_block_excluded(self) -> None:
        # Block from 9:30 to 10:30 on Monday
        busy = [
            FreeBusyBlock(
                start=_utc(MONDAY, 9, 30),
                end=_utc(MONDAY, 10, 30),
                title="Injected block",
                event_id="injected",
            )
        ]
        result = find_free_slots(busy, _utc(MONDAY, 9), _utc(MONDAY, 12), 30)
        for slot in result:
            # No slot should overlap 09:30–10:30
            assert not (
                slot.start < _utc(MONDAY, 10, 30) and slot.end > _utc(MONDAY, 9, 30)
            ), f"{slot} overlaps busy block"

    def test_adjacent_slot_before_busy_block_is_valid(self) -> None:
        # Busy 10:00–11:00; slot 09:00–09:30 should be valid
        busy = [
            FreeBusyBlock(
                start=_utc(MONDAY, 10),
                end=_utc(MONDAY, 11),
                title="Meeting",
                event_id="m1",
            )
        ]
        result = find_free_slots(busy, _utc(MONDAY, 9), _utc(MONDAY, 12), 30)
        starts = [s.start for s in result]
        assert _utc(MONDAY, 9) in starts

    def test_adjacent_slot_after_busy_block_is_valid(self) -> None:
        # Busy 9:00–10:00; slot starting at 10:00 should be valid
        busy = [
            FreeBusyBlock(
                start=_utc(MONDAY, 9),
                end=_utc(MONDAY, 10),
                title="Morning block",
                event_id="m1",
            )
        ]
        result = find_free_slots(busy, _utc(MONDAY, 9), _utc(MONDAY, 12), 30)
        starts = [s.start for s in result]
        assert _utc(MONDAY, 10) in starts

    def test_slot_entirely_within_free_gap(self) -> None:
        # Busy 9:00–10:00 and 11:00–12:00 → free gap 10:00–11:00
        busy = [
            FreeBusyBlock(
                start=_utc(MONDAY, 9),
                end=_utc(MONDAY, 10),
                title="A",
                event_id="a",
            ),
            FreeBusyBlock(
                start=_utc(MONDAY, 11),
                end=_utc(MONDAY, 12),
                title="B",
                event_id="b",
            ),
        ]
        result = find_free_slots(busy, _utc(MONDAY, 9), _utc(MONDAY, 12), 30)
        # Exactly one slot should fit in the 10:00–11:00 gap
        gap_slots = [s for s in result if s.start >= _utc(MONDAY, 10)]
        assert any(s.start == _utc(MONDAY, 10) for s in gap_slots)

    def test_multi_day_slots_exclude_busy_on_correct_day(self) -> None:
        # Only add busy on Monday; Tuesday should be fully free
        busy = [
            FreeBusyBlock(
                start=_utc(MONDAY, 9),
                end=_utc(MONDAY, 18),
                title="Full Monday",
                event_id="m",
            )
        ]
        result = find_free_slots(busy, MONDAY, TUESDAY + timedelta(days=1), 30)
        tuesday_slots = [s for s in result if s.start.date() == TUESDAY.date()]
        assert len(tuesday_slots) > 0


# ---------------------------------------------------------------------------
# AC3 — at least three slots when calendar is not fully booked
# ---------------------------------------------------------------------------


class TestAtLeastThreeSlots:
    def test_full_week_yields_at_least_three(self) -> None:
        busy = get_stub_busy(MONDAY, FRIDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, FRIDAY + timedelta(days=1), 30)
        assert len(result) >= 3, f"Expected ≥3 slots, got {len(result)}"

    def test_single_day_yields_at_least_three(self) -> None:
        # Monday has gaps: 9:15–10:00 and 13:30 onwards
        busy = get_stub_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 30)
        assert len(result) >= 3, f"Expected ≥3 slots on Monday, got {len(result)}"

    def test_empty_calendar_yields_at_least_three(self) -> None:
        result = find_free_slots([], MONDAY, MONDAY + timedelta(days=1), 30)
        assert len(result) >= 3

    def test_max_slots_parameter_respected(self) -> None:
        busy = get_stub_busy(MONDAY, FRIDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, FRIDAY + timedelta(days=1), 30, max_slots=3)
        assert len(result) == 3

    def test_max_slots_larger_than_available(self) -> None:
        # Single narrow gap — only one 30-min slot
        busy = [
            FreeBusyBlock(
                start=_utc(MONDAY, 9, 30),
                end=_utc(MONDAY, 18),
                title="Long block",
                event_id="lb",
            )
        ]
        result = find_free_slots(
            busy,
            _utc(MONDAY, 9),
            _utc(MONDAY, 18),
            30,
            max_slots=10,
        )
        # At most one slot fits in 9:00–9:30
        assert len(result) <= 1


# ---------------------------------------------------------------------------
# AC4 — works against stub/mock calendar data
# ---------------------------------------------------------------------------


class TestAgainstStubData:
    def test_uses_calendar_stub_directly(self) -> None:
        stub = CalendarStub()
        start = MONDAY
        end = FRIDAY + timedelta(days=1)
        busy = stub.get_free_busy(start, end)
        result = find_free_slots(busy, start, end, 30)
        assert len(result) >= 3

    def test_stub_with_60_min_meeting(self) -> None:
        stub = CalendarStub()
        start = MONDAY
        end = FRIDAY + timedelta(days=1)
        busy = stub.get_free_busy(start, end)
        result = find_free_slots(busy, start, end, 60)
        assert len(result) >= 3

    def test_stub_with_extra_injected_event(self) -> None:
        from rex.calendar_service import CalendarEvent

        extra = CalendarEvent(
            event_id="extra-001",
            title="Extra meeting",
            start_time=_utc(THURSDAY, 10),
            end_time=_utc(THURSDAY, 11),
        )
        stub = CalendarStub(extra_events=[extra])
        start = THURSDAY
        end = THURSDAY + timedelta(days=1)
        busy = stub.get_free_busy(start, end)

        result = find_free_slots(busy, start, end, 30)
        # No slot should overlap 10:00–11:00
        for slot in result:
            assert not (
                slot.start < _utc(THURSDAY, 11) and slot.end > _utc(THURSDAY, 10)
            ), f"{slot} overlaps injected event"

    def test_stub_cleared_calendar_is_fully_free(self) -> None:
        stub = CalendarStub()
        stub.clear_events()
        busy = stub.get_free_busy(MONDAY, MONDAY + timedelta(days=1))
        assert busy == []
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 30)
        assert len(result) >= 3

    def test_no_network_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import socket

        def _no_connect(*args: object, **kwargs: object) -> None:
            raise AssertionError("Network call made during free-time finder test")

        monkeypatch.setattr(socket.socket, "connect", _no_connect)

        stub = CalendarStub()
        busy = stub.get_free_busy(MONDAY, MONDAY + timedelta(days=1))
        result = find_free_slots(busy, MONDAY, MONDAY + timedelta(days=1), 30)
        assert isinstance(result, list)

    def test_slot_step_controls_granularity(self) -> None:
        # With step=60, slots on a free day should be on the hour
        result = find_free_slots(
            [],
            _utc(MONDAY, 9),
            _utc(MONDAY, 13),
            30,
            slot_step_minutes=60,
        )
        for slot in result:
            assert slot.start.minute % 60 == 0

    def test_custom_working_hours(self) -> None:
        # Restrict to 14:00–16:00 — tiny window
        result = find_free_slots(
            [],
            _utc(MONDAY, 14),
            _utc(MONDAY, 16),
            30,
            day_start_hour=14,
            day_end_hour=16,
        )
        assert len(result) > 0
        for slot in result:
            assert slot.start >= _utc(MONDAY, 14)
            assert slot.end <= _utc(MONDAY, 16)


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_minutes must be positive"):
            find_free_slots([], MONDAY, MONDAY + timedelta(days=1), 0)

    def test_invalid_day_hours_raises(self) -> None:
        with pytest.raises(ValueError, match="day_start_hour must be less than"):
            find_free_slots(
                [],
                MONDAY,
                MONDAY + timedelta(days=1),
                30,
                day_start_hour=17,
                day_end_hour=9,
            )

    def test_invalid_step_raises(self) -> None:
        with pytest.raises(ValueError, match="slot_step_minutes must be positive"):
            find_free_slots(
                [],
                MONDAY,
                MONDAY + timedelta(days=1),
                30,
                slot_step_minutes=0,
            )

    def test_range_start_equals_end_returns_empty(self) -> None:
        result = find_free_slots([], MONDAY, MONDAY, 30)
        assert result == []

    def test_naive_datetimes_treated_as_utc(self) -> None:
        naive_start = datetime(2026, 3, 9, 9, 0)
        naive_end = datetime(2026, 3, 9, 12, 0)
        result = find_free_slots([], naive_start, naive_end, 30)
        assert len(result) > 0

    def test_time_slot_repr(self) -> None:
        slot = TimeSlot(
            start=_utc(MONDAY, 10),
            end=_utc(MONDAY, 10, 30),
        )
        r = repr(slot)
        assert "TimeSlot" in r
        assert "2026-03-09" in r
