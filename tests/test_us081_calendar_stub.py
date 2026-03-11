"""Tests for US-081: Calendar free/busy stub.

Acceptance criteria verified:
- [x] CalendarStub class returns mock free/busy blocks for a configurable date range
- [x] stub implements the same interface as the real calendar backend (US-045)
- [x] tests can query availability without any live credentials or network calls
- [x] Typecheck passes (enforced by mypy in CI)
"""

from __future__ import annotations

import socket
from datetime import date, datetime, timedelta, timezone

import pytest

from rex.calendar_backends.base import CalendarBackend
from rex.calendar_backends.free_busy_stub import CalendarStub, FreeBusyBlock
from rex.calendar_service import CalendarEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANCHOR = date(2026, 3, 9)  # Monday


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _window(day_offset: int, start_h: int, end_h: int) -> tuple[datetime, datetime]:
    """Return a [start, end) window relative to the default anchor Monday."""
    monday = datetime(2026, 3, 9, tzinfo=timezone.utc)
    base = monday + timedelta(days=day_offset)
    return (
        base.replace(hour=start_h, minute=0, second=0, microsecond=0),
        base.replace(hour=end_h, minute=0, second=0, microsecond=0),
    )


# ---------------------------------------------------------------------------
# AC1 — CalendarStub returns mock free/busy blocks for a configurable date range
# ---------------------------------------------------------------------------


class TestMockFreeBusyData:
    def test_stub_has_built_in_events(self) -> None:
        stub = CalendarStub()
        assert len(stub.all_events) >= 1

    def test_stub_spans_full_working_week(self) -> None:
        stub = CalendarStub()
        events = stub.all_events
        days = {e.start_time.date() for e in events}
        # Should cover Mon–Fri of the anchor week
        monday = date(2026, 3, 9)
        expected = {monday + timedelta(days=i) for i in range(5)}
        assert expected.issubset(days)

    def test_get_free_busy_returns_blocks(self) -> None:
        stub = CalendarStub()
        stub.connect()
        start, end = _window(0, 8, 18)  # all of Monday
        blocks = stub.get_free_busy(start, end)
        assert len(blocks) >= 1

    def test_get_free_busy_returns_free_busy_block_objects(self) -> None:
        stub = CalendarStub()
        stub.connect()
        start, end = _window(0, 8, 18)
        for block in stub.get_free_busy(start, end):
            assert isinstance(block, FreeBusyBlock)

    def test_free_busy_blocks_are_ordered_by_start(self) -> None:
        stub = CalendarStub()
        stub.connect()
        monday = _utc(2026, 3, 9, 8)
        friday_end = _utc(2026, 3, 13, 18)
        blocks = stub.get_free_busy(monday, friday_end)
        starts = [b.start for b in blocks]
        assert starts == sorted(starts)

    def test_free_busy_block_has_required_fields(self) -> None:
        stub = CalendarStub()
        stub.connect()
        start, end = _window(0, 8, 18)
        blocks = stub.get_free_busy(start, end)
        assert len(blocks) >= 1
        block = blocks[0]
        assert isinstance(block.start, datetime)
        assert isinstance(block.end, datetime)
        assert isinstance(block.title, str)
        assert isinstance(block.event_id, str)
        assert block.start < block.end

    def test_anchor_date_is_configurable(self) -> None:
        anchor = date(2026, 4, 6)  # a different Monday
        stub = CalendarStub(anchor=anchor)
        events = stub.all_events
        assert any(e.start_time.date() == anchor for e in events)

    def test_extra_events_injected_at_construction(self) -> None:
        extra = CalendarEvent(
            event_id="custom-001",
            title="Custom meeting",
            start_time=_utc(2026, 3, 10, 15),
            end_time=_utc(2026, 3, 10, 16),
        )
        stub = CalendarStub(extra_events=[extra])
        ids = {e.event_id for e in stub.all_events}
        assert "custom-001" in ids

    def test_inject_event_at_runtime(self) -> None:
        stub = CalendarStub()
        before = len(stub.all_events)
        stub.inject_event(
            CalendarEvent(
                event_id="runtime-001",
                title="Late addition",
                start_time=_utc(2026, 3, 11, 16),
                end_time=_utc(2026, 3, 11, 17),
            )
        )
        assert len(stub.all_events) == before + 1

    def test_clear_events_empties_calendar(self) -> None:
        stub = CalendarStub()
        stub.clear_events()
        assert stub.all_events == []

    def test_get_free_busy_with_empty_calendar_returns_empty(self) -> None:
        stub = CalendarStub()
        stub.clear_events()
        stub.connect()
        start, end = _window(0, 8, 18)
        assert stub.get_free_busy(start, end) == []


# ---------------------------------------------------------------------------
# AC1b — configurable date range filtering
# ---------------------------------------------------------------------------


class TestFreeBusyFiltering:
    def test_get_free_busy_filters_to_requested_window(self) -> None:
        stub = CalendarStub()
        stub.connect()
        # Request only Tuesday afternoon
        start, end = _window(1, 13, 18)  # Tuesday 13:00–18:00
        blocks = stub.get_free_busy(start, end)
        for block in blocks:
            # All returned blocks must overlap the window
            assert block.start < end and block.end > start

    def test_get_free_busy_excludes_events_outside_window(self) -> None:
        stub = CalendarStub()
        stub.connect()
        # Request only a narrow morning window on Wednesday 09:00–09:30
        # (stand-up is 09:00–09:15, so it should appear)
        # Architecture review is 11:00–12:30, should NOT appear
        start, end = _window(2, 9, 10)
        blocks = stub.get_free_busy(start, end)
        titles = {b.title for b in blocks}
        assert "Daily stand-up" in titles
        assert "Architecture review" not in titles

    def test_events_that_span_the_window_boundary_are_included(self) -> None:
        stub = CalendarStub()
        stub.clear_events()
        stub.connect()
        # Event spans 08:30–09:30, window is 09:00–10:00 → overlaps
        stub.inject_event(
            CalendarEvent(
                event_id="span-001",
                title="Spanning meeting",
                start_time=_utc(2026, 3, 9, 8, 30),
                end_time=_utc(2026, 3, 9, 9, 30),
            )
        )
        start = _utc(2026, 3, 9, 9, 0)
        end = _utc(2026, 3, 9, 10, 0)
        blocks = stub.get_free_busy(start, end)
        assert any(b.title == "Spanning meeting" for b in blocks)

    def test_adjacent_event_not_in_window(self) -> None:
        stub = CalendarStub()
        stub.clear_events()
        stub.connect()
        # Event ends exactly at window start — should NOT appear (exclusive end)
        stub.inject_event(
            CalendarEvent(
                event_id="adj-001",
                title="Adjacent before",
                start_time=_utc(2026, 3, 9, 8, 0),
                end_time=_utc(2026, 3, 9, 9, 0),
            )
        )
        start = _utc(2026, 3, 9, 9, 0)
        end = _utc(2026, 3, 9, 10, 0)
        blocks = stub.get_free_busy(start, end)
        assert not any(b.title == "Adjacent before" for b in blocks)

    def test_whole_week_window_returns_all_events(self) -> None:
        stub = CalendarStub()
        stub.connect()
        monday = _utc(2026, 3, 9, 0)
        saturday = _utc(2026, 3, 14, 0)
        blocks = stub.get_free_busy(monday, saturday)
        assert len(blocks) == len(stub.all_events)


# ---------------------------------------------------------------------------
# AC2 — implements the same interface as the real calendar backend
# ---------------------------------------------------------------------------


class TestCalendarBackendInterface:
    def test_calendar_stub_is_subclass_of_calendar_backend(self) -> None:
        assert issubclass(CalendarStub, CalendarBackend)

    def test_connect_returns_true(self) -> None:
        stub = CalendarStub()
        assert stub.connect() is True

    def test_fetch_events_returns_list_of_calendar_events(self) -> None:
        stub = CalendarStub()
        stub.connect()
        events = stub.fetch_events()
        assert isinstance(events, list)
        assert all(isinstance(e, CalendarEvent) for e in events)

    def test_fetch_events_auto_connects(self) -> None:
        stub = CalendarStub()
        # Do NOT call connect() — fetch_events should auto-connect
        events = stub.fetch_events()
        assert len(events) >= 1

    def test_disconnect_sets_is_connected_false(self) -> None:
        stub = CalendarStub()
        stub.connect()
        assert stub.is_connected is True
        stub.disconnect()
        assert stub.is_connected is False

    def test_backend_name_property(self) -> None:
        assert CalendarStub().backend_name == "free_busy_stub"

    def test_test_connection_returns_ok(self) -> None:
        ok, msg = CalendarStub().test_connection()
        assert ok is True
        assert msg is None

    def test_has_connect_method(self) -> None:
        assert callable(CalendarStub().connect)

    def test_has_fetch_events_method(self) -> None:
        assert callable(CalendarStub().fetch_events)

    def test_has_disconnect_method(self) -> None:
        assert callable(CalendarStub().disconnect)

    def test_has_is_connected_property(self) -> None:
        stub = CalendarStub()
        assert hasattr(stub, "is_connected")

    def test_calendar_event_fields_present(self) -> None:
        stub = CalendarStub()
        stub.connect()
        for event in stub.fetch_events():
            assert hasattr(event, "event_id")
            assert hasattr(event, "title")
            assert hasattr(event, "start_time")
            assert hasattr(event, "end_time")


# ---------------------------------------------------------------------------
# AC3 — no live credentials or network calls
# ---------------------------------------------------------------------------


class TestNoNetworkCalls:
    def test_connect_makes_no_network_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connected: list[tuple[str, ...]] = []

        def fake_connect(self: Any, addr: Any) -> None:  # type: ignore[misc]
            connected.append(addr)

        monkeypatch.setattr(socket.socket, "connect", fake_connect)
        stub = CalendarStub()
        stub.connect()
        assert connected == []

    def test_fetch_events_makes_no_network_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connected: list[tuple[str, ...]] = []

        def fake_connect(self: Any, addr: Any) -> None:  # type: ignore[misc]
            connected.append(addr)

        monkeypatch.setattr(socket.socket, "connect", fake_connect)
        stub = CalendarStub()
        stub.connect()
        _ = stub.fetch_events()
        assert connected == []

    def test_get_free_busy_makes_no_network_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connected: list[tuple[str, ...]] = []

        def fake_connect(self: Any, addr: Any) -> None:  # type: ignore[misc]
            connected.append(addr)

        monkeypatch.setattr(socket.socket, "connect", fake_connect)
        stub = CalendarStub()
        stub.connect()
        start, end = _window(0, 8, 18)
        _ = stub.get_free_busy(start, end)
        assert connected == []

    def test_stub_requires_no_credentials(self) -> None:
        """CalendarStub instantiates without any credentials."""
        # Should not raise regardless of environment
        stub = CalendarStub()
        assert stub is not None

    def test_stub_works_in_isolation_with_fixed_anchor(self) -> None:
        """Results are deterministic for a fixed anchor date."""
        stub_a = CalendarStub(anchor=date(2026, 3, 9))
        stub_b = CalendarStub(anchor=date(2026, 3, 9))
        titles_a = sorted(e.title for e in stub_a.all_events)
        titles_b = sorted(e.title for e in stub_b.all_events)
        assert titles_a == titles_b


# ---------------------------------------------------------------------------
# FreeBusyBlock helpers
# ---------------------------------------------------------------------------


class TestFreeBusyBlock:
    def test_overlaps_true_when_event_is_inside_window(self) -> None:
        block = FreeBusyBlock(
            start=_utc(2026, 3, 9, 10),
            end=_utc(2026, 3, 9, 11),
        )
        assert block.overlaps(_utc(2026, 3, 9, 9), _utc(2026, 3, 9, 12))

    def test_overlaps_false_when_event_is_before_window(self) -> None:
        block = FreeBusyBlock(
            start=_utc(2026, 3, 9, 7),
            end=_utc(2026, 3, 9, 8),
        )
        assert not block.overlaps(_utc(2026, 3, 9, 9), _utc(2026, 3, 9, 12))

    def test_overlaps_false_when_event_is_after_window(self) -> None:
        block = FreeBusyBlock(
            start=_utc(2026, 3, 9, 13),
            end=_utc(2026, 3, 9, 14),
        )
        assert not block.overlaps(_utc(2026, 3, 9, 9), _utc(2026, 3, 9, 12))

    def test_overlaps_true_when_event_straddles_window_start(self) -> None:
        block = FreeBusyBlock(
            start=_utc(2026, 3, 9, 8, 30),
            end=_utc(2026, 3, 9, 9, 30),
        )
        assert block.overlaps(_utc(2026, 3, 9, 9), _utc(2026, 3, 9, 12))

    def test_overlaps_true_when_event_straddles_window_end(self) -> None:
        block = FreeBusyBlock(
            start=_utc(2026, 3, 9, 11, 30),
            end=_utc(2026, 3, 9, 12, 30),
        )
        assert block.overlaps(_utc(2026, 3, 9, 9), _utc(2026, 3, 9, 12))

    def test_repr_contains_start_and_end(self) -> None:
        block = FreeBusyBlock(
            start=_utc(2026, 3, 9, 10),
            end=_utc(2026, 3, 9, 11),
            title="Test",
        )
        r = repr(block)
        assert "FreeBusyBlock" in r
        assert "2026-03-09T10:00:00" in r


# Type alias used only for monkeypatching in tests
from typing import Any  # noqa: E402
