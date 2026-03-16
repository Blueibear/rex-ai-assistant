"""Unit tests for rex.integrations.calendar_service — stub mode."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rex.integrations.calendar_service import CalendarService
from rex.integrations.models import CalendarEvent


class TestCalendarServiceStub:
    """Tests for CalendarService running in stub mode (calendar_provider='none')."""

    def setup_method(self) -> None:
        self.service = CalendarService(calendar_provider="none")

    # ------------------------------------------------------------------
    # get_events
    # ------------------------------------------------------------------

    def test_get_events_returns_list(self) -> None:
        now = datetime.now(timezone.utc)
        events = self.service.get_events(now, now + timedelta(days=14))
        assert isinstance(events, list)
        assert len(events) > 0
        assert all(isinstance(e, CalendarEvent) for e in events)

    def test_get_events_count_within_two_weeks(self) -> None:
        now = datetime.now(timezone.utc)
        events = self.service.get_events(now, now + timedelta(days=14))
        # Stub has 8 events spanning ~12 days; most should appear in a 2-week window
        assert len(events) >= 5

    def test_get_events_respects_window(self) -> None:
        now = datetime.now(timezone.utc)
        # Request a tiny 1-hour window far in the future — no stub events should match
        far_future = now + timedelta(days=365)
        events = self.service.get_events(far_future, far_future + timedelta(hours=1))
        assert events == []

    def test_get_events_start_less_than_end_required(self) -> None:
        now = datetime.now(timezone.utc)
        # start == end: no events overlap (start < end is False for all)
        events = self.service.get_events(now, now)
        assert events == []

    # ------------------------------------------------------------------
    # create_event
    # ------------------------------------------------------------------

    def test_create_event_returns_calendar_event(self) -> None:
        now = datetime.now(timezone.utc)
        event = self.service.create_event(
            {
                "title": "Test meeting",
                "start": now,
                "end": now + timedelta(hours=1),
            }
        )
        assert isinstance(event, CalendarEvent)
        assert event.title == "Test meeting"
        assert event.source == "stub"

    def test_create_event_with_optional_fields(self) -> None:
        now = datetime.now(timezone.utc)
        event = self.service.create_event(
            {
                "title": "Team lunch",
                "start": now,
                "end": now + timedelta(hours=2),
                "location": "Cafeteria",
                "description": "Monthly team lunch",
                "attendees": ["a@example.com", "b@example.com"],
            }
        )
        assert event.location == "Cafeteria"
        assert event.description == "Monthly team lunch"
        assert event.attendees == ["a@example.com", "b@example.com"]

    def test_create_event_with_string_datetimes(self) -> None:
        now = datetime.now(timezone.utc)
        event = self.service.create_event(
            {
                "title": "ISO datetime event",
                "start": now.isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
            }
        )
        assert isinstance(event, CalendarEvent)
        assert isinstance(event.start, datetime)

    def test_create_event_assigns_id(self) -> None:
        now = datetime.now(timezone.utc)
        event = self.service.create_event({"title": "No ID event", "start": now, "end": now + timedelta(hours=1)})
        assert event.id != ""

    # ------------------------------------------------------------------
    # update_event
    # ------------------------------------------------------------------

    def test_update_event_returns_calendar_event(self) -> None:
        now = datetime.now(timezone.utc)
        event = self.service.update_event(
            "stub-cal-001",
            {"title": "Updated meeting", "start": now, "end": now + timedelta(hours=1)},
        )
        assert isinstance(event, CalendarEvent)
        assert event.id == "stub-cal-001"
        assert event.title == "Updated meeting"

    # ------------------------------------------------------------------
    # delete_event
    # ------------------------------------------------------------------

    def test_delete_event_does_not_raise(self) -> None:
        # Stub mode is a no-op — should not raise
        self.service.delete_event("stub-cal-001")

    # ------------------------------------------------------------------
    # model round-trip
    # ------------------------------------------------------------------

    def test_calendar_event_model_dump_round_trip(self) -> None:
        now = datetime.now(timezone.utc)
        events = self.service.get_events(now, now + timedelta(days=14))
        assert events
        dumped = events[0].model_dump()
        restored = CalendarEvent(**dumped)
        assert restored == events[0]
