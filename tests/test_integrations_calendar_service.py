"""Unit tests for rex.integrations.calendar_service.

When calendar_provider='none' (not configured), get_events() returns an empty
list so the GUI can show an empty-state prompt instead of fake data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rex.integrations.calendar_service import CalendarService
from rex.integrations.models import CalendarEvent


class TestCalendarServiceStub:
    """Tests for CalendarService running in 'none' (unconfigured) mode."""

    def setup_method(self) -> None:
        self.service = CalendarService(calendar_provider="none")

    # ------------------------------------------------------------------
    # get_events — returns empty when not configured
    # ------------------------------------------------------------------

    def test_get_events_returns_list(self) -> None:
        now = datetime.now(UTC)
        events = self.service.get_events(now, now + timedelta(days=14))
        assert isinstance(events, list)

    def test_get_events_empty_when_not_configured(self) -> None:
        """US-315: provider='none' must return [] not stub data."""
        now = datetime.now(UTC)
        events = self.service.get_events(now, now + timedelta(days=14))
        assert events == []

    def test_get_events_respects_window(self) -> None:
        now = datetime.now(UTC)
        far_future = now + timedelta(days=365)
        events = self.service.get_events(far_future, far_future + timedelta(hours=1))
        assert events == []

    def test_get_events_start_less_than_end_required(self) -> None:
        now = datetime.now(UTC)
        events = self.service.get_events(now, now)
        assert events == []

    # ------------------------------------------------------------------
    # create_event
    # ------------------------------------------------------------------

    def test_create_event_returns_calendar_event(self) -> None:
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
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
        now = datetime.now(UTC)
        event = self.service.create_event(
            {"title": "No ID event", "start": now, "end": now + timedelta(hours=1)}
        )
        assert event.id != ""

    # ------------------------------------------------------------------
    # update_event
    # ------------------------------------------------------------------

    def test_update_event_returns_calendar_event(self) -> None:
        now = datetime.now(UTC)
        event = self.service.update_event(
            "cal-001",
            {"title": "Updated meeting", "start": now, "end": now + timedelta(hours=1)},
        )
        assert isinstance(event, CalendarEvent)

    # ------------------------------------------------------------------
    # delete_event
    # ------------------------------------------------------------------

    def test_delete_event_does_not_raise(self) -> None:
        self.service.delete_event("cal-001")

    # ------------------------------------------------------------------
    # model round-trip (using create_event to obtain an instance)
    # ------------------------------------------------------------------

    def test_calendar_event_model_dump_round_trip(self) -> None:
        now = datetime.now(UTC)
        event = self.service.create_event(
            {"title": "Round trip", "start": now, "end": now + timedelta(hours=1)}
        )
        dumped = event.model_dump()
        restored = CalendarEvent(**dumped)
        assert restored == event
