"""Tests for the calendar service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rex.calendar_service import CalendarEvent, CalendarService
from rex.event_bus import EventBus


def test_calendar_conflict_detection_and_publish():
    bus = EventBus()
    events = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    bus.subscribe("calendar.created", handler)

    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    existing = CalendarEvent(
        event_id="event-1",
        title="Team sync",
        start_time=start,
        end_time=start + timedelta(hours=1),
        location="Zoom",
    )
    service = CalendarService(bus, mock_events=[existing])

    new_event = CalendarEvent(
        event_id="event-2",
        title="Overlap",
        start_time=start + timedelta(minutes=30),
        end_time=start + timedelta(hours=1, minutes=30),
    )

    conflicts = service.detect_conflicts(new_event)
    assert conflicts == [existing]

    created = service.create_event(
        "Planning",
        start + timedelta(days=1),
        start + timedelta(days=1, hours=1),
    )

    assert created.title == "Planning"
    assert events[0][0] == "calendar.created"
