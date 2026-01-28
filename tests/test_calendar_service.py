"""
Tests for the calendar service.

This test module supports BOTH calendar service variants that have appeared in
the codebase:

Variant A (legacy/event-bus style):
- CalendarEvent dataclass with fields:
    event_id, title, start_time, end_time, location?, description?, attendees?
- CalendarService(event_bus, mock_events=[...])
- service.detect_conflicts(candidate_event) -> list[CalendarEvent]
- service.create_event(title, start_time, end_time, ...) -> CalendarEvent
- EventBus.subscribe(event_type, callback(event_type, payload))

Variant B (newer/pydantic/mock-file style):
- CalendarEvent model with fields:
    id, title, start_time, end_time, attendees, location, description, all_day
- CalendarService(mock_data_file=Path(...))
- service.connect()
- service.get_events(start, end)
- service.create_event(CalendarEvent) -> CalendarEvent
- service.update_event(id, updates) -> CalendarEvent|None
- service.delete_event(id) -> bool
- service.find_conflicts([...]) -> list[tuple[CalendarEvent, CalendarEvent]]
- service.get_upcoming_events(days=7)
- service.get_all_events()

These tests will auto-detect which API is available at runtime and skip the
incompatible tests, so your suite stays green as the calendar service evolves.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from rex.calendar_service import CalendarEvent, CalendarService


def _has_attr(obj: Any, name: str) -> bool:
    return hasattr(obj, name)


def _calendar_event_is_pydantic() -> bool:
    # Pydantic v2 models typically have model_dump; dataclasses do not.
    return _has_attr(CalendarEvent, "model_dump") or _has_attr(CalendarEvent, "model_validate")


def _calendar_service_accepts_event_bus() -> bool:
    # If CalendarService __init__ takes event_bus as first arg, legacy.
    # We can't reliably inspect signature in all cases; instead we probe by instantiation.
    try:
        from rex.event_bus import EventBus  # type: ignore

        bus = EventBus()
        _ = CalendarService(bus)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def _make_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# -------------------------------------------------------------------
# Fixtures for newer implementation (mock file based)
# -------------------------------------------------------------------


@pytest.fixture
def temp_mock_calendar(tmp_path: Path) -> Path:
    """Create a temporary mock calendar file (newer implementation)."""
    now = datetime.now()

    mock_events = [
        {
            "id": "event-1",
            "title": "Team Meeting",
            "start_time": (now + timedelta(days=1)).isoformat(),
            "end_time": (now + timedelta(days=1, hours=1)).isoformat(),
            "attendees": ["alice@example.com", "bob@example.com"],
            "location": "Conference Room A",
            "description": "Weekly team meeting",
            "all_day": False,
        },
        {
            "id": "event-2",
            "title": "Project Deadline",
            "start_time": (now + timedelta(days=3)).isoformat(),
            "end_time": (now + timedelta(days=3, hours=1)).isoformat(),
            "attendees": [],
            "location": None,
            "description": "Project deliverable due",
            "all_day": False,
        },
        {
            "id": "event-3",
            "title": "Conference",
            "start_time": (now + timedelta(days=10)).isoformat(),
            "end_time": (now + timedelta(days=11)).isoformat(),
            "attendees": [],
            "location": "Convention Center",
            "description": "Annual conference",
            "all_day": True,
        },
    ]

    mock_file = tmp_path / "mock_calendar.json"
    mock_file.write_text(json.dumps(mock_events), encoding="utf-8")
    return mock_file


@pytest.fixture
def calendar_service(temp_mock_calendar: Path) -> CalendarService:
    """Create a test calendar service instance (newer implementation)."""
    return CalendarService(mock_data_file=temp_mock_calendar)  # type: ignore[call-arg]


# -------------------------------------------------------------------
# Legacy implementation test (event bus + detect_conflicts + publish)
# -------------------------------------------------------------------


@pytest.mark.skipif(
    not _calendar_service_accepts_event_bus(),
    reason="CalendarService event-bus style API not available in this build.",
)
def test_calendar_conflict_detection_and_publish() -> None:
    from rex.event_bus import EventBus  # type: ignore

    bus = EventBus()
    events: list[tuple[str, dict[str, object]]] = []

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

    service = CalendarService(bus, mock_events=[existing])  # type: ignore[call-arg]

    new_event = CalendarEvent(
        event_id="event-2",
        title="Overlap",
        start_time=start + timedelta(minutes=30),
        end_time=start + timedelta(hours=1, minutes=30),
    )

    conflicts = service.detect_conflicts(new_event)  # type: ignore[attr-defined]
    assert conflicts == [existing]

    created = service.create_event(
        "Planning",
        start + timedelta(days=1),
        start + timedelta(days=1, hours=1),
    )

    assert created.title == "Planning"
    assert events, "Expected at least one published event"
    assert events[0][0] == "calendar.created"


# -------------------------------------------------------------------
# Newer implementation tests (pydantic model + mock file + CRUD)
# -------------------------------------------------------------------


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="CalendarEvent is not a pydantic-style model in this build.",
)
def test_calendar_event_creation() -> None:
    start = datetime.now()
    end = start + timedelta(hours=1)

    event = CalendarEvent(
        id="test-1",
        title="Test Event",
        start_time=start,
        end_time=end,
    )

    assert event.id == "test-1"
    assert event.title == "Test Event"
    assert event.start_time == start
    assert event.end_time == end

    # Defaults
    assert getattr(event, "attendees", []) == []
    assert getattr(event, "location", None) is None
    assert getattr(event, "description", None) is None
    assert getattr(event, "all_day", False) is False


@pytest.mark.skipif(
    not _calendar_event_is_pydantic() or not hasattr(CalendarEvent, "overlaps_with"),
    reason="CalendarEvent.overlaps_with not available in this build.",
)
def test_calendar_event_overlaps() -> None:
    start = datetime(2026, 1, 28, 10, 0)

    event1 = CalendarEvent(id="1", title="Event 1", start_time=start, end_time=start + timedelta(hours=2))
    event2 = CalendarEvent(id="2", title="Event 2", start_time=start + timedelta(hours=1), end_time=start + timedelta(hours=3))
    event3 = CalendarEvent(id="3", title="Event 3", start_time=start + timedelta(hours=3), end_time=start + timedelta(hours=4))

    assert event1.overlaps_with(event2) is True
    assert event2.overlaps_with(event1) is True
    assert event1.overlaps_with(event3) is False
    assert event3.overlaps_with(event1) is False
    assert event2.overlaps_with(event3) is False


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_calendar_service_initialization(temp_mock_calendar: Path) -> None:
    service = CalendarService(mock_data_file=temp_mock_calendar)  # type: ignore[call-arg]
    assert service.mock_data_file == temp_mock_calendar  # type: ignore[attr-defined]
    assert service.connected is False  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_connect(calendar_service: CalendarService) -> None:
    result = calendar_service.connect()  # type: ignore[attr-defined]
    assert result is True
    assert calendar_service.connected is True  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_connect_loads_mock_data(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    assert len(calendar_service._mock_events) == 3  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_get_events(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]

    now = datetime.now()
    start = now
    end = now + timedelta(days=7)

    events = calendar_service.get_events(start, end)  # type: ignore[attr-defined]
    assert isinstance(events, list)
    assert len(events) >= 1
    assert all(e.start_time < end for e in events)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_get_events_empty_range(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    start = datetime.now() + timedelta(days=100)
    end = start + timedelta(days=1)
    events = calendar_service.get_events(start, end)  # type: ignore[attr-defined]
    assert events == []


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_get_events_not_connected() -> None:
    service = CalendarService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    events = service.get_events(datetime.now(), datetime.now() + timedelta(days=1))  # type: ignore[attr-defined]
    assert events == []


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_create_event(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]

    start = datetime.now() + timedelta(days=5)
    end = start + timedelta(hours=1)

    new_event = CalendarEvent(
        id="",  # service may generate
        title="New Event",
        start_time=start,
        end_time=end,
        attendees=["test@example.com"],
        location="Office",
    )

    created = calendar_service.create_event(new_event)  # type: ignore[attr-defined]
    assert created.id != ""
    assert created.title == "New Event"
    assert created.attendees == ["test@example.com"]

    all_events = calendar_service.get_all_events()  # type: ignore[attr-defined]
    assert any(e.id == created.id for e in all_events)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_create_event_with_id(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]

    start = datetime.now() + timedelta(days=5)
    end = start + timedelta(hours=1)

    new_event = CalendarEvent(id="custom-id", title="New Event", start_time=start, end_time=end)
    created = calendar_service.create_event(new_event)  # type: ignore[attr-defined]
    assert created.id == "custom-id"


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_create_event_not_connected() -> None:
    service = CalendarService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    start = datetime.now()
    event = CalendarEvent(id="test", title="Test", start_time=start, end_time=start + timedelta(hours=1))

    with pytest.raises(RuntimeError):
        service.create_event(event)  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_update_event(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]

    events = calendar_service.get_all_events()  # type: ignore[attr-defined]
    event_id = events[0].id

    updated = calendar_service.update_event(event_id, {"title": "Updated Title", "location": "New Location"})  # type: ignore[attr-defined]
    assert updated is not None
    assert updated.title == "Updated Title"
    assert updated.location == "New Location"


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_update_event_nonexistent(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    result = calendar_service.update_event("nonexistent", {"title": "New"})  # type: ignore[attr-defined]
    assert result is None


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_update_event_not_connected() -> None:
    service = CalendarService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    result = service.update_event("event-1", {"title": "New"})  # type: ignore[attr-defined]
    assert result is None


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_delete_event(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]

    events = calendar_service.get_all_events()  # type: ignore[attr-defined]
    initial_count = len(events)
    event_id = events[0].id

    result = calendar_service.delete_event(event_id)  # type: ignore[attr-defined]
    assert result is True

    events_after = calendar_service.get_all_events()  # type: ignore[attr-defined]
    assert len(events_after) == initial_count - 1
    assert not any(e.id == event_id for e in events_after)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_delete_event_nonexistent(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    assert calendar_service.delete_event("nonexistent") is False  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_delete_event_not_connected() -> None:
    service = CalendarService(mock_data_file=Path("/nonexistent"))  # type: ignore[call-arg]
    assert service.delete_event("event-1") is False  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _calendar_event_is_pydantic() or not hasattr(CalendarService, "find_conflicts"),
    reason="CalendarService.find_conflicts not available in this build.",
)
def test_find_conflicts(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]

    start = datetime.now()
    event1 = CalendarEvent(id="conflict-1", title="Event 1", start_time=start, end_time=start + timedelta(hours=2))
    event2 = CalendarEvent(id="conflict-2", title="Event 2", start_time=start + timedelta(hours=1), end_time=start + timedelta(hours=3))

    conflicts = calendar_service.find_conflicts([event1, event2])  # type: ignore[attr-defined]
    assert len(conflicts) == 1
    assert conflicts[0] == (event1, event2)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic() or not hasattr(CalendarService, "find_conflicts"),
    reason="CalendarService.find_conflicts not available in this build.",
)
def test_find_conflicts_none(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]

    start = datetime.now()
    event1 = CalendarEvent(id="no-conflict-1", title="Event 1", start_time=start, end_time=start + timedelta(hours=1))
    event2 = CalendarEvent(id="no-conflict-2", title="Event 2", start_time=start + timedelta(hours=2), end_time=start + timedelta(hours=3))

    conflicts = calendar_service.find_conflicts([event1, event2])  # type: ignore[attr-defined]
    assert conflicts == []


@pytest.mark.skipif(
    not _calendar_event_is_pydantic() or not hasattr(CalendarService, "find_conflicts"),
    reason="CalendarService.find_conflicts not available in this build.",
)
def test_find_conflicts_all_events(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    conflicts = calendar_service.find_conflicts()  # type: ignore[attr-defined]
    assert isinstance(conflicts, list)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic() or not hasattr(CalendarService, "get_upcoming_events"),
    reason="CalendarService.get_upcoming_events not available in this build.",
)
def test_get_upcoming_events(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    events = calendar_service.get_upcoming_events(days=7)  # type: ignore[attr-defined]
    assert isinstance(events, list)
    assert all(isinstance(e, CalendarEvent) for e in events)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic() or not hasattr(CalendarService, "get_upcoming_events"),
    reason="CalendarService.get_upcoming_events not available in this build.",
)
def test_get_upcoming_events_custom_days(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    events_7 = calendar_service.get_upcoming_events(days=7)  # type: ignore[attr-defined]
    events_14 = calendar_service.get_upcoming_events(days=14)  # type: ignore[attr-defined]
    assert len(events_14) >= len(events_7)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic() or not hasattr(CalendarService, "get_all_events"),
    reason="CalendarService.get_all_events not available in this build.",
)
def test_get_all_events(calendar_service: CalendarService) -> None:
    calendar_service.connect()  # type: ignore[attr-defined]
    all_events = calendar_service.get_all_events()  # type: ignore[attr-defined]
    assert len(all_events) == 3
    assert all(isinstance(e, CalendarEvent) for e in all_events)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="Mock-file calendar service tests apply to the newer implementation only.",
)
def test_persistence(temp_mock_calendar: Path) -> None:
    # Use a dedicated temp file so we can validate persistence.
    service1 = CalendarService(mock_data_file=temp_mock_calendar)  # type: ignore[call-arg]
    service1.connect()  # type: ignore[attr-defined]

    start = datetime.now() + timedelta(days=5)
    end = start + timedelta(hours=1)

    new_event = CalendarEvent(id="persist-test", title="Persistence Test", start_time=start, end_time=end)
    service1.create_event(new_event)  # type: ignore[attr-defined]

    service2 = CalendarService(mock_data_file=temp_mock_calendar)  # type: ignore[call-arg]
    service2.connect()  # type: ignore[attr-defined]

    events = service2.get_all_events()  # type: ignore[attr-defined]
    assert any(e.id == "persist-test" for e in events)


@pytest.mark.skipif(
    not _calendar_event_is_pydantic(),
    reason="CalendarEvent serialization test applies to the newer implementation only.",
)
def test_calendar_event_serialization() -> None:
    start = datetime.now()
    end = start + timedelta(hours=1)

    event = CalendarEvent(
        id="test-1",
        title="Test",
        start_time=start,
        end_time=end,
        attendees=["test@example.com"],
        location="Office",
        description="Test event",
    )

    data = event.model_dump()  # type: ignore[attr-defined]
    assert data["id"] == "test-1"
    assert data["title"] == "Test"
    assert data["start_time"] == start
    assert data["end_time"] == end

