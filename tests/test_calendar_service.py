"""Tests for calendar service module."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rex.calendar_service import CalendarEvent, CalendarService


@pytest.fixture
def temp_mock_calendar(tmp_path):
    """Create a temporary mock calendar file."""
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
            "all_day": False
        },
        {
            "id": "event-2",
            "title": "Project Deadline",
            "start_time": (now + timedelta(days=3)).isoformat(),
            "end_time": (now + timedelta(days=3, hours=1)).isoformat(),
            "attendees": [],
            "location": None,
            "description": "Project deliverable due",
            "all_day": False
        },
        {
            "id": "event-3",
            "title": "Conference",
            "start_time": (now + timedelta(days=10)).isoformat(),
            "end_time": (now + timedelta(days=11)).isoformat(),
            "attendees": [],
            "location": "Convention Center",
            "description": "Annual conference",
            "all_day": True
        }
    ]

    mock_file = tmp_path / "mock_calendar.json"
    with open(mock_file, 'w') as f:
        json.dump(mock_events, f)

    return mock_file


@pytest.fixture
def calendar_service(temp_mock_calendar):
    """Create a test calendar service instance."""
    return CalendarService(mock_data_file=temp_mock_calendar)


def test_calendar_event_creation():
    """Test creating a CalendarEvent."""
    start = datetime.now()
    end = start + timedelta(hours=1)

    event = CalendarEvent(
        id="test-1",
        title="Test Event",
        start_time=start,
        end_time=end
    )

    assert event.id == "test-1"
    assert event.title == "Test Event"
    assert event.start_time == start
    assert event.end_time == end
    assert event.attendees == []
    assert event.location is None
    assert event.description is None
    assert event.all_day is False


def test_calendar_event_overlaps():
    """Test event overlap detection."""
    start = datetime(2026, 1, 28, 10, 0)

    event1 = CalendarEvent(
        id="1",
        title="Event 1",
        start_time=start,
        end_time=start + timedelta(hours=2)
    )

    event2 = CalendarEvent(
        id="2",
        title="Event 2",
        start_time=start + timedelta(hours=1),
        end_time=start + timedelta(hours=3)
    )

    event3 = CalendarEvent(
        id="3",
        title="Event 3",
        start_time=start + timedelta(hours=3),
        end_time=start + timedelta(hours=4)
    )

    # event1 and event2 overlap
    assert event1.overlaps_with(event2) is True
    assert event2.overlaps_with(event1) is True

    # event1 and event3 don't overlap
    assert event1.overlaps_with(event3) is False
    assert event3.overlaps_with(event1) is False

    # event2 and event3 don't overlap (end time = start time is not overlap)
    assert event2.overlaps_with(event3) is False


def test_calendar_service_initialization(temp_mock_calendar):
    """Test calendar service initializes correctly."""
    service = CalendarService(mock_data_file=temp_mock_calendar)
    assert service.mock_data_file == temp_mock_calendar
    assert service.connected is False


def test_connect(calendar_service):
    """Test connecting to calendar service."""
    result = calendar_service.connect()
    assert result is True
    assert calendar_service.connected is True


def test_connect_loads_mock_data(calendar_service):
    """Test that connect loads mock data."""
    calendar_service.connect()
    assert len(calendar_service._mock_events) == 3


def test_get_events(calendar_service):
    """Test getting events in a time range."""
    calendar_service.connect()

    now = datetime.now()
    start = now
    end = now + timedelta(days=7)

    events = calendar_service.get_events(start, end)

    # Should get events within the next 7 days
    assert len(events) >= 1
    assert all(event.start_time < end for event in events)


def test_get_events_empty_range(calendar_service):
    """Test getting events with no events in range."""
    calendar_service.connect()

    # Far future range with no events
    start = datetime.now() + timedelta(days=100)
    end = start + timedelta(days=1)

    events = calendar_service.get_events(start, end)
    assert len(events) == 0


def test_get_events_not_connected():
    """Test getting events when not connected."""
    service = CalendarService(mock_data_file=Path("/nonexistent"))
    events = service.get_events(datetime.now(), datetime.now() + timedelta(days=1))
    assert events == []


def test_create_event(calendar_service):
    """Test creating a new event."""
    calendar_service.connect()

    start = datetime.now() + timedelta(days=5)
    end = start + timedelta(hours=1)

    new_event = CalendarEvent(
        id="",  # Will be generated
        title="New Event",
        start_time=start,
        end_time=end,
        attendees=["test@example.com"],
        location="Office"
    )

    created = calendar_service.create_event(new_event)

    assert created.id != ""
    assert created.title == "New Event"
    assert created.attendees == ["test@example.com"]

    # Verify it's in the service
    all_events = calendar_service.get_all_events()
    assert any(e.id == created.id for e in all_events)


def test_create_event_with_id(calendar_service):
    """Test creating an event with specified ID."""
    calendar_service.connect()

    start = datetime.now() + timedelta(days=5)
    end = start + timedelta(hours=1)

    new_event = CalendarEvent(
        id="custom-id",
        title="New Event",
        start_time=start,
        end_time=end
    )

    created = calendar_service.create_event(new_event)
    assert created.id == "custom-id"


def test_create_event_not_connected():
    """Test creating event when not connected."""
    service = CalendarService(mock_data_file=Path("/nonexistent"))

    start = datetime.now()
    event = CalendarEvent(
        id="test",
        title="Test",
        start_time=start,
        end_time=start + timedelta(hours=1)
    )

    with pytest.raises(RuntimeError):
        service.create_event(event)


def test_update_event(calendar_service):
    """Test updating an event."""
    calendar_service.connect()

    # Get an existing event
    events = calendar_service.get_all_events()
    event_id = events[0].id

    # Update it
    updates = {"title": "Updated Title", "location": "New Location"}
    updated = calendar_service.update_event(event_id, updates)

    assert updated is not None
    assert updated.title == "Updated Title"
    assert updated.location == "New Location"


def test_update_event_nonexistent(calendar_service):
    """Test updating a non-existent event."""
    calendar_service.connect()
    result = calendar_service.update_event("nonexistent", {"title": "New"})
    assert result is None


def test_update_event_not_connected():
    """Test updating event when not connected."""
    service = CalendarService(mock_data_file=Path("/nonexistent"))
    result = service.update_event("event-1", {"title": "New"})
    assert result is None


def test_delete_event(calendar_service):
    """Test deleting an event."""
    calendar_service.connect()

    # Get an existing event
    events = calendar_service.get_all_events()
    initial_count = len(events)
    event_id = events[0].id

    # Delete it
    result = calendar_service.delete_event(event_id)
    assert result is True

    # Verify it's gone
    events_after = calendar_service.get_all_events()
    assert len(events_after) == initial_count - 1
    assert not any(e.id == event_id for e in events_after)


def test_delete_event_nonexistent(calendar_service):
    """Test deleting a non-existent event."""
    calendar_service.connect()
    result = calendar_service.delete_event("nonexistent")
    assert result is False


def test_delete_event_not_connected():
    """Test deleting event when not connected."""
    service = CalendarService(mock_data_file=Path("/nonexistent"))
    result = service.delete_event("event-1")
    assert result is False


def test_find_conflicts(calendar_service):
    """Test finding conflicting events."""
    calendar_service.connect()

    start = datetime.now()

    # Create overlapping events
    event1 = CalendarEvent(
        id="conflict-1",
        title="Event 1",
        start_time=start,
        end_time=start + timedelta(hours=2)
    )

    event2 = CalendarEvent(
        id="conflict-2",
        title="Event 2",
        start_time=start + timedelta(hours=1),
        end_time=start + timedelta(hours=3)
    )

    conflicts = calendar_service.find_conflicts([event1, event2])

    assert len(conflicts) == 1
    assert conflicts[0] == (event1, event2)


def test_find_conflicts_none(calendar_service):
    """Test finding conflicts when there are none."""
    calendar_service.connect()

    start = datetime.now()

    # Create non-overlapping events
    event1 = CalendarEvent(
        id="no-conflict-1",
        title="Event 1",
        start_time=start,
        end_time=start + timedelta(hours=1)
    )

    event2 = CalendarEvent(
        id="no-conflict-2",
        title="Event 2",
        start_time=start + timedelta(hours=2),
        end_time=start + timedelta(hours=3)
    )

    conflicts = calendar_service.find_conflicts([event1, event2])
    assert len(conflicts) == 0


def test_find_conflicts_all_events(calendar_service):
    """Test finding conflicts in all events."""
    calendar_service.connect()

    # By default, mock events shouldn't conflict
    conflicts = calendar_service.find_conflicts()
    # We don't know if mock events conflict, just test it returns a list
    assert isinstance(conflicts, list)


def test_get_upcoming_events(calendar_service):
    """Test getting upcoming events."""
    calendar_service.connect()

    events = calendar_service.get_upcoming_events(days=7)

    # Should get events in the next 7 days
    assert isinstance(events, list)
    assert all(isinstance(e, CalendarEvent) for e in events)


def test_get_upcoming_events_custom_days(calendar_service):
    """Test getting upcoming events with custom days."""
    calendar_service.connect()

    events_7 = calendar_service.get_upcoming_events(days=7)
    events_14 = calendar_service.get_upcoming_events(days=14)

    # More days should return same or more events
    assert len(events_14) >= len(events_7)


def test_get_all_events(calendar_service):
    """Test getting all events."""
    calendar_service.connect()
    all_events = calendar_service.get_all_events()

    assert len(all_events) == 3
    assert all(isinstance(event, CalendarEvent) for event in all_events)


def test_persistence(calendar_service, temp_mock_calendar):
    """Test event persistence."""
    calendar_service.connect()

    start = datetime.now() + timedelta(days=5)
    end = start + timedelta(hours=1)

    # Create event
    new_event = CalendarEvent(
        id="persist-test",
        title="Persistence Test",
        start_time=start,
        end_time=end
    )
    calendar_service.create_event(new_event)

    # Create new service instance
    service2 = CalendarService(mock_data_file=temp_mock_calendar)
    service2.connect()

    # Event should be loaded
    events = service2.get_all_events()
    assert any(e.id == "persist-test" for e in events)


def test_calendar_event_serialization():
    """Test CalendarEvent JSON serialization."""
    start = datetime.now()
    end = start + timedelta(hours=1)

    event = CalendarEvent(
        id="test-1",
        title="Test",
        start_time=start,
        end_time=end,
        attendees=["test@example.com"],
        location="Office",
        description="Test event"
    )

    data = event.model_dump()
    assert data['id'] == "test-1"
    assert data['title'] == "Test"
    assert data['start_time'] == start
    assert data['end_time'] == end
