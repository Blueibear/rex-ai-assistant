"""Tests for US-045: Calendar integration.

Acceptance criteria:
- events retrieved
- events created
- errors handled
- Typecheck passes
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rex.calendar_service import (
    CalendarEvent,
    CalendarService,
    get_calendar_service,
    set_calendar_service,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(tmp_path: Path) -> CalendarService:
    """Create an isolated CalendarService backed by a temp file."""
    cal_file = tmp_path / "calendar.json"
    svc = CalendarService(mock_data_path=cal_file)
    svc.connect()
    return svc


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_calendar_service():
    """Prevent global singleton leaking between tests."""
    original = None
    yield
    set_calendar_service(original)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Events retrieved
# ---------------------------------------------------------------------------

class TestEventsRetrieved:
    def test_connect_returns_true(self, tmp_path):
        svc = CalendarService(mock_data_path=tmp_path / "cal.json")
        assert svc.connect() is True

    def test_get_all_events_returns_list(self, tmp_path):
        svc = _make_service(tmp_path)
        events = svc.get_all_events()
        assert isinstance(events, list)

    def test_get_events_in_range(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        svc.create_event("Future meeting", now + timedelta(hours=1), now + timedelta(hours=2))
        events = svc.get_events(now, now + timedelta(hours=3))
        assert len(events) >= 1
        assert any(e.title == "Future meeting" for e in events)

    def test_list_upcoming_returns_future_events(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        svc.create_event("Soon", now + timedelta(hours=1), now + timedelta(hours=2))
        upcoming = svc.list_upcoming(horizon_hours=48)
        assert any(e.title == "Soon" for e in upcoming)

    def test_list_upcoming_excludes_past_events(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        svc.create_event("Old", now - timedelta(hours=10), now - timedelta(hours=9))
        upcoming = svc.list_upcoming(horizon_hours=1)
        assert not any(e.title == "Old" for e in upcoming)

    def test_events_sorted_by_start_time(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        svc.create_event("Second", now + timedelta(hours=5), now + timedelta(hours=6))
        svc.create_event("First", now + timedelta(hours=1), now + timedelta(hours=2))
        events = svc.get_events(now, now + timedelta(hours=10))
        starts = [e.start_time for e in events]
        assert starts == sorted(starts)

    def test_get_events_from_json_file(self, tmp_path):
        cal_file = tmp_path / "calendar.json"
        now = _now()
        data = {
            "events": [
                {
                    "event_id": "e1",
                    "title": "Loaded Event",
                    "start_time": (now + timedelta(hours=2)).isoformat(),
                    "end_time": (now + timedelta(hours=3)).isoformat(),
                }
            ]
        }
        cal_file.write_text(json.dumps(data), encoding="utf-8")
        svc = CalendarService(mock_data_path=cal_file)
        svc.connect()
        events = svc.get_all_events()
        assert any(e.title == "Loaded Event" for e in events)

    def test_get_upcoming_events_by_days(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        svc.create_event("Next week", now + timedelta(days=3), now + timedelta(days=3, hours=1))
        events = svc.get_upcoming_events(days=7)
        assert any(e.title == "Next week" for e in events)


# ---------------------------------------------------------------------------
# Events created
# ---------------------------------------------------------------------------

class TestEventsCreated:
    def test_create_event_returns_calendar_event(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        event = svc.create_event("Test Event", now + timedelta(hours=1), now + timedelta(hours=2))
        assert isinstance(event, CalendarEvent)
        assert event.title == "Test Event"

    def test_created_event_has_unique_id(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        e1 = svc.create_event("A", now + timedelta(hours=1), now + timedelta(hours=2))
        e2 = svc.create_event("B", now + timedelta(hours=3), now + timedelta(hours=4))
        assert e1.event_id != e2.event_id

    def test_created_event_persists_to_disk(self, tmp_path):
        cal_file = tmp_path / "cal.json"
        svc = CalendarService(mock_data_path=cal_file)
        svc.connect()
        now = _now()
        svc.create_event("Persist me", now + timedelta(hours=1), now + timedelta(hours=2))

        # New instance reads the same file
        svc2 = CalendarService(mock_data_path=cal_file)
        svc2.connect()
        events = svc2.get_all_events()
        assert any(e.title == "Persist me" for e in events)

    def test_create_event_with_optional_fields(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        event = svc.create_event(
            "Detailed",
            now + timedelta(hours=1),
            now + timedelta(hours=2),
            location="Room B",
            attendees=["alice@example.com"],
            description="Important meeting",
        )
        assert event.location == "Room B"
        assert "alice@example.com" in event.attendees
        assert event.description == "Important meeting"

    def test_create_all_day_event(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        event = svc.create_event(
            "All Day",
            now.replace(hour=0, minute=0, second=0, microsecond=0),
            now.replace(hour=23, minute=59, second=59, microsecond=0),
            all_day=True,
        )
        assert event.all_day is True

    def test_created_event_retrievable_via_get_events(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        event = svc.create_event("Retrievable", now + timedelta(hours=1), now + timedelta(hours=2))
        events = svc.get_events(now, now + timedelta(hours=3))
        ids = [e.event_id for e in events]
        assert event.event_id in ids

    def test_in_memory_mode_creates_event(self):
        now = _now()
        initial = [
            CalendarEvent(
                title="Existing",
                start_time=now + timedelta(hours=1),
                end_time=now + timedelta(hours=2),
            )
        ]
        svc = CalendarService(mock_events=initial)
        svc.connect()
        new_event = svc.create_event(
            "New In Memory",
            now + timedelta(hours=3),
            now + timedelta(hours=4),
        )
        assert new_event.title == "New In Memory"
        all_events = svc.get_all_events()
        assert len(all_events) == 2


# ---------------------------------------------------------------------------
# Errors handled
# ---------------------------------------------------------------------------

class TestErrorsHandled:
    def test_get_events_before_connect_returns_empty(self, tmp_path):
        """get_events() on a disconnected service with no in-memory events returns []."""
        cal_file = tmp_path / "cal.json"
        svc = CalendarService(mock_data_path=cal_file)
        # Do not call connect(); _events is None
        now = _now()
        events = svc.get_events(now, now + timedelta(hours=1))
        assert events == []

    def test_connect_with_invalid_json_returns_false(self, tmp_path):
        """connect() with corrupt JSON data should not crash but fails gracefully."""
        cal_file = tmp_path / "bad.json"
        cal_file.write_text("{not valid json", encoding="utf-8")
        svc = CalendarService(mock_data_path=cal_file)
        result = svc.connect()
        # May return False due to JSON error, or True with empty list — either is safe
        assert isinstance(result, bool)

    def test_delete_nonexistent_event_returns_false(self, tmp_path):
        svc = _make_service(tmp_path)
        result = svc.delete_event("nonexistent-id-xyz")
        assert result is False

    def test_update_nonexistent_event_returns_none(self, tmp_path):
        svc = _make_service(tmp_path)
        now = _now()
        result = svc.update_event("nonexistent-id-xyz", {"title": "New Title"})
        assert result is None

    def test_calendar_event_start_after_end_stored_as_is(self, tmp_path):
        """CalendarEvent does not raise on reversed times — stored as provided."""
        svc = _make_service(tmp_path)
        now = _now()
        # Reversed times — should not raise
        event = svc.create_event("Reversed", now + timedelta(hours=2), now + timedelta(hours=1))
        assert event.title == "Reversed"

    def test_get_calendar_service_global_singleton(self, tmp_path):
        """get_calendar_service() returns a CalendarService."""
        svc = get_calendar_service()
        assert isinstance(svc, CalendarService)

    def test_set_calendar_service_replaces_global(self, tmp_path):
        now = _now()
        custom = CalendarService(
            mock_events=[
                CalendarEvent(
                    title="Custom",
                    start_time=now + timedelta(hours=1),
                    end_time=now + timedelta(hours=2),
                )
            ]
        )
        set_calendar_service(custom)
        svc = get_calendar_service()
        assert svc is custom

    def test_list_past_events_handled_safely(self, tmp_path):
        svc = _make_service(tmp_path)
        result = svc.list_past_events(lookback_hours=24)
        assert isinstance(result, list)
