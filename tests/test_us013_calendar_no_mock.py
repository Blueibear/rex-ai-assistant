"""Tests for US-013: Remove mock calendar data and connect real backend.

Acceptance criteria:
- No hardcoded fake calendar events in CalendarService or calendar_backends/
- Unconfigured calendar raises IntegrationNotConfiguredError
- Configured ICS backend returns real parsed events
- Test covers both paths
- Typecheck passes
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.calendar_service import (
    CalendarEvent,
    CalendarService,
    get_calendar_service,
    set_calendar_service,
)


@pytest.fixture(autouse=True)
def reset_calendar_singleton():
    """Prevent global singleton leaking between tests."""
    yield
    set_calendar_service(None)  # type: ignore[arg-type]


class TestNoHardcodedFakeEvents:
    def test_new_service_with_no_file_returns_empty_list(self, tmp_path):
        """CalendarService with a non-existent path returns [] — no fake events."""
        svc = CalendarService(mock_data_path=tmp_path / "does_not_exist.json")
        svc.connect(user_id="default")
        events = svc.get_all_events(user_id="default")
        assert events == []

    def test_default_mode_no_seed_file_returns_empty(self, monkeypatch, tmp_path):
        """Without any seed file or runtime file, _load_events returns []."""
        svc = CalendarService(mock_data_path=tmp_path / "missing.json")
        events = svc.get_all_events(user_id="default")
        assert events == []

    def test_no_product_sync_event_in_default_state(self, tmp_path):
        """'Product sync' hardcoded event must not appear in any default state."""
        svc = CalendarService(mock_data_path=tmp_path / "empty.json")
        svc.connect(user_id="default")
        titles = [e.title for e in svc.get_all_events(user_id="default")]
        assert "Product sync" not in titles

    def test_no_checkin_event_in_default_state(self, tmp_path):
        """'1:1 check-in' hardcoded event must not appear in any default state."""
        svc = CalendarService(mock_data_path=tmp_path / "empty.json")
        svc.connect(user_id="default")
        titles = [e.title for e in svc.get_all_events(user_id="default")]
        assert "1:1 check-in" not in titles


class TestNotConfiguredPath:
    def test_get_calendar_service_raises_when_no_config(self):
        """get_calendar_service() with empty config raises IntegrationNotConfiguredError."""
        with pytest.raises(IntegrationNotConfiguredError):
            get_calendar_service(config={})

    def test_get_calendar_service_raises_with_stub_backend(self):
        """get_calendar_service() with backend=stub raises IntegrationNotConfiguredError."""
        with pytest.raises(IntegrationNotConfiguredError):
            get_calendar_service(config={"calendar": {"backend": "stub"}})

    def test_get_calendar_service_raises_with_no_ics_source(self):
        """get_calendar_service() with ics backend but empty source raises."""
        with pytest.raises(IntegrationNotConfiguredError):
            get_calendar_service(config={"calendar": {"backend": "ics", "ics": {"source": ""}}})


class TestConfiguredPath:
    def test_service_with_mock_events_returns_them(self):
        """CalendarService(mock_events=[...]) returns exactly those events — not fake data."""
        now = datetime.now(UTC)
        real_event = CalendarEvent(
            event_id="real-001",
            title="Real Meeting",
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2),
        )
        svc = CalendarService(mock_events=[real_event])
        svc.connect(user_id="default")
        events = svc.get_all_events(user_id="default")
        assert len(events) == 1
        assert events[0].title == "Real Meeting"

    def test_service_with_json_file_returns_file_events(self, tmp_path):
        """CalendarService backed by a real JSON file returns those events."""
        import json

        now = datetime.now(UTC)
        cal_file = tmp_path / "calendar.json"
        cal_file.write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "event_id": "file-001",
                            "title": "File Event",
                            "start_time": (now + timedelta(hours=3)).isoformat(),
                            "end_time": (now + timedelta(hours=4)).isoformat(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        svc = CalendarService(mock_data_path=cal_file)
        svc.connect(user_id="default")
        events = svc.get_all_events(user_id="default")
        assert len(events) == 1
        assert events[0].title == "File Event"

    def test_ics_backend_parses_real_ics(self, tmp_path):
        """ICSCalendarBackend with a local .ics file returns parsed events."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        now = datetime.now(UTC)
        dtstart = now.strftime("%Y%m%dT%H%M%SZ")
        dtend = (now + timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ")
        ics_content = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "BEGIN:VEVENT\r\n"
            f"DTSTART:{dtstart}\r\n"
            f"DTEND:{dtend}\r\n"
            "SUMMARY:ICS Real Event\r\n"
            "UID:ics-001@test\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        ics_file = tmp_path / "test.ics"
        ics_file.write_text(ics_content, encoding="utf-8")

        backend = ICSCalendarBackend(source=str(ics_file))
        ok = backend.connect()
        assert ok is True
        events = backend.fetch_events()
        assert len(events) == 1
        assert events[0].title == "ICS Real Event"
