"""Tests for the ICS calendar backend and parser.

All tests are offline and deterministic — no network calls are made.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.calendar_backends.ics_parser import parse_ics
from rex.calendar_service import CalendarEvent

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_ICS = FIXTURE_DIR / "sample_calendar.ics"


# ---------------------------------------------------------------
# ICS parser tests
# ---------------------------------------------------------------


class TestICSParser:
    """Tests for rex.calendar_backends.ics_parser.parse_ics."""

    def test_parse_sample_fixture(self):
        """Parse the sample ICS fixture and verify event count."""
        text = SAMPLE_ICS.read_text(encoding="utf-8")
        events = parse_ics(text)
        assert len(events) == 4

    def test_event_fields(self):
        """Verify parsed fields for a known event."""
        text = SAMPLE_ICS.read_text(encoding="utf-8")
        events = parse_ics(text)
        # Events are sorted by start_time; find Team Standup
        standup = [e for e in events if e.title == "Team Standup"]
        assert len(standup) == 1
        evt = standup[0]
        assert evt.event_id == "evt-ics-001@rex-ai"
        assert evt.start_time == datetime(2026, 2, 20, 14, 0, tzinfo=UTC)
        assert evt.end_time == datetime(2026, 2, 20, 15, 0, tzinfo=UTC)
        assert evt.location == "Zoom"
        assert evt.description == "Daily team standup meeting"
        assert "alice@example.com" in evt.attendees
        assert "bob@example.com" in evt.attendees
        assert evt.all_day is False

    def test_all_day_event(self):
        """Verify all-day event parsing."""
        text = SAMPLE_ICS.read_text(encoding="utf-8")
        events = parse_ics(text)
        holiday = [e for e in events if e.title == "Company Holiday"]
        assert len(holiday) == 1
        evt = holiday[0]
        assert evt.all_day is True
        assert evt.start_time.date() == datetime(2026, 3, 1).date()

    def test_duration_event(self):
        """Verify DURATION-based end time calculation."""
        text = SAMPLE_ICS.read_text(encoding="utf-8")
        events = parse_ics(text)
        review = [e for e in events if e.title == "Architecture Review"]
        assert len(review) == 1
        evt = review[0]
        expected_end = datetime(2026, 2, 25, 10, 30, tzinfo=UTC)
        assert evt.end_time == expected_end

    def test_events_sorted_by_start_time(self):
        """Verify events are returned sorted by start_time."""
        text = SAMPLE_ICS.read_text(encoding="utf-8")
        events = parse_ics(text)
        for i in range(len(events) - 1):
            assert events[i].start_time <= events[i + 1].start_time

    def test_empty_ics(self):
        """Empty ICS content returns no events."""
        events = parse_ics("")
        assert events == []

    def test_ics_no_vevent(self):
        """ICS with no VEVENT blocks returns empty list."""
        text = "BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n"
        events = parse_ics(text)
        assert events == []

    def test_missing_summary_skipped(self):
        """VEVENT without SUMMARY is skipped."""
        text = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:no-summary\n"
            "DTSTART:20260101T120000Z\n"
            "DTEND:20260101T130000Z\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        events = parse_ics(text)
        assert events == []

    def test_missing_dtstart_skipped(self):
        """VEVENT without DTSTART is skipped."""
        text = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:no-dtstart\n"
            "SUMMARY:Missing Start\n"
            "DTEND:20260101T130000Z\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        events = parse_ics(text)
        assert events == []

    def test_line_unfolding(self):
        """Long lines folded with leading whitespace are unfolded."""
        text = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:fold-test\n"
            "DTSTART:20260301T100000Z\n"
            "DTEND:20260301T110000Z\n"
            "SUMMARY:This is a very long \n"
            " event title that wraps\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        events = parse_ics(text)
        assert len(events) == 1
        # RFC 5545: continuation space is consumed, but the trailing space
        # before the fold is preserved, giving "long event".
        assert events[0].title == "This is a very long event title that wraps"

    def test_escaped_characters(self):
        """ICS escaped characters are unescaped in text fields."""
        text = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "UID:escape-test\n"
            "DTSTART:20260301T100000Z\n"
            "DTEND:20260301T110000Z\n"
            "SUMMARY:Meeting\\, Planning\n"
            "DESCRIPTION:Line 1\\nLine 2\\nLine 3\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        events = parse_ics(text)
        assert len(events) == 1
        assert events[0].title == "Meeting, Planning"
        assert "Line 1\nLine 2\nLine 3" == events[0].description

    def test_returns_calendar_event_type(self):
        """Parsed results are CalendarEvent instances."""
        text = SAMPLE_ICS.read_text(encoding="utf-8")
        events = parse_ics(text)
        for evt in events:
            assert isinstance(evt, CalendarEvent)


# ---------------------------------------------------------------
# ICS backend tests
# ---------------------------------------------------------------


class TestICSCalendarBackend:
    """Tests for rex.calendar_backends.ics_backend.ICSCalendarBackend."""

    def test_connect_from_file(self, tmp_path: Path):
        """Backend connects and parses a local ICS file."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        ics_file = tmp_path / "test.ics"
        ics_file.write_text(SAMPLE_ICS.read_text(encoding="utf-8"), encoding="utf-8")

        backend = ICSCalendarBackend(source=str(ics_file))
        assert backend.connect() is True
        assert backend.is_connected is True

        events = backend.fetch_events()
        assert len(events) == 4

    def test_connect_file_not_found(self, tmp_path: Path):
        """Backend returns False when ICS file doesn't exist."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        backend = ICSCalendarBackend(source=str(tmp_path / "nonexistent.ics"))
        assert backend.connect() is False
        assert backend.is_connected is False

    def test_test_connection_file(self, tmp_path: Path):
        """test_connection reports success for a valid file."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        ics_file = tmp_path / "test.ics"
        ics_file.write_text(SAMPLE_ICS.read_text(encoding="utf-8"), encoding="utf-8")

        backend = ICSCalendarBackend(source=str(ics_file))
        ok, message = backend.test_connection()
        assert ok is True
        assert "4 event(s)" in message

    def test_test_connection_missing_file(self, tmp_path: Path):
        """test_connection reports failure for missing file."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        backend = ICSCalendarBackend(source=str(tmp_path / "nope.ics"))
        ok, message = backend.test_connection()
        assert ok is False
        assert "not found" in message

    def test_disconnect(self, tmp_path: Path):
        """disconnect clears connected state."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        ics_file = tmp_path / "test.ics"
        ics_file.write_text(SAMPLE_ICS.read_text(encoding="utf-8"), encoding="utf-8")

        backend = ICSCalendarBackend(source=str(ics_file))
        backend.connect()
        assert backend.is_connected is True
        backend.disconnect()
        assert backend.is_connected is False

    def test_backend_name(self):
        """Backend reports its name correctly."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        backend = ICSCalendarBackend(source="/tmp/test.ics")
        assert backend.backend_name == "ics"

    def test_url_fetch_mocked(self):
        """URL-based source fetches via requests (mocked)."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        ics_text = SAMPLE_ICS.read_text(encoding="utf-8")
        mock_response = MagicMock()
        mock_response.text = ics_text
        mock_response.raise_for_status = MagicMock()

        # Mock both DNS resolution (SSRF check) and the HTTP request itself
        # so the test stays fully offline.
        fake_addrinfo = [(0, 0, 0, "", ("93.184.216.34", 0))]
        with (
            patch("socket.getaddrinfo", return_value=fake_addrinfo),
            patch("requests.get", return_value=mock_response) as mock_get,
        ):
            backend = ICSCalendarBackend(
                source="https://example.com/calendar.ics",
                url_timeout=10,
            )
            assert backend.connect() is True
            events = backend.fetch_events()
            assert len(events) == 4

            mock_get.assert_called_once_with(
                "https://example.com/calendar.ics",
                timeout=10,
                headers={"Accept": "text/calendar, text/plain"},
            )

    def test_http_url_rejected(self):
        """Non-HTTPS URLs are rejected."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        backend = ICSCalendarBackend(source="http://example.com/calendar.ics")
        assert backend.connect() is False

    def test_file_scheme_rejected(self):
        """Explicit file:// scheme is rejected as unsafe URL input."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        backend = ICSCalendarBackend(source="file:///tmp/calendar.ics")
        ok, message = backend.test_connection()
        assert ok is False
        assert "Only HTTPS URLs" in message

    def test_localhost_rejected_for_https(self):
        """HTTPS localhost URL is rejected to reduce SSRF risk."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        backend = ICSCalendarBackend(source="https://localhost/calendar.ics")
        ok, message = backend.test_connection()
        assert ok is False
        assert "localhost" in message

    def test_private_ip_rejected_for_https(self):
        """Resolved private IPs are rejected for HTTPS sources."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        with patch("socket.getaddrinfo", return_value=[(0, 0, 0, "", ("192.168.1.10", 0))]):
            backend = ICSCalendarBackend(source="https://calendar.example.com/feed.ics")
            ok, message = backend.test_connection()
            assert ok is False
            assert "local or reserved" in message

    def test_windows_path_not_treated_as_url(self, tmp_path: Path):
        """Windows-style drive path should be handled as a file path."""
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        backend = ICSCalendarBackend(source="C:\\calendar\\work.ics")
        with patch.object(
            backend, "_read_file", return_value=SAMPLE_ICS.read_text(encoding="utf-8")
        ):
            assert backend.connect() is True


# ---------------------------------------------------------------
# Backend factory tests
# ---------------------------------------------------------------


class TestCalendarBackendFactory:
    """Tests for rex.calendar_backends.factory.create_calendar_backend."""

    def test_default_returns_stub(self):
        """No config returns stub backend."""
        from rex.calendar_backends.factory import create_calendar_backend
        from rex.calendar_backends.stub import StubCalendarBackend

        backend = create_calendar_backend(config={})
        assert isinstance(backend, StubCalendarBackend)

    def test_explicit_stub(self):
        """Explicit stub config returns stub backend."""
        from rex.calendar_backends.factory import create_calendar_backend
        from rex.calendar_backends.stub import StubCalendarBackend

        backend = create_calendar_backend(config={"calendar": {"backend": "stub"}})
        assert isinstance(backend, StubCalendarBackend)

    def test_ics_backend_created(self, tmp_path: Path):
        """ICS config creates ICS backend."""
        from rex.calendar_backends.factory import create_calendar_backend
        from rex.calendar_backends.ics_backend import ICSCalendarBackend

        ics_file = tmp_path / "cal.ics"
        ics_file.write_text(SAMPLE_ICS.read_text(encoding="utf-8"), encoding="utf-8")

        config = {
            "calendar": {
                "backend": "ics",
                "ics": {
                    "source": str(ics_file),
                    "url_timeout": 5,
                },
            }
        }
        backend = create_calendar_backend(config=config)
        assert isinstance(backend, ICSCalendarBackend)

    def test_ics_no_source_falls_back_to_stub(self):
        """ICS config without source falls back to stub."""
        from rex.calendar_backends.factory import create_calendar_backend
        from rex.calendar_backends.stub import StubCalendarBackend

        config = {
            "calendar": {
                "backend": "ics",
                "ics": {"source": ""},
            }
        }
        backend = create_calendar_backend(config=config)
        assert isinstance(backend, StubCalendarBackend)

    def test_get_backend_names(self):
        """get_backend_names returns known backends."""
        from rex.calendar_backends.factory import get_backend_names

        names = get_backend_names()
        assert "stub" in names
        assert "ics" in names


# ---------------------------------------------------------------
# CalendarService with ICS backend integration
# ---------------------------------------------------------------


class TestCalendarServiceICSIntegration:
    """Test that CalendarService works with events loaded from ICS backend."""

    def test_get_calendar_service_with_ics_config(self, tmp_path: Path):
        """get_calendar_service returns service backed by ICS events."""
        from rex.calendar_service import (
            CalendarService,
            get_calendar_service,
            set_calendar_service,
        )

        # Reset global state
        set_calendar_service(None)

        ics_file = tmp_path / "cal.ics"
        ics_file.write_text(SAMPLE_ICS.read_text(encoding="utf-8"), encoding="utf-8")

        config = {
            "calendar": {
                "backend": "ics",
                "ics": {"source": str(ics_file)},
            }
        }

        try:
            svc = get_calendar_service(config=config)
            assert isinstance(svc, CalendarService)
            assert svc.connected is True
            events = svc.get_all_events()
            assert len(events) == 4
        finally:
            set_calendar_service(None)

    def test_ics_events_are_queryable(self, tmp_path: Path):
        """Events from ICS source are queryable via CalendarService methods."""
        from rex.calendar_service import get_calendar_service, set_calendar_service

        set_calendar_service(None)

        ics_file = tmp_path / "cal.ics"
        ics_file.write_text(SAMPLE_ICS.read_text(encoding="utf-8"), encoding="utf-8")

        config = {
            "calendar": {
                "backend": "ics",
                "ics": {"source": str(ics_file)},
            }
        }

        try:
            svc = get_calendar_service(config=config)
            # Query a range that includes Team Standup (2026-02-20 14:00 UTC)
            start = datetime(2026, 2, 20, 0, 0, tzinfo=UTC)
            end = datetime(2026, 2, 21, 0, 0, tzinfo=UTC)
            day_events = svc.get_events(start, end)
            titles = [e.title for e in day_events]
            assert "Team Standup" in titles
            assert "1-on-1 with Manager" in titles
        finally:
            set_calendar_service(None)


# ---------------------------------------------------------------
# Stub backend tests
# ---------------------------------------------------------------


class TestStubCalendarBackend:
    """Tests for rex.calendar_backends.stub.StubCalendarBackend."""

    def test_connect(self):
        """Stub backend connects successfully."""
        from rex.calendar_backends.stub import StubCalendarBackend

        backend = StubCalendarBackend()
        assert backend.connect() is True
        assert backend.is_connected is True

    def test_fetch_events(self):
        """Stub backend returns mock events."""
        from rex.calendar_backends.stub import StubCalendarBackend

        backend = StubCalendarBackend()
        backend.connect()
        events = backend.fetch_events()
        assert len(events) > 0
        for evt in events:
            assert isinstance(evt, CalendarEvent)

    def test_backend_name(self):
        """Stub backend reports its name."""
        from rex.calendar_backends.stub import StubCalendarBackend

        backend = StubCalendarBackend()
        assert backend.backend_name == "stub"

    def test_test_connection(self):
        """Stub backend test_connection always succeeds."""
        from rex.calendar_backends.stub import StubCalendarBackend

        backend = StubCalendarBackend()
        ok, message = backend.test_connection()
        assert ok is True


# ---------------------------------------------------------------
# ICSFeedBackend tests (US-210)
# ---------------------------------------------------------------

UPCOMING_ICS = FIXTURE_DIR / "upcoming_feed.ics"


class TestICSFeedBackend:
    """Tests for rex.integrations.calendar.backends.ics_feed.ICSFeedBackend."""

    def test_get_upcoming_from_file(self):
        """Backend returns future events from a local .ics file."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        backend = ICSFeedBackend(source=str(UPCOMING_ICS))
        events = backend.get_upcoming(days=36500)  # 100 years ahead
        # 3 valid events (malformed one without DTSTART is skipped)
        assert len(events) == 3

    def test_get_upcoming_returns_required_keys(self):
        """Each returned dict has id, title, start, end keys."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        backend = ICSFeedBackend(source=str(UPCOMING_ICS))
        events = backend.get_upcoming(days=36500)
        for event in events:
            assert "id" in event
            assert "title" in event
            assert "start" in event
            assert "end" in event

    def test_get_upcoming_sorted_chronologically(self):
        """Events are returned in chronological order."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        backend = ICSFeedBackend(source=str(UPCOMING_ICS))
        events = backend.get_upcoming(days=36500)
        for i in range(len(events) - 1):
            assert events[i]["start"] <= events[i + 1]["start"]

    def test_get_upcoming_respects_days_window(self):
        """Events beyond the look-ahead window are excluded."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        backend = ICSFeedBackend(source=str(UPCOMING_ICS))
        # Window of 0 days — nothing is upcoming
        events = backend.get_upcoming(days=0)
        assert events == []

    def test_get_upcoming_skips_malformed_vevent(self):
        """VEVENT blocks missing DTSTART are silently skipped."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        # The fixture contains one VEVENT with no DTSTART
        backend = ICSFeedBackend(source=str(UPCOMING_ICS))
        events = backend.get_upcoming(days=36500)
        titles = [e["title"] for e in events]
        assert "No Start Time" not in titles

    def test_get_upcoming_from_inline_ics(self, tmp_path):
        """Backend works with a simple hand-crafted ICS file."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        ics_content = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "BEGIN:VEVENT\r\n"
            "UID:inline-001@test\r\n"
            "DTSTART:20990601T090000Z\r\n"
            "DTEND:20990601T100000Z\r\n"
            "SUMMARY:Inline Event\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        ics_file = tmp_path / "inline.ics"
        ics_file.write_bytes(ics_content.encode())
        backend = ICSFeedBackend(source=str(ics_file))
        events = backend.get_upcoming(days=36500)
        assert len(events) == 1
        assert events[0]["title"] == "Inline Event"
        assert events[0]["id"] == "inline-001@test"

    def test_get_upcoming_utc_normalisation(self, tmp_path):
        """Datetime strings in returned events are UTC ISO-8601 strings."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        ics_content = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "BEGIN:VEVENT\r\n"
            "UID:tz-001@test\r\n"
            "DTSTART:20990615T120000Z\r\n"
            "DTEND:20990615T130000Z\r\n"
            "SUMMARY:UTC Event\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        ics_file = tmp_path / "tz.ics"
        ics_file.write_bytes(ics_content.encode())
        backend = ICSFeedBackend(source=str(ics_file))
        events = backend.get_upcoming(days=36500)
        assert len(events) == 1
        # start and end are timezone-aware UTC ISO strings
        start_dt = datetime.fromisoformat(events[0]["start"])
        assert start_dt.tzinfo is not None
        assert start_dt.tzinfo == UTC

    def test_get_upcoming_empty_calendar(self, tmp_path):
        """Calendar with no VEVENT blocks returns empty list."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        ics_content = b"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n"
        ics_file = tmp_path / "empty.ics"
        ics_file.write_bytes(ics_content)
        backend = ICSFeedBackend(source=str(ics_file))
        assert backend.get_upcoming() == []

    def test_get_upcoming_missing_file_returns_empty(self, tmp_path):
        """Missing file logs an error and returns empty list."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        backend = ICSFeedBackend(source=str(tmp_path / "missing.ics"))
        assert backend.get_upcoming() == []

    def test_get_upcoming_via_http_fetch_injection(self):
        """HTTP source uses injected http_fetch callable (no live network)."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        raw = UPCOMING_ICS.read_bytes()
        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> bytes:
            fetched_urls.append(url)
            return raw

        backend = ICSFeedBackend(
            source="https://calendar.example.com/feed.ics",
            http_fetch=fake_fetch,
        )
        events = backend.get_upcoming(days=36500)
        assert len(events) == 3
        assert fetched_urls == ["https://calendar.example.com/feed.ics"]

    def test_create_event_raises_not_implemented(self):
        """create_event() raises NotImplementedError — backend is read-only."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        backend = ICSFeedBackend(source=str(UPCOMING_ICS))
        with pytest.raises(NotImplementedError):
            backend.create_event(
                "Meeting", "2099-01-01T10:00:00+00:00", "2099-01-01T11:00:00+00:00"
            )

    def test_synthetic_uid_for_event_without_uid(self, tmp_path):
        """VEVENT without UID gets a deterministic synthetic uid."""
        from rex.integrations.calendar.backends.ics_feed import ICSFeedBackend

        ics_content = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "BEGIN:VEVENT\r\n"
            "DTSTART:20990701T100000Z\r\n"
            "DTEND:20990701T110000Z\r\n"
            "SUMMARY:No UID Event\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        ics_file = tmp_path / "nouid.ics"
        ics_file.write_bytes(ics_content.encode())
        backend = ICSFeedBackend(source=str(ics_file))
        events = backend.get_upcoming(days=36500)
        assert len(events) == 1
        assert events[0]["id"].startswith("synth-")
