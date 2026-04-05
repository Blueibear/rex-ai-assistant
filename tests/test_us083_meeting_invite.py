"""Tests for US-083: Meeting invite scaffold.

Acceptance criteria:
- MeetingInvite data structure contains: title, attendees list, start time,
  end time, agenda
- Rex can populate all invite fields from a natural language description
- completed invite is displayed to the user for review before any action
- stub send logs the invite and returns success without calling any real
  calendar API
- Typecheck passes
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime

import pytest

from rex.calendar_backends.meeting_invite import (
    MeetingInvite,
    format_invite_for_review,
    parse_invite_from_text,
    stub_send_invite,
)

# ---------------------------------------------------------------------------
# MeetingInvite data structure
# ---------------------------------------------------------------------------


class TestMeetingInviteDataStructure:
    def test_has_title_field(self) -> None:
        invite = MeetingInvite(title="Budget Review")
        assert invite.title == "Budget Review"

    def test_has_attendees_field(self) -> None:
        invite = MeetingInvite(title="x", attendees=["a@b.com", "c@d.com"])
        assert invite.attendees == ["a@b.com", "c@d.com"]

    def test_attendees_defaults_to_empty_list(self) -> None:
        invite = MeetingInvite(title="x")
        assert invite.attendees == []

    def test_has_start_time_field(self) -> None:
        dt = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
        invite = MeetingInvite(title="x", start_time=dt)
        assert invite.start_time == dt

    def test_start_time_defaults_to_none(self) -> None:
        assert MeetingInvite(title="x").start_time is None

    def test_has_end_time_field(self) -> None:
        dt = datetime(2026, 3, 15, 11, 0, tzinfo=UTC)
        invite = MeetingInvite(title="x", end_time=dt)
        assert invite.end_time == dt

    def test_end_time_defaults_to_none(self) -> None:
        assert MeetingInvite(title="x").end_time is None

    def test_has_agenda_field(self) -> None:
        invite = MeetingInvite(title="x", agenda="Q1 numbers")
        assert invite.agenda == "Q1 numbers"

    def test_agenda_defaults_to_empty_string(self) -> None:
        assert MeetingInvite(title="x").agenda == ""

    def test_is_complete_true_when_all_fields_set(self) -> None:
        invite = MeetingInvite(
            title="Budget Review",
            attendees=["a@b.com"],
            start_time=datetime(2026, 3, 15, 10, tzinfo=UTC),
            end_time=datetime(2026, 3, 15, 11, tzinfo=UTC),
            agenda="Q1",
        )
        assert invite.is_complete() is True

    def test_is_complete_false_missing_attendees(self) -> None:
        invite = MeetingInvite(
            title="x",
            start_time=datetime(2026, 3, 15, 10, tzinfo=UTC),
            end_time=datetime(2026, 3, 15, 11, tzinfo=UTC),
        )
        assert invite.is_complete() is False

    def test_is_complete_false_missing_start(self) -> None:
        invite = MeetingInvite(
            title="x",
            attendees=["a@b.com"],
            end_time=datetime(2026, 3, 15, 11, tzinfo=UTC),
        )
        assert invite.is_complete() is False

    def test_is_complete_false_missing_end(self) -> None:
        invite = MeetingInvite(
            title="x",
            attendees=["a@b.com"],
            start_time=datetime(2026, 3, 15, 10, tzinfo=UTC),
        )
        assert invite.is_complete() is False

    def test_is_complete_false_missing_title(self) -> None:
        invite = MeetingInvite(
            title="",
            attendees=["a@b.com"],
            start_time=datetime(2026, 3, 15, 10, tzinfo=UTC),
            end_time=datetime(2026, 3, 15, 11, tzinfo=UTC),
        )
        assert invite.is_complete() is False


# ---------------------------------------------------------------------------
# Natural-language parser — field population
# ---------------------------------------------------------------------------


class TestParseInviteFromText:
    """parse_invite_from_text must extract all fields from natural language."""

    _FULL_TEXT = (
        "Schedule a budget review with alice@example.com and bob@example.com "
        "on 2026-03-15 from 10:00 to 11:00. Agenda: Q1 numbers."
    )

    def test_extracts_attendees(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert "alice@example.com" in invite.attendees
        assert "bob@example.com" in invite.attendees

    def test_extracts_start_time(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert invite.start_time is not None
        assert invite.start_time.hour == 10
        assert invite.start_time.minute == 0

    def test_extracts_end_time(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert invite.end_time is not None
        assert invite.end_time.hour == 11
        assert invite.end_time.minute == 0

    def test_extracts_date_into_times(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert invite.start_time is not None
        assert invite.start_time.year == 2026
        assert invite.start_time.month == 3
        assert invite.start_time.day == 15

    def test_extracts_agenda(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert "Q1 numbers" in invite.agenda

    def test_extracts_title(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert invite.title  # non-empty
        assert "budget review" in invite.title.lower()

    def test_returns_meeting_invite_instance(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert isinstance(invite, MeetingInvite)

    def test_no_attendees_when_none_in_text(self) -> None:
        invite = parse_invite_from_text("Set up a team sync on 2026-03-20.")
        assert invite.attendees == []

    def test_multiple_attendees_parsed(self) -> None:
        text = "Book a call with a@x.com, b@x.com, c@x.com " "on 2026-04-01 from 09:00 to 09:30."
        invite = parse_invite_from_text(text)
        assert len(invite.attendees) == 3

    def test_iso_datetime_fallback(self) -> None:
        """Parser falls back to ISO datetimes when no time-range pattern present."""
        text = "Meeting on 2026-03-15T14:00 until 2026-03-15T15:00 " "with dev@corp.com."
        invite = parse_invite_from_text(text)
        assert invite.start_time is not None
        assert invite.start_time.hour == 14
        assert invite.end_time is not None
        assert invite.end_time.hour == 15

    def test_times_are_utc_aware(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert invite.start_time is not None
        assert invite.start_time.tzinfo is not None
        assert invite.end_time is not None
        assert invite.end_time.tzinfo is not None

    def test_start_before_end(self) -> None:
        invite = parse_invite_from_text(self._FULL_TEXT)
        assert invite.start_time is not None
        assert invite.end_time is not None
        assert invite.start_time < invite.end_time

    def test_text_with_topic_keyword(self) -> None:
        text = (
            "Arrange a sprint planning with pm@co.com on 2026-03-20 "
            "from 09:00 to 10:00. Topic: velocity and backlog."
        )
        invite = parse_invite_from_text(text)
        assert "velocity" in invite.agenda.lower() or "backlog" in invite.agenda.lower()

    def test_empty_text_returns_invite_with_defaults(self) -> None:
        invite = parse_invite_from_text("")
        assert isinstance(invite, MeetingInvite)
        assert invite.attendees == []
        assert invite.start_time is None
        assert invite.end_time is None


# ---------------------------------------------------------------------------
# Review formatter
# ---------------------------------------------------------------------------


class TestFormatInviteForReview:
    """format_invite_for_review must produce a complete, human-readable string."""

    _INVITE = MeetingInvite(
        title="Budget Review",
        attendees=["alice@example.com", "bob@example.com"],
        start_time=datetime(2026, 3, 15, 10, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 15, 11, 0, tzinfo=UTC),
        agenda="Q1 numbers",
    )

    def test_returns_string(self) -> None:
        result = format_invite_for_review(self._INVITE)
        assert isinstance(result, str)

    def test_contains_title(self) -> None:
        result = format_invite_for_review(self._INVITE)
        assert "Budget Review" in result

    def test_contains_attendees(self) -> None:
        result = format_invite_for_review(self._INVITE)
        assert "alice@example.com" in result
        assert "bob@example.com" in result

    def test_contains_start_time(self) -> None:
        result = format_invite_for_review(self._INVITE)
        assert "2026-03-15" in result
        assert "10:00" in result

    def test_contains_end_time(self) -> None:
        result = format_invite_for_review(self._INVITE)
        assert "11:00" in result

    def test_contains_agenda(self) -> None:
        result = format_invite_for_review(self._INVITE)
        assert "Q1 numbers" in result

    def test_missing_fields_shown_as_not_set(self) -> None:
        invite = MeetingInvite(title="")
        result = format_invite_for_review(invite)
        assert "not set" in result.lower() or "(none)" in result.lower()

    def test_review_label_present(self) -> None:
        result = format_invite_for_review(self._INVITE)
        # The formatter should make it clear this is for review
        assert "review" in result.lower()


# ---------------------------------------------------------------------------
# Stub sender — no real network calls
# ---------------------------------------------------------------------------


class TestStubSendInvite:
    _INVITE = MeetingInvite(
        title="Budget Review",
        attendees=["alice@example.com"],
        start_time=datetime(2026, 3, 15, 10, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 15, 11, 0, tzinfo=UTC),
        agenda="Q1 numbers",
    )

    def test_returns_ok_status(self) -> None:
        result = stub_send_invite(self._INVITE)
        assert result["status"] == "ok"

    def test_returns_invite_in_result(self) -> None:
        result = stub_send_invite(self._INVITE)
        assert result["invite"] is self._INVITE

    def test_no_network_calls(self) -> None:
        """Stub must not open any real network connections."""
        original_connect = socket.socket.connect

        def _deny_connect(self: socket.socket, *args: object) -> None:
            raise AssertionError("stub_send_invite must not make network calls")

        socket.socket.connect = _deny_connect  # type: ignore[method-assign]
        try:
            result = stub_send_invite(self._INVITE)
            assert result["status"] == "ok"
        finally:
            socket.socket.connect = original_connect  # type: ignore[method-assign]

    def test_error_when_title_missing(self) -> None:
        invite = MeetingInvite(title="")
        result = stub_send_invite(invite)
        assert result["status"] == "error"
        assert "reason" in result

    def test_logs_invite_details(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="rex.calendar_backends.meeting_invite"):
            stub_send_invite(self._INVITE)
        assert any("stub" in r.message.lower() for r in caplog.records)

    def test_result_is_dict(self) -> None:
        result = stub_send_invite(self._INVITE)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Import from package __init__
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meeting_invite_importable_from_package(self) -> None:
        from rex.calendar_backends import MeetingInvite as MI

        assert MI is MeetingInvite

    def test_parse_invite_importable_from_package(self) -> None:
        from rex.calendar_backends import parse_invite_from_text as p

        assert p is parse_invite_from_text

    def test_format_invite_importable_from_package(self) -> None:
        from rex.calendar_backends import format_invite_for_review as f

        assert f is format_invite_for_review

    def test_stub_send_importable_from_package(self) -> None:
        from rex.calendar_backends import stub_send_invite as s

        assert s is stub_send_invite
