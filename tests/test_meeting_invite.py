"""Smoke tests for rex.calendar_backends.meeting_invite."""

from __future__ import annotations

from datetime import UTC, datetime


def test_import():
    """Module imports without error."""
    from rex.calendar_backends import meeting_invite

    assert meeting_invite is not None


def test_meeting_invite_defaults():
    """MeetingInvite has sensible defaults."""
    from rex.calendar_backends.meeting_invite import MeetingInvite

    invite = MeetingInvite(title="Standup")

    assert invite.title == "Standup"
    assert invite.attendees == []
    assert invite.start_time is None
    assert invite.end_time is None
    assert invite.agenda == ""
    assert invite.uid  # auto-generated UUID


def test_is_complete_full():
    """is_complete() returns True when all required fields are set."""
    from rex.calendar_backends.meeting_invite import MeetingInvite

    invite = MeetingInvite(
        title="Budget review",
        attendees=["alice@example.com"],
        start_time=datetime(2026, 3, 15, 10, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 15, 11, 0, tzinfo=UTC),
    )
    assert invite.is_complete()


def test_is_complete_missing_time():
    """is_complete() returns False when times are missing."""
    from rex.calendar_backends.meeting_invite import MeetingInvite

    invite = MeetingInvite(title="No time", attendees=["a@b.com"])
    assert not invite.is_complete()


def test_parse_invite_from_text_full():
    """parse_invite_from_text extracts attendees, title, time, and agenda."""
    from rex.calendar_backends.meeting_invite import parse_invite_from_text

    text = (
        "Schedule a budget review with alice@example.com and bob@example.com "
        "on 2026-03-15 from 10:00 to 11:00. Agenda: Q1 numbers."
    )
    invite = parse_invite_from_text(text)

    assert "alice@example.com" in invite.attendees
    assert "bob@example.com" in invite.attendees
    assert invite.start_time is not None
    assert invite.end_time is not None
    assert invite.start_time.hour == 10
    assert invite.end_time.hour == 11
    assert "Q1 numbers" in invite.agenda


def test_parse_invite_from_text_iso_datetimes():
    """parse_invite_from_text handles ISO datetime pairs."""
    from rex.calendar_backends.meeting_invite import parse_invite_from_text

    text = "Meeting on 2026-04-01T14:00 to 2026-04-01T15:00 with jane@corp.io"
    invite = parse_invite_from_text(text)

    assert invite.start_time is not None
    assert invite.start_time.hour == 14
    assert "jane@corp.io" in invite.attendees


def test_parse_invite_no_emails():
    """parse_invite_from_text returns empty attendees when none found."""
    from rex.calendar_backends.meeting_invite import parse_invite_from_text

    invite = parse_invite_from_text("Quick sync on 2026-03-20 from 09:00 to 09:30")
    assert invite.attendees == []


def test_format_invite_for_review():
    """format_invite_for_review returns a non-empty string with title."""
    from rex.calendar_backends.meeting_invite import MeetingInvite, format_invite_for_review

    invite = MeetingInvite(
        title="Design review",
        attendees=["eng@company.com"],
        start_time=datetime(2026, 3, 20, 9, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 20, 10, 0, tzinfo=UTC),
        agenda="Review wireframes",
    )
    output = format_invite_for_review(invite)

    assert "Design review" in output
    assert "eng@company.com" in output
    assert "Review wireframes" in output


def test_format_invite_missing_fields():
    """format_invite_for_review handles None times gracefully."""
    from rex.calendar_backends.meeting_invite import MeetingInvite, format_invite_for_review

    invite = MeetingInvite(title="Incomplete")
    output = format_invite_for_review(invite)

    assert "(not set)" in output


def test_stub_send_invite_success():
    """stub_send_invite returns status=ok for a valid invite."""
    from rex.calendar_backends.meeting_invite import MeetingInvite, stub_send_invite

    invite = MeetingInvite(title="All-hands")
    result = stub_send_invite(invite)

    assert result["status"] == "ok"
    assert result["invite"] is invite


def test_stub_send_invite_no_title():
    """stub_send_invite returns status=error when title is empty."""
    from rex.calendar_backends.meeting_invite import MeetingInvite, stub_send_invite

    invite = MeetingInvite(title="")
    result = stub_send_invite(invite)

    assert result["status"] == "error"
    assert "reason" in result


def test_to_ical_basic():
    """to_ical returns valid iCalendar text with VCALENDAR wrapper."""
    from rex.calendar_backends.meeting_invite import MeetingInvite, to_ical

    invite = MeetingInvite(
        title="Sync",
        attendees=["a@b.com"],
        start_time=datetime(2026, 3, 20, 9, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 20, 9, 30, tzinfo=UTC),
        agenda="Quick update",
    )
    ical = to_ical(invite)

    assert "BEGIN:VCALENDAR" in ical
    assert "END:VCALENDAR" in ical
    assert "SUMMARY:Sync" in ical
    assert "mailto:a@b.com" in ical
    assert "DESCRIPTION:Quick update" in ical
    assert "DTSTART:20260320T090000Z" in ical


def test_to_ical_no_times():
    """to_ical omits DTSTART/DTEND when times are None (incomplete draft)."""
    from rex.calendar_backends.meeting_invite import MeetingInvite, to_ical

    invite = MeetingInvite(title="Draft")
    ical = to_ical(invite)

    assert "DTSTART" not in ical
    assert "DTEND" not in ical
    assert "SUMMARY:Draft" in ical
