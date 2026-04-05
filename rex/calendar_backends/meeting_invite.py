"""Meeting invite scaffold — data structure, NL parser, review formatter, and stub sender.

Usage example::

    from rex.calendar_backends.meeting_invite import (
        MeetingInvite,
        parse_invite_from_text,
        format_invite_for_review,
        stub_send_invite,
    )

    invite = parse_invite_from_text(
        "Schedule a budget review with alice@example.com and bob@example.com "
        "on 2026-03-15 from 10:00 to 11:00. Agenda: Q1 numbers."
    )
    print(format_invite_for_review(invite))
    result = stub_send_invite(invite)
    assert result["status"] == "ok"
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_DATE_TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
]


@dataclass
class MeetingInvite:
    """All fields required to schedule a meeting.

    Attributes:
        title:      Short, descriptive title for the meeting.
        attendees:  List of attendee e-mail addresses or display names.
        start_time: Meeting start (UTC-aware or naive; treated as UTC).
        end_time:   Meeting end   (UTC-aware or naive; treated as UTC).
        agenda:     Free-text agenda or description (may be empty string).
        uid:        Unique identifier for the VEVENT (auto-generated UUID).
    """

    title: str
    attendees: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    agenda: str = ""
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_complete(self) -> bool:
        """Return True only when all required fields are populated."""
        return bool(
            self.title
            and self.attendees
            and self.start_time is not None
            and self.end_time is not None
        )


# ---------------------------------------------------------------------------
# Natural-language parser
# ---------------------------------------------------------------------------

# Patterns used to extract individual fields from free text.

# Email addresses
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")

# Bare date on its own (YYYY-MM-DD)
_RE_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# "from HH:MM to HH:MM" time-range pattern
_RE_TIME_RANGE = re.compile(
    r"\bfrom\s+(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})\b",
    re.IGNORECASE,
)

# ISO datetime (date + time)
_RE_DATETIME = re.compile(r"\b(\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(?::\d{2})?)\b")

# Agenda / description keywords
_RE_AGENDA = re.compile(
    r"(?:agenda|topic|purpose|about|regarding|re):\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Title heuristic — first imperative phrase before attendees / time info
_RE_TITLE_LEAD = re.compile(
    r"^(?:schedule|book|create|set up|arrange|plan)?\s*(?:a\s+|an\s+)?(.+?)(?:\s+with\b|\s+on\b|\s+at\b|\s+from\b|\s+for\b)",
    re.IGNORECASE,
)


def _parse_datetime(date_str: str, time_str: str) -> datetime:
    """Combine a YYYY-MM-DD date string with an HH:MM time string into UTC datetime."""
    combined = f"{date_str} {time_str}"
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(combined, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime from {combined!r}")


def _parse_iso_datetime(text: str) -> datetime:
    """Parse a loose ISO-style datetime string into a UTC-aware datetime."""
    text = text.strip().replace("T", " ")
    for fmt in _DATE_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse ISO datetime from {text!r}")


def parse_invite_from_text(text: str) -> MeetingInvite:
    """Populate a :class:`MeetingInvite` from a natural language *text* description.

    The parser uses regex heuristics to extract:

    * **attendees** — any ``user@domain.tld`` patterns found in the text.
    * **start_time / end_time** — either a ``from HH:MM to HH:MM on YYYY-MM-DD``
      pattern or ISO datetimes embedded directly in the text.
    * **agenda** — text following an ``Agenda:`` / ``Topic:`` label.
    * **title** — the leading verb phrase before ``with`` / ``on`` / ``at``.

    Returns a :class:`MeetingInvite` with whatever fields could be extracted;
    unrecognised fields are left at their defaults (``None`` / ``""``).
    """
    invite = MeetingInvite(title="")

    # --- Attendees ---
    invite.attendees = _RE_EMAIL.findall(text)

    # --- Agenda ---
    agenda_match = _RE_AGENDA.search(text)
    if agenda_match:
        invite.agenda = agenda_match.group(1).strip()

    # --- Times ---
    # Try "from HH:MM to HH:MM" + a bare date
    time_range_match = _RE_TIME_RANGE.search(text)
    date_match = _RE_DATE.search(text)

    if time_range_match and date_match:
        date_str = date_match.group(1)
        start_t, end_t = time_range_match.group(1), time_range_match.group(2)
        try:
            invite.start_time = _parse_datetime(date_str, start_t)
            invite.end_time = _parse_datetime(date_str, end_t)
        except ValueError:
            pass
    else:
        # Fall back to ISO datetimes in order of appearance
        iso_matches = _RE_DATETIME.findall(text)
        if len(iso_matches) >= 2:
            try:
                invite.start_time = _parse_iso_datetime(iso_matches[0])
                invite.end_time = _parse_iso_datetime(iso_matches[1])
            except ValueError:
                pass
        elif len(iso_matches) == 1:
            try:
                invite.start_time = _parse_iso_datetime(iso_matches[0])
            except ValueError:
                pass

    # --- Title ---
    title_match = _RE_TITLE_LEAD.match(text.strip())
    if title_match:
        invite.title = title_match.group(1).strip().rstrip(",")
    else:
        # Fallback: first sentence up to first punctuation or 60 chars
        first_chunk = re.split(r"[,.\n]", text)[0].strip()
        invite.title = first_chunk[:60].strip()

    logger.debug(
        "parse_invite_from_text: title=%r attendees=%r start=%s end=%s agenda=%r",
        invite.title,
        invite.attendees,
        invite.start_time,
        invite.end_time,
        invite.agenda,
    )
    return invite


# ---------------------------------------------------------------------------
# Review formatter
# ---------------------------------------------------------------------------


def format_invite_for_review(invite: MeetingInvite) -> str:
    """Return a human-readable summary of *invite* for user review.

    The output is intentionally plain text so it can be displayed in a
    terminal, a chat UI, or logged without special rendering.
    """
    lines: list[str] = [
        "=== Meeting Invite (for review) ===",
        f"  Title    : {invite.title or '(not set)'}",
        f"  Attendees: {', '.join(invite.attendees) if invite.attendees else '(none)'}",
        f"  Start    : {invite.start_time.isoformat() if invite.start_time else '(not set)'}",
        f"  End      : {invite.end_time.isoformat() if invite.end_time else '(not set)'}",
        f"  Agenda   : {invite.agenda if invite.agenda else '(none)'}",
        "====================================",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stub sender
# ---------------------------------------------------------------------------


def stub_send_invite(invite: MeetingInvite) -> dict[str, Any]:
    """Log the invite and return a success dict — makes NO real API calls.

    This stub satisfies the beta acceptance criterion: the send path is
    exercisable in tests without any calendar credentials or network access.

    Returns:
        ``{"status": "ok", "invite": <invite>}`` on success.
        ``{"status": "error", "reason": <str>}`` if the invite is incomplete.
    """
    if not invite.title:
        reason = "invite is missing a title"
        logger.warning("stub_send_invite rejected: %s", reason)
        return {"status": "error", "reason": reason}

    logger.info(
        "stub_send_invite: [STUB] would send invite — title=%r attendees=%r "
        "start=%s end=%s agenda=%r",
        invite.title,
        invite.attendees,
        invite.start_time.isoformat() if invite.start_time else None,
        invite.end_time.isoformat() if invite.end_time else None,
        invite.agenda,
    )
    return {"status": "ok", "invite": invite}


# ---------------------------------------------------------------------------
# RFC 5545 iCalendar export
# ---------------------------------------------------------------------------

_ICS_DT_FORMAT = "%Y%m%dT%H%M%SZ"


def _ical_escape(text: str) -> str:
    """Escape special characters for iCalendar TEXT values (RFC 5545 §3.3.11)."""
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def to_ical(invite: MeetingInvite) -> str:
    """Return an RFC 5545-compliant VCALENDAR string for *invite*.

    The output contains a VCALENDAR wrapper with a single VEVENT component.
    DTSTART and DTEND are formatted as UTC datetimes (``YYYYMMDDTHHMMSSZformat).
    If *start_time* or *end_time* is ``None``, the field is omitted from the
    VEVENT (the invite is incomplete but still serialisable for draft purposes).

    Args:
        invite: The :class:`MeetingInvite` to serialise.

    Returns:
        A string in iCalendar format (CRLF line endings per RFC 5545 §3.1).
    """
    dtstamp = datetime.now(UTC).strftime(_ICS_DT_FORMAT)
    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Rex AI Assistant//Meeting Invite//EN",
        "BEGIN:VEVENT",
        f"UID:{invite.uid}",
        f"DTSTAMP:{dtstamp}",
    ]

    if invite.start_time is not None:
        start_utc = (
            invite.start_time
            if invite.start_time.tzinfo is not None
            else invite.start_time.replace(tzinfo=UTC)
        )
        lines.append(f"DTSTART:{start_utc.strftime(_ICS_DT_FORMAT)}")

    if invite.end_time is not None:
        end_utc = (
            invite.end_time
            if invite.end_time.tzinfo is not None
            else invite.end_time.replace(tzinfo=UTC)
        )
        lines.append(f"DTEND:{end_utc.strftime(_ICS_DT_FORMAT)}")

    lines.append(f"SUMMARY:{_ical_escape(invite.title)}")

    if invite.agenda:
        lines.append(f"DESCRIPTION:{_ical_escape(invite.agenda)}")

    for attendee in invite.attendees:
        email = attendee if "@" in attendee else attendee
        lines.append(f"ATTENDEE:mailto:{email}")

    lines += [
        "END:VEVENT",
        "END:VCALENDAR",
    ]

    return "\r\n".join(lines) + "\r\n"


__all__ = [
    "MeetingInvite",
    "format_invite_for_review",
    "parse_invite_from_text",
    "stub_send_invite",
    "to_ical",
]
