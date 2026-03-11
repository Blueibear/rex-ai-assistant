"""Calendar backend adapters.

Provides pluggable calendar sources (stub/mock, ICS file/URL) so the
CalendarService can swap backends transparently, following the same
pattern used by ``rex.email_backends``.
"""

from rex.calendar_backends.base import CalendarBackend
from rex.calendar_backends.free_busy_stub import CalendarStub, FreeBusyBlock
from rex.calendar_backends.free_time_finder import TimeSlot, find_free_slots
from rex.calendar_backends.meeting_invite import (
    MeetingInvite,
    format_invite_for_review,
    parse_invite_from_text,
    stub_send_invite,
)

__all__ = [
    "CalendarBackend",
    "CalendarStub",
    "FreeBusyBlock",
    "MeetingInvite",
    "TimeSlot",
    "find_free_slots",
    "format_invite_for_review",
    "parse_invite_from_text",
    "stub_send_invite",
]
