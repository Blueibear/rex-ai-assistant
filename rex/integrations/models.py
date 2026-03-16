"""Shared data models for Rex integrations (email, calendar, SMS).

All models are Pydantic v2 models with full type annotations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

PriorityLevel = Literal["low", "medium", "high", "critical"]


class EmailMessage(BaseModel):
    """A single email message.

    Attributes:
        id: Unique message identifier (provider-assigned).
        thread_id: Identifier of the thread this message belongs to.
        subject: Email subject line.
        sender: Sender's email address.
        recipients: List of recipient email addresses.
        body_text: Plain-text body of the message.
        body_html: HTML body of the message (may be ``None`` if not available).
        received_at: UTC datetime when the message was received.
        labels: List of label strings applied to the message (e.g. ``["INBOX"]``).
        is_read: Whether the message has been read.
        priority: Triage priority assigned by :class:`EmailTriageEngine`.
    """

    id: str
    thread_id: str
    subject: str
    sender: str
    recipients: list[str] = Field(default_factory=list)
    body_text: str = ""
    body_html: Optional[str] = None
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    labels: list[str] = Field(default_factory=list)
    is_read: bool = False
    priority: PriorityLevel = "low"


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


class CalendarEvent(BaseModel):
    """A single calendar event.

    Attributes:
        id: Unique event identifier (provider-assigned).
        title: Human-readable event title.
        start: UTC datetime when the event starts.
        end: UTC datetime when the event ends.
        location: Optional physical or virtual location string.
        description: Optional free-text event description.
        attendees: List of attendee email addresses.
        source: Origin of the event (e.g. ``"google"``, ``"rex"``, ``"stub"``).
        is_all_day: Whether the event spans a full day (no specific time).
        recurrence: Optional RRULE string describing recurrence (e.g.
            ``"RRULE:FREQ=WEEKLY;BYDAY=MO"``).
    """

    id: str
    title: str
    start: datetime
    end: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    attendees: list[str] = Field(default_factory=list)
    source: str = "rex"
    is_all_day: bool = False
    recurrence: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CalendarEvent",
    "EmailMessage",
    "PriorityLevel",
]
