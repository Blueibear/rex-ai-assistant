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
# Scheduling
# ---------------------------------------------------------------------------


class TimeSlot(BaseModel):
    """A candidate meeting time slot.

    Attributes:
        start: UTC datetime when the slot starts.
        end: UTC datetime when the slot ends.
        confidence: Score in [0.0, 1.0] indicating how suitable the slot is.
    """

    start: datetime
    end: datetime
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# SMS
# ---------------------------------------------------------------------------

SMSDirection = Literal["inbound", "outbound"]
SMSStatus = Literal["sent", "delivered", "failed", "stub"]


class SMSMessage(BaseModel):
    """A single SMS message.

    Attributes:
        id: Unique message identifier (provider-assigned or stub-generated).
        thread_id: Identifier of the thread this message belongs to.
        direction: Whether the message was received (``"inbound"``) or sent
            (``"outbound"``).
        body: Plain-text body of the message.
        from_number: Sender phone number in E.164 format (e.g. ``"+14155550100"``).
        to_number: Recipient phone number in E.164 format.
        sent_at: UTC datetime when the message was sent or received.
        status: Delivery status of the message.
    """

    id: str
    thread_id: str
    direction: SMSDirection
    body: str
    from_number: str
    to_number: str
    sent_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: SMSStatus = "stub"


class SMSThread(BaseModel):
    """A conversation thread grouping SMS messages with one contact.

    Attributes:
        id: Unique thread identifier.
        contact_name: Display name for the contact (may be a number if unknown).
        contact_number: Phone number of the contact in E.164 format.
        messages: Ordered list of messages in the thread (oldest first).
        last_message_at: UTC datetime of the most recent message.
        unread_count: Number of unread inbound messages.
    """

    id: str
    contact_name: str
    contact_number: str
    messages: list[SMSMessage] = Field(default_factory=list)
    last_message_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    unread_count: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CalendarEvent",
    "EmailMessage",
    "PriorityLevel",
    "SMSDirection",
    "SMSMessage",
    "SMSStatus",
    "SMSThread",
    "TimeSlot",
]
