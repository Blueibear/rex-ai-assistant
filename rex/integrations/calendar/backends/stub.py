"""Stub calendar backend for offline development and testing.

Returns fixed in-memory events for ``get_upcoming`` and records ``create_event``
calls as a no-op.  Implements the :class:`CalendarBackend` transport-layer interface.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from rex.integrations.calendar.backends.base import CalendarBackend

logger = logging.getLogger(__name__)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _stub_events() -> list[dict]:
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "stub-cal-001",
            "title": "Team standup",
            "start": _iso(now + timedelta(hours=1)),
            "end": _iso(now + timedelta(hours=1, minutes=30)),
        },
        {
            "id": "stub-cal-002",
            "title": "Lunch with Bob",
            "start": _iso(now + timedelta(hours=3)),
            "end": _iso(now + timedelta(hours=4)),
        },
    ]


class StubCalendarBackend(CalendarBackend):
    """In-memory stub calendar backend implementing the transport-layer interface."""

    def __init__(self, events: list[dict] | None = None) -> None:
        self._events: list[dict] = list(events if events is not None else _stub_events())
        self._created: list[dict] = []

    def get_upcoming(self, days: int = 7) -> list[dict]:
        """Return stub events (all within the next *days* days by convention)."""
        return list(self._events)

    def create_event(self, title: str, start: str, end: str) -> dict:
        """Record a new event and return it with a generated id."""
        event = {
            "id": f"stub-cal-{uuid.uuid4().hex[:8]}",
            "title": title,
            "start": start,
            "end": end,
        }
        self._events.append(event)
        self._created.append(event)
        logger.info("[STUB CALENDAR] Created event: %r (%s – %s)", title, start, end)
        return event

    @property
    def created_events(self) -> list[dict]:
        """Events recorded by ``create_event()`` — useful for test assertions."""
        return list(self._created)


__all__ = ["StubCalendarBackend"]
