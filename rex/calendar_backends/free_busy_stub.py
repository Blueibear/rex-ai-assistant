"""CalendarStub — in-memory calendar backend with built-in mock free/busy data.

``CalendarStub`` implements the same :class:`CalendarBackend` interface as the
real ICS backend so that scheduling features (US-082, US-083) can be built and
tested without any live credentials or network access.

Mock events are generated relative to a configurable *anchor date* (defaulting
to today) so that tests can use a fixed reference date and still get
deterministic results regardless of when they run.

Free/busy querying
------------------
In addition to the standard ``fetch_events()`` method, the stub exposes
``get_free_busy(start, end)`` which returns only the events that overlap the
requested time window — the busy blocks used by the free-time finder.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

from rex.calendar_backends.base import CalendarBackend
from rex.calendar_service import CalendarEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free/busy block
# ---------------------------------------------------------------------------


class FreeBusyBlock:
    """A single busy interval returned by :meth:`CalendarStub.get_free_busy`.

    Attributes:
        start:   Start of the busy period (timezone-aware UTC).
        end:     End of the busy period (timezone-aware UTC).
        title:   Human-readable label (event title).
        event_id: Corresponding :class:`CalendarEvent` id.
    """

    __slots__ = ("end", "event_id", "start", "title")

    def __init__(
        self,
        *,
        start: datetime,
        end: datetime,
        title: str = "",
        event_id: str = "",
    ) -> None:
        self.start = start
        self.end = end
        self.title = title
        self.event_id = event_id

    def overlaps(self, window_start: datetime, window_end: datetime) -> bool:
        """Return True if this block overlaps [window_start, window_end)."""
        return self.start < window_end and self.end > window_start

    def __repr__(self) -> str:
        return (
            f"FreeBusyBlock(start={self.start.isoformat()!r}, "
            f"end={self.end.isoformat()!r}, title={self.title!r})"
        )


# ---------------------------------------------------------------------------
# Mock data factory
# ---------------------------------------------------------------------------


def _utc(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=UTC)


def _build_mock_events(anchor: date) -> list[CalendarEvent]:
    """Generate a realistic week of mock events relative to *anchor* (Monday).

    Events are spread across Monday–Friday of the same ISO week as *anchor*
    so the free-time finder has varied busy slots to work around.
    """
    # Find the Monday of the same ISO week
    monday = anchor - timedelta(days=anchor.weekday())

    tue = monday + timedelta(days=1)
    wed = monday + timedelta(days=2)
    thu = monday + timedelta(days=3)
    fri = monday + timedelta(days=4)

    events: list[CalendarEvent] = [
        # Monday
        CalendarEvent(
            event_id="stub-cal-001",
            title="Daily stand-up",
            start_time=_utc(monday, 9, 0),
            end_time=_utc(monday, 9, 15),
            attendees=["team@example.com"],
        ),
        CalendarEvent(
            event_id="stub-cal-002",
            title="Sprint planning",
            start_time=_utc(monday, 10, 0),
            end_time=_utc(monday, 12, 0),
            attendees=["team@example.com"],
        ),
        CalendarEvent(
            event_id="stub-cal-003",
            title="Lunch with Sarah",
            start_time=_utc(monday, 12, 30),
            end_time=_utc(monday, 13, 30),
            location="Bistro",
        ),
        # Tuesday
        CalendarEvent(
            event_id="stub-cal-004",
            title="Daily stand-up",
            start_time=_utc(tue, 9, 0),
            end_time=_utc(tue, 9, 15),
        ),
        CalendarEvent(
            event_id="stub-cal-005",
            title="1:1 with manager",
            start_time=_utc(tue, 14, 0),
            end_time=_utc(tue, 14, 30),
        ),
        # Wednesday
        CalendarEvent(
            event_id="stub-cal-006",
            title="Daily stand-up",
            start_time=_utc(wed, 9, 0),
            end_time=_utc(wed, 9, 15),
        ),
        CalendarEvent(
            event_id="stub-cal-007",
            title="Architecture review",
            start_time=_utc(wed, 11, 0),
            end_time=_utc(wed, 12, 30),
            attendees=["eng@example.com"],
        ),
        CalendarEvent(
            event_id="stub-cal-008",
            title="Team demo",
            start_time=_utc(wed, 16, 0),
            end_time=_utc(wed, 17, 0),
        ),
        # Thursday
        CalendarEvent(
            event_id="stub-cal-009",
            title="Daily stand-up",
            start_time=_utc(thu, 9, 0),
            end_time=_utc(thu, 9, 15),
        ),
        CalendarEvent(
            event_id="stub-cal-010",
            title="Product sync",
            start_time=_utc(thu, 15, 0),
            end_time=_utc(thu, 16, 0),
        ),
        # Friday
        CalendarEvent(
            event_id="stub-cal-011",
            title="Daily stand-up",
            start_time=_utc(fri, 9, 0),
            end_time=_utc(fri, 9, 15),
        ),
        CalendarEvent(
            event_id="stub-cal-012",
            title="Weekly retrospective",
            start_time=_utc(fri, 14, 0),
            end_time=_utc(fri, 15, 0),
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# CalendarStub
# ---------------------------------------------------------------------------


class CalendarStub(CalendarBackend):
    """In-memory calendar backend with built-in mock free/busy data.

    Parameters
    ----------
    anchor:
        Reference date used to generate mock events.  Defaults to
        2026-03-09 (a Monday) so tests have a stable, deterministic dataset.
    extra_events:
        Additional :class:`CalendarEvent` objects to inject alongside the
        built-in mock events (useful for scenario-specific tests).
    """

    #: Default anchor date — a Monday for deterministic weekly events.
    DEFAULT_ANCHOR: date = date(2026, 3, 9)

    def __init__(
        self,
        *,
        anchor: date | None = None,
        extra_events: list[CalendarEvent] | None = None,
    ) -> None:
        self._anchor = anchor or self.DEFAULT_ANCHOR
        self._events: list[CalendarEvent] = _build_mock_events(self._anchor)
        if extra_events:
            self._events.extend(extra_events)
        self._connected = False

    # ------------------------------------------------------------------
    # CalendarBackend interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        self._connected = True
        logger.info(
            "CalendarStub connected (anchor=%s, %d mock events)",
            self._anchor,
            len(self._events),
        )
        return True

    def fetch_events(self) -> list[CalendarEvent]:
        """Return all mock calendar events."""
        if not self._connected:
            self.connect()
        return list(self._events)

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def backend_name(self) -> str:
        return "free_busy_stub"

    def test_connection(self) -> tuple[bool, str | None]:
        return True, None

    # ------------------------------------------------------------------
    # Free/busy API
    # ------------------------------------------------------------------

    def get_free_busy(
        self,
        start: datetime,
        end: datetime,
    ) -> list[FreeBusyBlock]:
        """Return busy blocks that overlap the [start, end) window.

        Parameters
        ----------
        start:
            Start of the query window (timezone-aware recommended).
        end:
            End of the query window (exclusive, timezone-aware recommended).

        Returns
        -------
        list[FreeBusyBlock]
            Busy intervals ordered by start time.
        """
        if not self._connected:
            self.connect()

        blocks: list[FreeBusyBlock] = []
        for event in self._events:
            ev_start = _ensure_utc(event.start_time)
            ev_end = _ensure_utc(event.end_time)
            block = FreeBusyBlock(
                start=ev_start,
                end=ev_end,
                title=event.title,
                event_id=event.event_id,
            )
            if block.overlaps(_ensure_utc(start), _ensure_utc(end)):
                blocks.append(block)

        blocks.sort(key=lambda b: b.start)
        return blocks

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def all_events(self) -> list[CalendarEvent]:
        """All mock events — read + upcoming (for test assertions)."""
        return list(self._events)

    def inject_event(self, event: CalendarEvent) -> None:
        """Add an event to the stub at test runtime."""
        self._events.append(event)

    def clear_events(self) -> None:
        """Remove all events (useful to test the empty-calendar case)."""
        self._events = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as a UTC-aware datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


__all__ = [
    "CalendarStub",
    "FreeBusyBlock",
]
