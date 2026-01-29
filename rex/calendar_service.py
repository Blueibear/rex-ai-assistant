"""
Calendar service module for Rex AI Assistant.

Provides calendar integration with read/write capabilities using mock data.
A real calendar API integration (Google Calendar, Outlook, etc.) can be added later.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from rex.event_bus import EventBus

logger = logging.getLogger(__name__)


class _NoOpEventBus:
    """Fallback event bus used when no EventBus is provided."""

    def publish(self, *_args: Any, **_kwargs: Any) -> None:
        return


def _ensure_aware_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware in UTC.
    If a naive datetime is provided, it is assumed to already be UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True, init=False)
class CalendarEvent:
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    attendees: list[str] = field(default_factory=list)
    location: str | None = None
    description: str | None = None
    all_day: bool = False

    def __init__(
        self,
        *,
        event_id: str | None = None,
        id: str | None = None,
        title: str,
        start_time: datetime,
        end_time: datetime,
        attendees: list[str] | None = None,
        location: str | None = None,
        description: str | None = None,
        all_day: bool = False,
    ) -> None:
        resolved_id = event_id or id or str(uuid.uuid4())
        self.event_id = resolved_id
        self.title = title
        self.start_time = _ensure_aware_utc(start_time)
        self.end_time = _ensure_aware_utc(end_time)
        self.attendees = list(attendees) if attendees is not None else []
        self.location = location
        self.description = description
        self.all_day = all_day

    @property
    def id(self) -> str:
        """Compatibility alias for code that expects `id`."""
        return self.event_id

    @id.setter
    def id(self, value: str) -> None:
        self.event_id = value

    def to_summary(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "id": self.event_id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "attendees": list(self.attendees),
            "location": self.location,
            "description": self.description,
            "all_day": self.all_day,
        }

    def overlaps_with(self, other: "CalendarEvent") -> bool:
        """Return True if this event overlaps with another event."""
        return self.start_time < other.end_time and self.end_time > other.start_time


class CalendarService:
    """
    Read/write calendar service backed by mock data.

    Supports:
    - Loading from mock file (two formats supported)
    - Creating, updating, deleting events
    - Listing upcoming events
    - Getting events in a time range
    - Conflict detection
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        *,
        mock_data_path: Path | str | None = None,
        mock_events: list[CalendarEvent] | None = None,
    ) -> None:
        self._event_bus = event_bus if event_bus is not None else _NoOpEventBus()
        self._mock_data_path = Path(mock_data_path) if mock_data_path else Path("data/mock_calendar.json")
        self._events: list[CalendarEvent] | None = list(mock_events) if mock_events is not None else None
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to calendar service.

        For mock mode, this loads mock data. Always returns True unless a hard failure occurs.
        """
        try:
            self._events = self._load_events()
            self.connected = True
            logger.info("Calendar service connected (mock mode)")
            self._event_bus.publish("calendar.connected", {"connected": True, "count": len(self._events)})
            return True
        except Exception as e:
            logger.error("Failed to connect calendar service: %s", e, exc_info=True)
            self.connected = False
            self._event_bus.publish("calendar.connected", {"connected": False, "error": str(e)})
            return False

    def list_upcoming(self, *, horizon_hours: int = 72) -> list[CalendarEvent]:
        events = self._load_events()
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=horizon_hours)

        # Upcoming includes events that start within horizon or are currently ongoing
        upcoming = [
            event
            for event in events
            if (event.start_time <= horizon) and (event.end_time > now)
        ]
        upcoming.sort(key=lambda e: e.start_time)

        self._event_bus.publish(
            "calendar.upcoming",
            {"count": len(upcoming), "events": [e.to_summary() for e in upcoming]},
        )
        return upcoming

    def refresh_upcoming(self) -> list[CalendarEvent]:
        return self.list_upcoming()

    def get_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """
        Get calendar events that overlap a time range.
        """
        if not self.connected and self._events is None:
            logger.warning("Calendar service not connected; returning empty list")
            return []

        start_utc = _ensure_aware_utc(start)
        end_utc = _ensure_aware_utc(end)

        events = self._load_events()
        result = [e for e in events if e.start_time < end_utc and e.end_time > start_utc]
        result.sort(key=lambda e: e.start_time)

        self._event_bus.publish(
            "calendar.range",
            {"count": len(result), "start": start_utc.isoformat(), "end": end_utc.isoformat()},
        )
        return result

    def create_event(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        *,
        location: str | None = None,
        attendees: Optional[Iterable[str]] = None,
        description: str | None = None,
        all_day: bool = False,
    ) -> CalendarEvent:
        """
        Create a new calendar event and persist it to mock storage.
        """
        if not self.connected and self._events is None:
            # Allow creation even if connect() was not explicitly called
            self.connect()

        event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=location,
            attendees=list(attendees) if attendees is not None else [],
            description=description,
            all_day=all_day,
        )

        events = self._load_events()
        events.append(event)
        events.sort(key=lambda e: e.start_time)
        self._events = events
        self._save_events(events)

        self._event_bus.publish("calendar.created", event.to_summary())
        return event

    def update_event(self, event_id: str, updates: dict[str, Any]) -> CalendarEvent | None:
        """
        Update an existing calendar event by id.
        """
        events = self._load_events()
        for i, event in enumerate(events):
            if event.event_id != event_id:
                continue

            # Apply supported updates safely
            if "title" in updates:
                event.title = str(updates["title"])
            if "start_time" in updates and isinstance(updates["start_time"], datetime):
                event.start_time = _ensure_aware_utc(updates["start_time"])
            if "end_time" in updates and isinstance(updates["end_time"], datetime):
                event.end_time = _ensure_aware_utc(updates["end_time"])
            if "location" in updates:
                event.location = updates["location"]
            if "description" in updates:
                event.description = updates["description"]
            if "all_day" in updates:
                event.all_day = bool(updates["all_day"])
            if "attendees" in updates:
                att = updates["attendees"]
                if isinstance(att, (list, tuple)):
                    event.attendees = [str(a) for a in att]

            events[i] = event
            events.sort(key=lambda e: e.start_time)
            self._events = events
            self._save_events(events)

            self._event_bus.publish("calendar.updated", {"event": event.to_summary()})
            return event

        logger.warning("Event not found for update: %s", event_id)
        self._event_bus.publish("calendar.updated", {"event_id": event_id, "updated": False})
        return None

    def delete_event(self, event_id: str) -> bool:
        """
        Delete an event by id.
        """
        events = self._load_events()
        new_events = [e for e in events if e.event_id != event_id]
        if len(new_events) == len(events):
            logger.warning("Event not found for delete: %s", event_id)
            self._event_bus.publish("calendar.deleted", {"event_id": event_id, "deleted": False})
            return False

        self._events = new_events
        self._save_events(new_events)
        self._event_bus.publish("calendar.deleted", {"event_id": event_id, "deleted": True})
        return True

    def detect_conflicts(self, event: CalendarEvent) -> list[CalendarEvent]:
        """
        Detect conflicts between the provided event and existing events.
        """
        conflicts: list[CalendarEvent] = []
        for existing in self._load_events():
            if existing.event_id == event.event_id:
                continue
            if existing.overlaps_with(event):
                conflicts.append(existing)
        return conflicts

    def find_conflicts(
        self,
        events: Optional[list[CalendarEvent]] = None,
    ) -> list[tuple[CalendarEvent, CalendarEvent]]:
        """
        Find overlapping event pairs.
        """
        events_to_check = list(events) if events is not None else self._load_events()
        events_to_check.sort(key=lambda e: e.start_time)

        conflicts: list[tuple[CalendarEvent, CalendarEvent]] = []
        for i, e1 in enumerate(events_to_check):
            for e2 in events_to_check[i + 1 :]:
                if e1.overlaps_with(e2):
                    conflicts.append((e1, e2))
        return conflicts

    def get_upcoming_events(self, days: int = 7) -> list[CalendarEvent]:
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        return self.get_events(now, end)

    def get_all_events(self) -> list[CalendarEvent]:
        return self._load_events()

    def _load_events(self) -> list[CalendarEvent]:
        if self._events is not None:
            return list(self._events)

        path = self._mock_data_path
        if path and path.exists():
            events = self._load_mock_events(path)
            self._events = list(events)
            return list(events)

        events = self._default_mock_events()
        self._events = list(events)
        return list(events)

    def _load_mock_events(self, path: Path) -> list[CalendarEvent]:
        """
        Supports two file formats:

        Format A:
          { "events": [ { "event_id": "...", "title": "...", "start_time": "...", "end_time": "...", ... } ] }

        Format B:
          [ { "id": "...", "title": "...", "start_time": "...", "end_time": "...", ... }, ... ]
        """
        payload = json.loads(path.read_text(encoding="utf-8"))

        raw_events: list[dict[str, Any]]
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            raw_events = payload["events"]
        elif isinstance(payload, list):
            raw_events = payload
        else:
            logger.warning("Unrecognized mock calendar format in %s", path)
            return []

        events: list[CalendarEvent] = []
        for item in raw_events:
            event_id = item.get("event_id") or item.get("id") or str(uuid.uuid4())
            title = item.get("title", "Untitled event")

            start_raw = item.get("start_time")
            end_raw = item.get("end_time")

            if isinstance(start_raw, str):
                start_dt = datetime.fromisoformat(start_raw)
            elif isinstance(start_raw, datetime):
                start_dt = start_raw
            else:
                start_dt = datetime.now(timezone.utc)

            if isinstance(end_raw, str):
                end_dt = datetime.fromisoformat(end_raw)
            elif isinstance(end_raw, datetime):
                end_dt = end_raw
            else:
                end_dt = start_dt + timedelta(hours=1)

            attendees = item.get("attendees") or []
            if not isinstance(attendees, list):
                attendees = []

            events.append(
                CalendarEvent(
                    event_id=str(event_id),
                    title=str(title),
                    start_time=_ensure_aware_utc(start_dt),
                    end_time=_ensure_aware_utc(end_dt),
                    location=item.get("location"),
                    description=item.get("description"),
                    attendees=[str(a) for a in attendees],
                    all_day=bool(item.get("all_day", False)),
                )
            )

        events.sort(key=lambda e: e.start_time)
        return events

    def _save_events(self, events: list[CalendarEvent]) -> None:
        """
        Persist events to mock storage, using Format A for stability:
          { "events": [ ... ] }
        """
        try:
            path = self._mock_data_path
            if not path:
                return

            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"events": [e.to_summary() for e in events]}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save mock calendar data: %s", e, exc_info=True)

    def _default_mock_events(self) -> list[CalendarEvent]:
        now = datetime.now(timezone.utc)
        return [
            CalendarEvent(
                event_id="event-001",
                title="Product sync",
                start_time=now + timedelta(hours=2),
                end_time=now + timedelta(hours=3),
                location="Zoom",
            ),
            CalendarEvent(
                event_id="event-002",
                title="1:1 check-in",
                start_time=now + timedelta(hours=6),
                end_time=now + timedelta(hours=6, minutes=30),
                location="Conference Room A",
            ),
        ]


# Global calendar service instance (optional convenience)
_calendar_service: Optional[CalendarService] = None


def get_calendar_service(event_bus: Optional[EventBus] = None) -> CalendarService:
    """Get the global calendar service instance."""
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = CalendarService(event_bus=event_bus)
        _calendar_service.connect()
    return _calendar_service


def set_calendar_service(service: CalendarService) -> None:
    """Set the global calendar service instance (for testing)."""
    global _calendar_service
    _calendar_service = service


__all__ = [
    "CalendarEvent",
    "CalendarService",
    "get_calendar_service",
    "set_calendar_service",
]
