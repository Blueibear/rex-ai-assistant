"""Calendar integration using mock data."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rex.event_bus import EventBus


@dataclass
class CalendarEvent:
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    location: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "location": self.location,
        }


class CalendarService:
    """Read/write calendar service backed by mock data."""

    def __init__(
        self,
        event_bus: EventBus,
        *,
        mock_data_path: Path | str | None = None,
        mock_events: list[CalendarEvent] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._mock_data_path = Path(mock_data_path) if mock_data_path else None
        self._events = mock_events if mock_events is not None else None

    def list_upcoming(self, *, horizon_hours: int = 72) -> list[CalendarEvent]:
        events = self._load_events()
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=horizon_hours)
        upcoming = [event for event in events if event.start_time <= horizon]
        self._event_bus.publish(
            "calendar.upcoming",
            {"count": len(upcoming), "events": [e.to_summary() for e in upcoming]},
        )
        return upcoming

    def create_event(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        *,
        location: str | None = None,
    ) -> CalendarEvent:
        event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=location,
        )
        events = self._load_events()
        events.append(event)
        self._events = events
        self._event_bus.publish("calendar.created", event.to_summary())
        return event

    def detect_conflicts(self, event: CalendarEvent) -> list[CalendarEvent]:
        conflicts = []
        for existing in self._load_events():
            if existing.event_id == event.event_id:
                continue
            if existing.start_time < event.end_time and event.start_time < existing.end_time:
                conflicts.append(existing)
        return conflicts

    def refresh_upcoming(self) -> list[CalendarEvent]:
        return self.list_upcoming()

    def _load_events(self) -> list[CalendarEvent]:
        if self._events is not None:
            return list(self._events)
        if self._mock_data_path and self._mock_data_path.exists():
            return self._load_mock_events(self._mock_data_path)
        return self._default_mock_events()

    def _load_mock_events(self, path: Path) -> list[CalendarEvent]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = []
        for item in payload.get("events", []):
            events.append(
                CalendarEvent(
                    event_id=item["event_id"],
                    title=item["title"],
                    start_time=datetime.fromisoformat(item["start_time"]).astimezone(
                        timezone.utc
                    ),
                    end_time=datetime.fromisoformat(item["end_time"]).astimezone(
                        timezone.utc
                    ),
                    location=item.get("location"),
                )
            )
        return events

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


__all__ = ["CalendarEvent", "CalendarService"]
