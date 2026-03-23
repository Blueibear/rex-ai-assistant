"""
Calendar service module for Rex AI Assistant.

Provides calendar integration with read/write capabilities using mock data.
A real calendar API integration (Google Calendar, Outlook, etc.) can be added later.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rex.openclaw.event_bus import EventBus

logger = logging.getLogger(__name__)

_REPO_SEED_PATH = Path("data/mock_calendar.json")


def _runtime_calendar_path() -> Path:
    """Return a writable runtime path for calendar persistence.

    Uses OS-appropriate app data directories so the repo's
    data/mock_calendar.json is never modified at runtime.
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "rex-ai" / "calendar.json"


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


@dataclass(slots=True, init=False)  # type: ignore[call-overload]
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

    def overlaps_with(self, other: CalendarEvent) -> bool:
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

    Storage modes:
    - mock_events provided: in-memory only, no disk writes.
    - mock_data_path provided: read/write to that path (tests use tmp_path).
    - Neither provided: read seed from data/mock_calendar.json, write to a
      user-specific runtime directory so tracked repo files are never modified.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        *,
        mock_data_path: Path | str | None = None,
        mock_events: list[CalendarEvent] | None = None,
    ) -> None:
        self._event_bus = event_bus if event_bus is not None else _NoOpEventBus()

        if mock_events is not None:
            # In-memory mode: caller supplied events, never touch disk.
            self._seed_path: Path | None = None
            self._storage_path: Path | None = None
            self._events: list[CalendarEvent] | None = list(mock_events)
        elif mock_data_path is not None:
            # Explicit path: read and write to the caller-supplied location.
            p = Path(mock_data_path)
            self._seed_path = p
            self._storage_path = p
            self._events = None
        else:
            # Default mode: seed from repo fixture, persist to runtime dir.
            self._seed_path = _REPO_SEED_PATH
            self._storage_path = _runtime_calendar_path()
            self._events = None

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
            self._event_bus.publish(
                "calendar.connected", {"connected": True, "count": len(self._events)}
            )
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
            event for event in events if (event.start_time <= horizon) and (event.end_time > now)
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

    def list_past_events(self, *, lookback_hours: int = 72) -> list[CalendarEvent]:
        """Return events that ended within the lookback window."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=lookback_hours)
        events = self.get_events(start, now)
        past_events = [event for event in events if event.end_time <= now]
        past_events.sort(key=lambda e: e.end_time)
        return past_events

    def create_event(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        *,
        location: str | None = None,
        attendees: Iterable[str] | None = None,
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
        events: list[CalendarEvent] | None = None,
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

        # Try storage path first (runtime copy), then seed path
        for path in (self._storage_path, self._seed_path):
            if path is not None and path.exists():
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
        """Persist events to storage.

        Writes only to ``_storage_path``.  When the service was created with
        ``mock_events`` (in-memory mode), ``_storage_path`` is ``None`` and
        this method is a no-op — the repo seed file is never modified.
        """
        if self._storage_path is None:
            return

        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"events": [e.to_summary() for e in events]}
            self._storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
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

    def get_past_events(
        self,
        hours: int = 72,
        *,
        now: datetime | None = None,
    ) -> list[CalendarEvent]:
        """Get events that ended within the specified time window.

        Args:
            hours: Look back this many hours for ended events.
            now: Current time (defaults to UTC now).

        Returns:
            List of events that ended within the window, sorted by end_time.
        """
        check_time = now or datetime.now(timezone.utc)
        window_start = check_time - timedelta(hours=hours)

        events = self._load_events()
        past_events = [e for e in events if e.end_time <= check_time and e.end_time >= window_start]
        past_events.sort(key=lambda e: e.end_time, reverse=True)
        return past_events

    def generate_followup_cues(
        self,
        user_id: str,
        *,
        lookback_hours: int = 72,
        expire_hours: int = 168,
        now: datetime | None = None,
    ) -> int:
        """Generate follow-up cues from recent past calendar events.

        Creates cues for events that have ended within the lookback window.
        Skips:
        - Events that already have a cue
        - All-day events that look like holidays
        - Events with 'no-followup' in metadata/description

        Args:
            user_id: User ID to create cues for.
            lookback_hours: Only consider events ended within this many hours.
            expire_hours: Hours until the cue expires.
            now: Current time (defaults to UTC now).

        Returns:
            Number of cues created.
        """
        try:
            from rex.cue_store import get_cue_store
        except ImportError:
            logger.warning("CueStore not available, cannot generate followup cues")
            return 0

        check_time = now or datetime.now(timezone.utc)
        past_events = self.get_past_events(hours=lookback_hours, now=check_time)
        cue_store = get_cue_store()

        created_count = 0
        for event in past_events:
            # Skip if cue already exists for this event
            if cue_store.has_cue_for_source(user_id, "calendar", event.event_id):
                continue

            # Skip all-day events that look like holidays
            if event.all_day and self._looks_like_holiday(event):
                continue

            # Skip events marked as no-followup
            if self._is_no_followup(event):
                continue

            # Create the cue
            prompt = f"How did '{event.title}' go?"
            cue_store.add_cue(
                user_id=user_id,
                source_type="calendar",
                source_id=event.event_id,
                title=event.title,
                prompt=prompt,
                eligible_after=event.end_time,
                expires_in=timedelta(hours=expire_hours),
                metadata={
                    "event_id": event.event_id,
                    "start_time": event.start_time.isoformat(),
                    "end_time": event.end_time.isoformat(),
                    "location": event.location,
                },
            )
            created_count += 1
            logger.debug(f"Created followup cue for event '{event.title}'")

        if created_count:
            logger.info(f"Generated {created_count} followup cue(s) from calendar events")

        return created_count

    def _looks_like_holiday(self, event: CalendarEvent) -> bool:
        """Check if an all-day event looks like a holiday.

        Simple heuristic based on common holiday keywords.
        """
        if not event.all_day:
            return False

        title_lower = event.title.lower()
        holiday_keywords = [
            "holiday",
            "day off",
            "vacation",
            "pto",
            "christmas",
            "thanksgiving",
            "easter",
            "new year",
            "independence day",
            "memorial day",
            "labor day",
            "birthday",
            "anniversary",
        ]
        for keyword in holiday_keywords:
            if keyword in title_lower:
                return True
        return False

    def _is_no_followup(self, event: CalendarEvent) -> bool:
        """Check if an event is marked as no-followup.

        Checks for 'no-followup' or 'nofollowup' in:
        - Description
        - Title (unlikely but possible)
        """
        markers = ["no-followup", "nofollowup", "no_followup", "[no followup]"]

        # Check title
        title_lower = event.title.lower()
        for marker in markers:
            if marker in title_lower:
                return True

        # Check description
        if event.description:
            desc_lower = event.description.lower()
            for marker in markers:
                if marker in desc_lower:
                    return True

        return False


# Global calendar service instance (optional convenience)
_calendar_service: CalendarService | None = None


def get_calendar_service(
    event_bus: EventBus | None = None,
    config: dict | None = None,
) -> CalendarService:
    """Get the global calendar service instance.

    When ``config`` contains ``calendar.backend = "ics"``, the service is
    backed by an :class:`~rex.calendar_backends.ics_backend.ICSCalendarBackend`.
    Otherwise the existing stub/mock behaviour is used.
    """
    global _calendar_service
    if _calendar_service is not None:
        return _calendar_service

    # Determine backend from config
    backend_name = "stub"
    cal_cfg: dict = {}
    if config:
        cal_cfg = config.get("calendar", {})
        backend_name = cal_cfg.get("backend", "stub")

    if not config:
        # Try loading from disk
        try:
            _config_path = Path("config/rex_config.json")
            project_root = Path(__file__).resolve().parent.parent
            cfg_file = project_root / _config_path
            if cfg_file.exists():
                import json as _json

                disk_config = _json.loads(cfg_file.read_text(encoding="utf-8"))
                cal_cfg = disk_config.get("calendar", {})
                backend_name = cal_cfg.get("backend", "stub")
        except Exception:
            pass

    if backend_name == "ics":
        _calendar_service = _create_ics_backed_service(cal_cfg, event_bus)
    else:
        _calendar_service = CalendarService(event_bus=event_bus)
        _calendar_service.connect()

    return _calendar_service


def _create_ics_backed_service(
    cal_cfg: dict,
    event_bus: EventBus | None = None,
) -> CalendarService:
    """Create a CalendarService whose events come from an ICS backend."""
    from rex.calendar_backends.ics_backend import ICSCalendarBackend

    ics_cfg = cal_cfg.get("ics", {})
    source = ics_cfg.get("source", "")
    url_timeout = int(ics_cfg.get("url_timeout", 15))

    if not source:
        logger.warning("calendar.backend is 'ics' but no source configured; using stub")
        svc = CalendarService(event_bus=event_bus)
        svc.connect()
        return svc

    backend = ICSCalendarBackend(source=source, url_timeout=url_timeout)
    ok = backend.connect()
    if not ok:
        logger.warning("ICS backend failed to connect; falling back to stub")
        svc = CalendarService(event_bus=event_bus)
        svc.connect()
        return svc

    events = backend.fetch_events()
    svc = CalendarService(event_bus=event_bus, mock_events=events)
    svc.connected = True
    return svc


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
