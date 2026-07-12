"""
Calendar service module for Rex AI Assistant.

Provides calendar integration with read/write capabilities using per-user
stores (stub/mock data or an ICS backend resolved per authorized account).

Per-user isolation (issue #303):
- Every operation that reads or mutates calendar data requires an explicit,
  validated ``user_id``.  Missing, blank, malformed, or traversal-style
  identities fail closed (``CalendarIdentityError``) before any account or
  credential lookup.
- Account selection is restricted to the requesting user's authorized
  accounts via :class:`rex.calendar_accounts.CalendarAccountResolver`;
  explicit foreign or nonexistent accounts raise the generic
  ``CalendarAccountAccessError`` (indistinguishable to the caller).
- Backends are cached per ``(user_id, account_id)``; a backend resolved for
  one user is never reused for another.
- Stub/mock mode keeps a separate mutable event store per user so one
  user's create/update/delete never alters another user's view.
- Private event fields (titles, attendees, locations, descriptions) are
  published only on user-scoped topics (``{topic}.user.{user_id}``); shared
  topics carry a safe envelope (user_id, count, success) only.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.calendar_accounts import (
    ACCOUNT_UNAVAILABLE_MSG,
    DEFAULT_PROFILE,
    CalendarAccountAccessError,
    CalendarAccountDefinition,
    CalendarAccountResolver,
    require_user_id,
)
from rex.openclaw.event_bus import EventBus

logger = logging.getLogger(__name__)

_REPO_SEED_PATH = Path("data/mock_calendar.json")

_NOT_CONFIGURED_FOR_USER_MSG = "Calendar: not configured"


def _runtime_calendar_path(user_id: str) -> Path:
    """Return a writable per-user runtime path for calendar persistence.

    Uses OS-appropriate app data directories so the repo's
    data/mock_calendar.json is never modified at runtime.  The legacy
    single-user file (``rex-ai/calendar.json``) is preserved for the
    ``default`` profile only; named users get isolated per-user files.
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    if user_id == DEFAULT_PROFILE:
        return base / "rex-ai" / "calendar.json"
    return base / "rex-ai" / "calendar" / f"{user_id}.json"


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
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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

    def overlaps_with(self, other: CalendarEvent) -> bool:
        """Return True if this event overlaps with another event."""
        return self.start_time < other.end_time and self.end_time > other.start_time


class _EventStore:
    """One user's isolated mutable event list, optionally disk-backed."""

    __slots__ = ("account_id", "events", "storage_path")

    def __init__(
        self,
        events: list[CalendarEvent],
        storage_path: Path | None,
        account_id: str | None,
    ) -> None:
        self.events = events
        self.storage_path = storage_path
        self.account_id = account_id


class CalendarService:
    """
    Read/write calendar service with per-user isolated stores.

    Storage modes:
    - ``mock_events`` provided: in-memory only, no disk writes.  The events
      belong to ``owner_user_id`` (default: the ``default`` profile); other
      users see their own empty isolated stores.
    - ``mock_data_path`` provided: read/write to that path (tests use
      tmp_path) for ``owner_user_id`` only; other users get isolated
      in-memory stores.
    - Neither provided: accounts are resolved per validated user via
      :class:`CalendarAccountResolver`.  Users whose resolved account is an
      ICS source get a read-seeded in-memory store; stub users get an
      isolated per-user runtime file seeded from ``data/mock_calendar.json``.
      When real accounts are configured, a user with no authorized account
      fails closed (reads return ``[]``, writes raise
      :class:`IntegrationNotConfiguredError`) — never another user's
      account.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        *,
        mock_data_path: Path | str | None = None,
        mock_events: list[CalendarEvent] | None = None,
        owner_user_id: str = DEFAULT_PROFILE,
        account_resolver: CalendarAccountResolver | None = None,
    ) -> None:
        self._event_bus = event_bus if event_bus is not None else _NoOpEventBus()
        self._owner = require_user_id(owner_user_id)

        self._mock_events: list[CalendarEvent] | None = (
            list(mock_events) if mock_events is not None else None
        )
        self._mock_data_path: Path | None = Path(mock_data_path) if mock_data_path else None

        # Per-user isolated stores, keyed by (user_id, account_id or "").
        self._stores: dict[tuple[str, str], _EventStore] = {}
        # ICS backends cached per (user_id, account_id).  A backend created
        # for one user is never served to another.
        self._user_backends: dict[tuple[str, str], Any] = {}

        # Injected authorization/routing resolver (tests/embedding only; the
        # production path loads lazily so config changes — including account
        # revocations — are picked up without a restart).
        self._injected_resolver = account_resolver
        self._resolver_cache: CalendarAccountResolver | None = None
        self._resolver_stamp: int | None = None

        self.connected = False

    # ------------------------------------------------------------------
    # Resolver / store plumbing
    # ------------------------------------------------------------------

    def _get_resolver(self) -> CalendarAccountResolver:
        """Return the account resolver, refreshing when the config changes.

        Authorization is re-checked on every operation, so cached stores and
        backends never outlive their owner's assignment.
        """
        if self._injected_resolver is not None:
            return self._injected_resolver

        from rex import calendar_accounts as _calendar_accounts

        stamp = _calendar_accounts.config_stamp()
        if self._resolver_cache is None or stamp != self._resolver_stamp:
            self._resolver_cache = CalendarAccountResolver.load()
            self._resolver_stamp = stamp
        return self._resolver_cache

    def _seeded_stub_events(self, user_id: str) -> tuple[list[CalendarEvent], Path | None]:
        """Load a user's stub store from disk (runtime file, then repo seed)."""
        storage_path = _runtime_calendar_path(user_id)
        for path in (storage_path, _REPO_SEED_PATH):
            if path is not None and path.exists():
                try:
                    return self._load_mock_events(path), storage_path
                except Exception as exc:
                    logger.warning("Failed to load calendar data from %s: %s", path, exc)
        return [], storage_path

    def _get_store(
        self,
        user_id: str,
        account_id: str | None = None,
    ) -> _EventStore | None:
        """Return *user_id*'s isolated store, after ownership validation.

        Returns ``None`` when real accounts are configured but none is
        available/usable for this user (documented not-configured result —
        callers fail closed, never through another user's account).

        Raises:
            CalendarIdentityError: On missing or invalid identity.
            CalendarAccountAccessError: When *account_id* is explicitly
                requested but not available to this user.
        """
        validated = require_user_id(user_id)

        # In-memory / explicit-path modes are bound to exactly one owner.
        if self._mock_events is not None or self._mock_data_path is not None:
            if account_id:
                # No configured accounts exist in these modes; unauthorized
                # and nonexistent are indistinguishable.
                raise CalendarAccountAccessError(
                    ACCOUNT_UNAVAILABLE_MSG.format(account_id=account_id, user_id=validated)
                )
            key = (validated, "")
            store = self._stores.get(key)
            if store is None:
                if validated == self._owner:
                    if self._mock_events is not None:
                        store = _EventStore(list(self._mock_events), None, None)
                    else:
                        events: list[CalendarEvent] = []
                        assert self._mock_data_path is not None
                        if self._mock_data_path.exists():
                            try:
                                events = self._load_mock_events(self._mock_data_path)
                            except Exception as exc:
                                logger.warning(
                                    "Failed to load calendar data from %s: %s",
                                    self._mock_data_path,
                                    exc,
                                )
                        store = _EventStore(events, self._mock_data_path, None)
                else:
                    # Isolated empty in-memory store for any other user.
                    store = _EventStore([], None, None)
                self._stores[key] = store
            return store

        resolver = self._get_resolver()
        if not resolver.has_configured_accounts():
            # Pure stub mode: isolated per-user disk store.
            if account_id:
                raise CalendarAccountAccessError(
                    ACCOUNT_UNAVAILABLE_MSG.format(account_id=account_id, user_id=validated)
                )
            key = (validated, "")
            store = self._stores.get(key)
            if store is None:
                events, storage_path = self._seeded_stub_events(validated)
                store = _EventStore(events, storage_path, None)
                self._stores[key] = store
            return store

        # Accounts are configured: ownership check before anything else.
        definition = resolver.resolve_account(validated, account_id)
        if definition is None:
            logger.warning("No usable calendar account for user %r", validated)
            return None

        if definition.provider == "ics":
            return self._ics_store(validated, definition)
        if definition.provider == "stub":
            key = (validated, definition.id)
            store = self._stores.get(key)
            if store is None:
                events, storage_path = self._seeded_stub_events(validated)
                store = _EventStore(events, storage_path, definition.id)
                self._stores[key] = store
            return store

        # google/outlook accounts are served by the provider-API surface
        # (rex.integrations.calendar_service), not this store-backed service.
        logger.warning(
            "Calendar account for user %r uses provider %r, which this surface " "does not serve",
            validated,
            definition.provider,
        )
        return None

    def _ics_store(self, user_id: str, definition: CalendarAccountDefinition) -> _EventStore | None:
        """Return the user's store seeded from their authorized ICS source."""
        key = (user_id, definition.id)
        store = self._stores.get(key)
        if store is not None:
            return store

        backend = self._user_backends.get(key)
        if backend is None:
            from rex.calendar_backends.ics_backend import ICSCalendarBackend

            backend = ICSCalendarBackend(
                source=definition.ics_source,
                url_timeout=definition.ics_url_timeout,
            )
            if not backend.connect():
                logger.warning("ICS calendar backend failed to connect for user %r", user_id)
                return None
            self._user_backends[key] = backend

        events = list(backend.fetch_events())
        store = _EventStore(events, None, definition.id)
        self._stores[key] = store
        return store

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            self._event_bus.publish(topic, payload)
        except Exception as exc:
            logger.debug("EventBus publish failed for %s: %s", topic, exc)

    def _publish_user_event(
        self,
        topic: str,
        user_id: str,
        shared_payload: dict[str, Any],
        private_payload: dict[str, Any] | None = None,
    ) -> None:
        """Publish a safe envelope on the shared topic and the full payload
        on the owner's user-scoped topic.

        The shared topic must never carry private event fields (titles,
        attendees, locations, descriptions); those go only to
        ``{topic}.user.{user_id}``.
        """
        self._publish(topic, shared_payload)
        if private_payload is not None:
            self._publish(f"{topic}.user.{user_id}", private_payload)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, user_id: str | None = None) -> bool:
        """
        Connect the calendar service for one validated user.

        Loads that user's isolated events.  Fails closed (returns ``False``)
        on missing/invalid identity or when no account is available to the
        user.
        """
        try:
            validated = require_user_id(user_id)
        except PermissionError:
            logger.warning("Calendar connect refused: a valid user identity is required")
            return False

        try:
            store = self._get_store(validated)
        except PermissionError:
            return False
        except Exception as exc:
            logger.error("Failed to connect calendar service: %s", exc, exc_info=True)
            self._publish(
                "calendar.connected",
                {"connected": False, "user_id": validated, "error": str(exc)},
            )
            return False

        if store is None:
            self._publish(
                "calendar.connected",
                {"connected": False, "user_id": validated},
            )
            return False

        self.connected = True
        logger.info("Calendar service connected (local storage) for user %s", validated)
        self._publish(
            "calendar.connected",
            {"connected": True, "count": len(store.events), "user_id": validated},
        )
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def _events_for(self, user_id: str, account_id: str | None = None) -> list[CalendarEvent]:
        """Return a copy of the requesting user's events (fail closed to [])."""
        store = self._get_store(user_id, account_id)
        if store is None:
            return []
        return list(store.events)

    def list_upcoming(
        self,
        *,
        horizon_hours: int = 72,
        user_id: str | None = None,
        account_id: str | None = None,
    ) -> list[CalendarEvent]:
        validated = require_user_id(user_id)
        events = self._events_for(validated, account_id)
        now = datetime.now(UTC)
        horizon = now + timedelta(hours=horizon_hours)

        # Upcoming includes events that start within horizon or are currently ongoing
        upcoming = [
            event for event in events if (event.start_time <= horizon) and (event.end_time > now)
        ]
        upcoming.sort(key=lambda e: e.start_time)

        self._publish_user_event(
            "calendar.upcoming",
            validated,
            {"count": len(upcoming), "user_id": validated},
            {
                "count": len(upcoming),
                "user_id": validated,
                "events": [e.to_summary() for e in upcoming],
            },
        )
        return upcoming

    def refresh_upcoming(
        self, *, user_id: str | None = None, account_id: str | None = None
    ) -> list[CalendarEvent]:
        return self.list_upcoming(user_id=user_id, account_id=account_id)

    def get_events(
        self,
        start: datetime,
        end: datetime,
        *,
        user_id: str | None = None,
        account_id: str | None = None,
    ) -> list[CalendarEvent]:
        """
        Get the requesting user's calendar events that overlap a time range.
        """
        validated = require_user_id(user_id)

        start_utc = _ensure_aware_utc(start)
        end_utc = _ensure_aware_utc(end)

        events = self._events_for(validated, account_id)
        result = [e for e in events if e.start_time < end_utc and e.end_time > start_utc]
        result.sort(key=lambda e: e.start_time)

        self._publish_user_event(
            "calendar.range",
            validated,
            {
                "count": len(result),
                "user_id": validated,
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
            },
        )
        return result

    def list_past_events(
        self, *, lookback_hours: int = 72, user_id: str | None = None
    ) -> list[CalendarEvent]:
        """Return the user's events that ended within the lookback window."""
        now = datetime.now(UTC)
        start = now - timedelta(hours=lookback_hours)
        events = self.get_events(start, now, user_id=user_id)
        past_events = [event for event in events if event.end_time <= now]
        past_events.sort(key=lambda e: e.end_time)
        return past_events

    def get_upcoming_events(
        self, days: int = 7, *, user_id: str | None = None
    ) -> list[CalendarEvent]:
        now = datetime.now(UTC)
        end = now + timedelta(days=days)
        return self.get_events(now, end, user_id=user_id)

    def get_all_events(self, *, user_id: str | None = None) -> list[CalendarEvent]:
        validated = require_user_id(user_id)
        return self._events_for(validated)

    def get_past_events(
        self,
        hours: int = 72,
        *,
        now: datetime | None = None,
        user_id: str | None = None,
    ) -> list[CalendarEvent]:
        """Get the user's events that ended within the specified time window.

        Args:
            hours: Look back this many hours for ended events.
            now: Current time (defaults to UTC now).
            user_id: Requesting user (required).

        Returns:
            List of events that ended within the window, sorted by end_time.
        """
        validated = require_user_id(user_id)
        check_time = now or datetime.now(UTC)
        window_start = check_time - timedelta(hours=hours)

        events = self._events_for(validated)
        past_events = [e for e in events if e.end_time <= check_time and e.end_time >= window_start]
        past_events.sort(key=lambda e: e.end_time, reverse=True)
        return past_events

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

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
        user_id: str | None = None,
        account_id: str | None = None,
    ) -> CalendarEvent:
        """
        Create a new calendar event in the requesting user's store.

        Raises:
            CalendarIdentityError: On missing or invalid identity.
            CalendarAccountAccessError: When *account_id* is not available
                to this user (unauthorized or nonexistent).
            IntegrationNotConfiguredError: When accounts are configured but
                none is available to this user (fail closed — never another
                user's account).
        """
        validated = require_user_id(user_id)
        store = self._get_store(validated, account_id)
        if store is None:
            raise IntegrationNotConfiguredError(_NOT_CONFIGURED_FOR_USER_MSG)

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

        store.events.append(event)
        store.events.sort(key=lambda e: e.start_time)
        self._save_store(store)

        self._publish_user_event(
            "calendar.created",
            validated,
            {"user_id": validated, "created": True},
            {"user_id": validated, "created": True, "event": event.to_summary()},
        )
        return event

    def update_event(
        self,
        event_id: str,
        updates: dict[str, Any],
        *,
        user_id: str | None = None,
    ) -> CalendarEvent | None:
        """
        Update an event by id within the requesting user's store only.
        """
        validated = require_user_id(user_id)
        store = self._get_store(validated)
        if store is None:
            logger.warning("Calendar update refused: no account for user %r", validated)
            return None

        for i, event in enumerate(store.events):
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

            store.events[i] = event
            store.events.sort(key=lambda e: e.start_time)
            self._save_store(store)

            self._publish_user_event(
                "calendar.updated",
                validated,
                {"user_id": validated, "updated": True},
                {"user_id": validated, "updated": True, "event": event.to_summary()},
            )
            return event

        logger.warning("Event not found for update: %s", event_id)
        self._publish_user_event(
            "calendar.updated",
            validated,
            {"user_id": validated, "updated": False},
        )
        return None

    def delete_event(self, event_id: str, *, user_id: str | None = None) -> bool:
        """
        Delete an event by id within the requesting user's store only.
        """
        validated = require_user_id(user_id)
        store = self._get_store(validated)
        if store is None:
            logger.warning("Calendar delete refused: no account for user %r", validated)
            return False

        new_events = [e for e in store.events if e.event_id != event_id]
        if len(new_events) == len(store.events):
            logger.warning("Event not found for delete: %s", event_id)
            self._publish_user_event(
                "calendar.deleted",
                validated,
                {"user_id": validated, "deleted": False},
            )
            return False

        store.events[:] = new_events
        self._save_store(store)
        self._publish_user_event(
            "calendar.deleted",
            validated,
            {"user_id": validated, "deleted": True},
            {"user_id": validated, "deleted": True, "event_id": event_id},
        )
        return True

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def detect_conflicts(
        self, event: CalendarEvent, *, user_id: str | None = None
    ) -> list[CalendarEvent]:
        """
        Detect conflicts between the provided event and the user's events.
        """
        validated = require_user_id(user_id)
        conflicts: list[CalendarEvent] = []
        for existing in self._events_for(validated):
            if existing.event_id == event.event_id:
                continue
            if existing.overlaps_with(event):
                conflicts.append(existing)
        return conflicts

    def find_conflicts(
        self,
        events: list[CalendarEvent] | None = None,
        *,
        user_id: str | None = None,
    ) -> list[tuple[CalendarEvent, CalendarEvent]]:
        """
        Find overlapping event pairs within the user's events.
        """
        validated = require_user_id(user_id)
        events_to_check = list(events) if events is not None else self._events_for(validated)
        events_to_check.sort(key=lambda e: e.start_time)

        conflicts: list[tuple[CalendarEvent, CalendarEvent]] = []
        for i, e1 in enumerate(events_to_check):
            for e2 in events_to_check[i + 1 :]:
                if e1.overlaps_with(e2):
                    conflicts.append((e1, e2))
        return conflicts

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

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
                start_dt = datetime.now(UTC)

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

    def _save_store(self, store: _EventStore) -> None:
        """Persist a store to its own path (no-op for in-memory stores).

        The repo seed file is never modified.
        """
        if store.storage_path is None:
            return

        try:
            store.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"events": [e.to_summary() for e in store.events]}
            store.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save calendar data: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Follow-up cues
    # ------------------------------------------------------------------

    def generate_followup_cues(
        self,
        user_id: str,
        *,
        lookback_hours: int = 72,
        expire_hours: int = 168,
        now: datetime | None = None,
    ) -> int:
        """Generate follow-up cues from the user's recent past events.

        Creates cues for events that have ended within the lookback window.
        Skips:
        - Events that already have a cue
        - All-day events that look like holidays
        - Events with 'no-followup' in metadata/description

        Args:
            user_id: User ID whose events are read and whose cues are created.
            lookback_hours: Only consider events ended within this many hours.
            expire_hours: Hours until the cue expires.
            now: Current time (defaults to UTC now).

        Returns:
            Number of cues created.
        """
        validated = require_user_id(user_id)
        try:
            from rex.cue_store import get_cue_store
        except ImportError:
            logger.warning("CueStore not available, cannot generate followup cues")
            return 0

        check_time = now or datetime.now(UTC)
        past_events = self.get_past_events(hours=lookback_hours, now=check_time, user_id=validated)
        cue_store = get_cue_store()

        created_count = 0
        for event in past_events:
            # Skip if cue already exists for this event
            if cue_store.has_cue_for_source(validated, "calendar", event.event_id):
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
                user_id=validated,
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

    The returned service enforces per-user account ownership internally;
    every operation requires a validated ``user_id``.  Accounts come from
    ``calendar.accounts`` plus the legacy global calendar configuration
    (``calendar.backend = "ics"`` / ``calendar.provider``), which is usable
    only by the explicit ``default`` profile.

    Raises:
        IntegrationNotConfiguredError: when no calendar accounts are
            configured at all.
    """
    global _calendar_service
    if _calendar_service is not None:
        return _calendar_service

    if config is not None:
        resolver = CalendarAccountResolver.from_raw_config(config)
        injected: CalendarAccountResolver | None = resolver
    else:
        resolver = CalendarAccountResolver.load()
        # No resolver is injected: the service reloads it when the config
        # file changes, so account revocations take effect in long-lived
        # processes without a restart.
        injected = None

    if not resolver.has_configured_accounts():
        raise IntegrationNotConfiguredError("Calendar: not configured")

    service = CalendarService(event_bus=event_bus, account_resolver=injected)

    # Legacy compatibility: when the default profile resolves to an ICS
    # account, connect it eagerly so misconfigured sources fail fast (the
    # pre-#303 behaviour).
    try:
        default_definition = resolver.resolve_account(DEFAULT_PROFILE)
    except PermissionError:
        default_definition = None
    if default_definition is not None and default_definition.provider == "ics":
        if not service.connect(user_id=DEFAULT_PROFILE):
            raise IntegrationNotConfiguredError("Calendar ICS backend failed to connect")

    _calendar_service = service
    return _calendar_service


def set_calendar_service(service: CalendarService | None) -> None:
    """Set the global calendar service instance (for testing)."""
    global _calendar_service
    _calendar_service = service


__all__ = [
    "CalendarEvent",
    "CalendarService",
    "IntegrationNotConfiguredError",
    "get_calendar_service",
    "set_calendar_service",
]
