"""CalendarService: credential-ready stub/live calendar access.

Without credentials (``calendar_provider == "none"``), all methods return
realistic mock data so the GUI and autonomy engine work out of the box.

When ``calendar_provider == "google"``, the service connects to the Google
Calendar API using OAuth tokens stored in the environment
(``GOOGLE_CALENDAR_ACCESS_TOKEN``).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from rex.integrations.models import CalendarEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub data helpers
# ---------------------------------------------------------------------------


def _days_from_now(days: float, hour: int = 9, minute: int = 0) -> datetime:
    base = datetime.now(timezone.utc).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return base + timedelta(days=days)


def _build_stub_events() -> list[CalendarEvent]:
    return [
        CalendarEvent(
            id="stub-cal-001",
            title="Team standup",
            start=_days_from_now(0, hour=9),
            end=_days_from_now(0, hour=9, minute=30),
            location="https://meet.example.com/standup",
            description="Daily team check-in",
            attendees=["alice@example.com", "bob@example.com"],
            source="stub",
            is_all_day=False,
        ),
        CalendarEvent(
            id="stub-cal-002",
            title="Product roadmap review",
            start=_days_from_now(1, hour=14),
            end=_days_from_now(1, hour=15),
            location="Conference Room B",
            description="Q3 roadmap priorities",
            attendees=["alice@example.com", "ceo@example.com"],
            source="stub",
            is_all_day=False,
        ),
        CalendarEvent(
            id="stub-cal-003",
            title="1:1 with manager",
            start=_days_from_now(2, hour=11),
            end=_days_from_now(2, hour=11, minute=30),
            location=None,
            description="Weekly 1:1 check-in",
            attendees=["manager@example.com"],
            source="stub",
            is_all_day=False,
            recurrence="RRULE:FREQ=WEEKLY;BYDAY=WE",
        ),
        CalendarEvent(
            id="stub-cal-004",
            title="Company all-hands",
            start=_days_from_now(3, hour=16),
            end=_days_from_now(3, hour=17),
            location="Main auditorium",
            description="Monthly company-wide update",
            attendees=[],
            source="stub",
            is_all_day=False,
        ),
        CalendarEvent(
            id="stub-cal-005",
            title="Off-site team day",
            start=_days_from_now(7, hour=8),
            end=_days_from_now(7, hour=18),
            location="Downtown conference center",
            description="Annual off-site planning day",
            attendees=["alice@example.com", "bob@example.com", "ceo@example.com"],
            source="stub",
            is_all_day=True,
        ),
        CalendarEvent(
            id="stub-cal-006",
            title="Sprint retrospective",
            start=_days_from_now(8, hour=15),
            end=_days_from_now(8, hour=16),
            location="https://meet.example.com/retro",
            description="End-of-sprint retrospective",
            attendees=["alice@example.com", "bob@example.com"],
            source="stub",
            is_all_day=False,
        ),
        CalendarEvent(
            id="stub-cal-007",
            title="Doctor appointment",
            start=_days_from_now(10, hour=10),
            end=_days_from_now(10, hour=11),
            location="City Medical Center",
            description=None,
            attendees=[],
            source="stub",
            is_all_day=False,
        ),
        CalendarEvent(
            id="stub-cal-008",
            title="Client demo",
            start=_days_from_now(12, hour=13),
            end=_days_from_now(12, hour=14),
            location="https://meet.example.com/demo",
            description="Quarterly product demo for Acme Corp",
            attendees=["contact@acme.example.com"],
            source="stub",
            is_all_day=False,
        ),
    ]


# ---------------------------------------------------------------------------
# CalendarService
# ---------------------------------------------------------------------------

# Minimal event input type accepted by create_event / update_event.
EventData = dict[str, Any]


class CalendarService:
    """Unified calendar access layer with stub and Google Calendar backends.

    Args:
        calendar_provider: One of ``"none"`` or ``"google"``.
            Defaults to ``"none"`` (stub mode).
    """

    def __init__(self, calendar_provider: str = "none") -> None:
        self._provider = calendar_provider.lower()
        logger.debug("CalendarService initialised with provider=%s", self._provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Return events that overlap the [start, end) window.

        Args:
            start: Window start (UTC).
            end: Window end (UTC).

        Returns:
            List of :class:`~rex.integrations.models.CalendarEvent` objects.
        """
        if self._provider == "google":
            return self._google_get_events(start, end)
        events = _build_stub_events()
        return [e for e in events if e.start < end and e.end > start]

    def create_event(self, event_data: EventData) -> CalendarEvent:
        """Create a new calendar event.

        Args:
            event_data: Dict with at minimum ``title``, ``start``, ``end``.

        Returns:
            The created :class:`~rex.integrations.models.CalendarEvent`.
        """
        if self._provider == "google":
            return self._google_create_event(event_data)
        return self._event_from_data(event_data, source="stub")

    def update_event(self, id: str, event_data: EventData) -> CalendarEvent:  # noqa: A002
        """Update an existing calendar event.

        Args:
            id: Event identifier.
            event_data: Fields to update.

        Returns:
            The updated :class:`~rex.integrations.models.CalendarEvent`.
        """
        if self._provider == "google":
            return self._google_update_event(id, event_data)
        merged = {"id": id, **event_data}
        return self._event_from_data(merged, source="stub")

    def delete_event(self, id: str) -> None:  # noqa: A002
        """Delete a calendar event.

        In stub mode this is a no-op.

        Args:
            id: Event identifier.
        """
        if self._provider == "google":
            self._google_delete_event(id)
            return
        logger.debug("Stub delete_event called for id=%s", id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _event_from_data(self, data: EventData, *, source: str = "rex") -> CalendarEvent:
        """Build a CalendarEvent from a raw dict, supplying defaults."""
        start = data.get("start")
        end = data.get("end")

        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if not isinstance(start, datetime):
            start = datetime.now(timezone.utc)

        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        if not isinstance(end, datetime):
            end = start + timedelta(hours=1)

        return CalendarEvent(
            id=str(data.get("id", uuid.uuid4().hex)),
            title=str(data.get("title", "")),
            start=start,
            end=end,
            location=str(data["location"]) if data.get("location") else None,
            description=str(data["description"]) if data.get("description") else None,
            attendees=list(data.get("attendees", [])),
            source=source,
            is_all_day=bool(data.get("is_all_day", False)),
            recurrence=str(data["recurrence"]) if data.get("recurrence") else None,
        )

    # ------------------------------------------------------------------
    # Google Calendar live backend
    # ------------------------------------------------------------------

    def _google_headers(self) -> dict[str, str]:
        import os

        token = os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN", "")
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def _google_get_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        try:
            import json
            import urllib.parse
            import urllib.request

            time_min = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            time_max = end.strftime("%Y-%m-%dT%H:%M:%SZ")
            params = urllib.parse.urlencode(
                {
                    "timeMin": time_min,
                    "timeMax": time_max,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": 50,
                }
            )
            url = "https://www.googleapis.com/calendar/v3/calendars/primary/events?" + params
            req = urllib.request.Request(url, headers=self._google_headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            parsed = [self._parse_google_event(item) for item in data.get("items", [])]
            return [e for e in parsed if e is not None and isinstance(e, CalendarEvent)]
        except Exception as exc:  # noqa: BLE001
            logger.error("Google Calendar get_events failed: %s", exc)
            events = _build_stub_events()
            return [e for e in events if e.start < end and e.end > start]

    def _google_create_event(self, event_data: EventData) -> CalendarEvent:
        try:
            import json
            import urllib.request

            body = self._to_google_event_body(event_data)
            payload = json.dumps(body).encode()
            headers = {**self._google_headers(), "Content-Type": "application/json"}
            req = urllib.request.Request(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            event = self._parse_google_event(data)
            if event is not None:
                return event
            return self._event_from_data(event_data, source="google")
        except Exception as exc:  # noqa: BLE001
            logger.error("Google Calendar create_event failed: %s", exc)
            return self._event_from_data(event_data, source="stub")

    def _google_update_event(self, id: str, event_data: EventData) -> CalendarEvent:  # noqa: A002
        try:
            import json
            import urllib.request

            body = self._to_google_event_body(event_data)
            payload = json.dumps(body).encode()
            url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{id}"
            headers = {**self._google_headers(), "Content-Type": "application/json"}
            req = urllib.request.Request(url, data=payload, headers=headers, method="PATCH")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            event = self._parse_google_event(data)
            if event is not None:
                return event
            return self._event_from_data({"id": id, **event_data}, source="google")
        except Exception as exc:  # noqa: BLE001
            logger.error("Google Calendar update_event failed: %s", exc)
            return self._event_from_data({"id": id, **event_data}, source="stub")

    def _google_delete_event(self, id: str) -> None:  # noqa: A002
        try:
            import urllib.request

            url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{id}"
            req = urllib.request.Request(url, headers=self._google_headers(), method="DELETE")
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as exc:  # noqa: BLE001
            logger.error("Google Calendar delete_event failed: %s", exc)

    def _to_google_event_body(self, event_data: EventData) -> dict[str, object]:
        """Convert an EventData dict to a Google Calendar API event body."""
        start = event_data.get("start")
        end = event_data.get("end")

        if isinstance(start, datetime):
            start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start_str = str(start)

        if isinstance(end, datetime):
            end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end_str = str(end)

        body: dict[str, object] = {
            "summary": str(event_data.get("title", "")),
            "start": {"dateTime": start_str, "timeZone": "UTC"},
            "end": {"dateTime": end_str, "timeZone": "UTC"},
        }
        if event_data.get("location"):
            body["location"] = str(event_data["location"])
        if event_data.get("description"):
            body["description"] = str(event_data["description"])
        if event_data.get("attendees"):
            attendees = event_data["attendees"]
            if isinstance(attendees, list):
                body["attendees"] = [{"email": str(a)} for a in attendees]
        return body

    def _parse_google_event(self, data: dict[str, object]) -> CalendarEvent | None:
        """Convert a raw Google Calendar API event dict to a CalendarEvent."""
        try:
            event_id = str(data.get("id", uuid.uuid4().hex))
            title = str(data.get("summary", "(no title)"))

            start_raw = data.get("start", {})
            assert isinstance(start_raw, dict)
            end_raw = data.get("end", {})
            assert isinstance(end_raw, dict)

            is_all_day = "date" in start_raw and "dateTime" not in start_raw

            if is_all_day:
                start_str = str(start_raw.get("date", ""))
                end_str = str(end_raw.get("date", ""))
                start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
                end = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)
            else:
                start_str = str(start_raw.get("dateTime", ""))
                end_str = str(end_raw.get("dateTime", ""))
                start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

            location_raw = data.get("location")
            location = str(location_raw) if location_raw is not None else None

            description_raw = data.get("description")
            description = str(description_raw) if description_raw is not None else None

            attendees_raw = data.get("attendees", [])
            assert isinstance(attendees_raw, list)
            attendees = [str(a.get("email", "")) for a in attendees_raw if isinstance(a, dict)]

            recurrence_raw = data.get("recurrence", [])
            assert isinstance(recurrence_raw, list)
            recurrence = str(recurrence_raw[0]) if recurrence_raw else None

            return CalendarEvent(
                id=event_id,
                title=title,
                start=start,
                end=end,
                location=location,
                description=description,
                attendees=attendees,
                source="google",
                is_all_day=is_all_day,
                recurrence=recurrence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse Google Calendar event: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["CalendarService"]
