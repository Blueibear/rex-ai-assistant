"""Minimal stdlib-only ICS (iCalendar RFC 5545) parser.

Extracts VEVENT components from ``.ics`` text and returns
:class:`~rex.calendar_service.CalendarEvent` instances.

Only the fields needed by Rex are extracted:
    SUMMARY, DTSTART, DTEND, LOCATION, DESCRIPTION, UID, ATTENDEE.

This avoids adding a third-party ``icalendar`` dependency while keeping
the calendar backend lightweight and pure-Python.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime, timedelta

from rex.calendar_service import CalendarEvent

logger = logging.getLogger(__name__)

# Date/time formats used in ICS
_DT_FORMAT_BASIC = "%Y%m%dT%H%M%S"
_DT_FORMAT_UTC = "%Y%m%dT%H%M%SZ"
_DATE_FORMAT = "%Y%m%d"


def _unfold(text: str) -> str:
    """Unfold long lines per RFC 5545 Section 3.1.

    Lines starting with a space or tab are continuations of the previous line.
    """
    return re.sub(r"\r?\n[ \t]", "", text)


def _parse_dt(value: str) -> tuple[datetime, bool]:
    """Parse an ICS date or datetime value.

    Returns:
        (datetime, is_all_day) where *is_all_day* is True for DATE values.
    """
    value = value.strip()

    # UTC datetime: 20260220T140000Z
    if value.endswith("Z"):
        return datetime.strptime(value, _DT_FORMAT_UTC).replace(tzinfo=UTC), False

    # Local datetime: 20260220T140000
    if "T" in value:
        dt = datetime.strptime(value, _DT_FORMAT_BASIC)
        # Treat naive datetimes as UTC (safe default)
        return dt.replace(tzinfo=UTC), False

    # Date only: 20260220
    if len(value) == 8 and value.isdigit():
        dt = datetime.strptime(value, _DATE_FORMAT).replace(tzinfo=UTC)
        return dt, True

    raise ValueError(f"Unrecognised ICS date/time value: {value!r}")


def _extract_value(line: str) -> str:
    """Extract the value portion after the property name and any parameters.

    ``DTSTART;VALUE=DATE:20260220`` -> ``20260220``
    ``SUMMARY:Team Meeting``        -> ``Team Meeting``
    """
    _, _, value = line.partition(":")
    return value.strip()


def _extract_dt_value(line: str) -> str:
    """Like ``_extract_value`` but aware of TZID parameters.

    ``DTSTART;TZID=America/New_York:20260220T090000`` -> ``20260220T090000``
    """
    return _extract_value(line)


def _unescape(text: str) -> str:
    """Unescape ICS text values (RFC 5545 Section 3.3.11)."""
    return (
        text.replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def parse_ics(text: str) -> list[CalendarEvent]:
    """Parse ICS text and return a list of CalendarEvents.

    Only VEVENT components are extracted.  Recurring events (RRULE) are not
    expanded — each VEVENT becomes exactly one CalendarEvent.

    Args:
        text: Raw ``.ics`` file content.

    Returns:
        List of parsed events, sorted by start_time.
    """
    text = _unfold(text)
    lines = text.splitlines()

    events: list[CalendarEvent] = []
    in_vevent = False
    current: dict[str, str | list[str]] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped == "BEGIN:VEVENT":
            in_vevent = True
            current = {"attendees": []}
            continue

        if stripped == "END:VEVENT":
            in_vevent = False
            event = _build_event(current)
            if event is not None:
                events.append(event)
            continue

        if not in_vevent:
            continue

        upper = stripped.upper()
        if upper.startswith("SUMMARY"):
            current["summary"] = _unescape(_extract_value(stripped))
        elif upper.startswith("DTSTART"):
            current["dtstart"] = _extract_dt_value(stripped)
        elif upper.startswith("DTEND"):
            current["dtend"] = _extract_dt_value(stripped)
        elif upper.startswith("DURATION"):
            current["duration"] = _extract_value(stripped)
        elif upper.startswith("LOCATION"):
            current["location"] = _unescape(_extract_value(stripped))
        elif upper.startswith("DESCRIPTION"):
            current["description"] = _unescape(_extract_value(stripped))
        elif upper.startswith("UID"):
            current["uid"] = _extract_value(stripped)
        elif upper.startswith("ATTENDEE"):
            # ATTENDEE;CN=Alice:mailto:alice@example.com
            value = _extract_value(stripped)
            if value.lower().startswith("mailto:"):
                value = value[7:]
            attendees = current.get("attendees")
            if isinstance(attendees, list):
                attendees.append(value)

    events.sort(key=lambda e: e.start_time)
    return events


def _parse_duration(dur: str) -> timedelta | None:
    """Parse an ICS DURATION value (subset of RFC 5545).

    Supports: PT1H, PT30M, P1D, PT1H30M, P1DT2H, etc.
    """
    m = re.match(
        r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$",
        dur.strip(),
    )
    if not m:
        return None
    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    minutes = int(m.group(3) or 0)
    seconds = int(m.group(4) or 0)
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _build_event(raw: dict) -> CalendarEvent | None:
    """Convert parsed ICS property dict into a CalendarEvent."""
    summary = raw.get("summary")
    dtstart_str = raw.get("dtstart")

    if not summary or not dtstart_str:
        logger.debug("Skipping VEVENT with missing SUMMARY or DTSTART")
        return None

    try:
        start_dt, is_all_day = _parse_dt(str(dtstart_str))
    except ValueError as exc:
        logger.warning("Skipping VEVENT: bad DTSTART: %s", exc)
        return None

    # Determine end time
    dtend_str = raw.get("dtend")
    duration_str = raw.get("duration")
    end_dt: datetime

    if dtend_str:
        try:
            end_dt, _ = _parse_dt(str(dtend_str))
        except ValueError:
            end_dt = start_dt + timedelta(hours=1)
    elif duration_str:
        delta = _parse_duration(str(duration_str))
        end_dt = start_dt + (delta if delta else timedelta(hours=1))
    elif is_all_day:
        end_dt = start_dt + timedelta(days=1)
    else:
        end_dt = start_dt + timedelta(hours=1)

    uid = str(raw.get("uid", "")) or str(uuid.uuid4())
    attendees = raw.get("attendees", [])
    if not isinstance(attendees, list):
        attendees = []

    return CalendarEvent(
        event_id=uid,
        title=str(summary),
        start_time=start_dt,
        end_time=end_dt,
        location=raw.get("location") if isinstance(raw.get("location"), str) else None,
        description=raw.get("description") if isinstance(raw.get("description"), str) else None,
        attendees=[str(a) for a in attendees],
        all_day=is_all_day,
    )


__all__ = ["parse_ics"]
