"""Read-only ICS feed backend for the CalendarBackend interface (US-210).

Reads calendar events from a local ``.ics`` file or an HTTP/HTTPS URL and
returns upcoming events as plain dicts.  Implements
:class:`~rex.integrations.calendar.backends.base.CalendarBackend`.

Parsing strategy (stdlib-first):
    1. If the ``icalendar`` package is installed, delegate to it for robust
       RFC-5545 parsing.
    2. Otherwise, fall back to a lightweight stdlib-only line-by-line parser
       that handles the common VEVENT fields used in practice.

In both strategies:
    - All datetimes are normalised to UTC.
    - Malformed VEVENT blocks are logged as warnings and skipped.
    - ``create_event()`` raises ``NotImplementedError`` (read-only backend).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from rex.integrations.calendar.backends.base import CalendarBackend

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class ICSFeedBackend(CalendarBackend):
    """Read-only calendar backend that parses ICS data from a file or URL.

    Args:
        source:     Local file path or ``http``/``https`` URL pointing to an
                    ICS feed.
        timeout:    HTTP request timeout in seconds (default: 10).
        http_fetch: Optional callable ``(url: str) -> bytes`` used for HTTP
                    requests.  Inject for unit tests to avoid live network calls.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        http_fetch: Callable[[str], bytes] | None = None,
    ) -> None:
        self._source = str(source)
        self._timeout = timeout
        self._http_fetch = http_fetch

    # ------------------------------------------------------------------
    # CalendarBackend interface
    # ------------------------------------------------------------------

    def get_upcoming(self, days: int = 7) -> list[dict]:
        """Return events starting within the next *days* calendar days.

        Args:
            days: Look-ahead window (default: 7).

        Returns:
            List of event dicts (``id``, ``title``, ``start``, ``end``),
            chronological order.  Empty list on any read/parse error.
        """
        try:
            raw = self._load_ics()
        except Exception as exc:
            logger.error("ICSFeedBackend: failed to load feed from %r: %s", self._source, exc)
            return []

        now = datetime.now(UTC)
        cutoff = now + timedelta(days=days)
        events: list[dict] = []

        for event in self._parse_events(raw):
            try:
                start_dt = event.get("_start_dt")
                if start_dt is None:
                    continue
                if now <= start_dt <= cutoff:
                    events.append(
                        {
                            "id": event["id"],
                            "title": event["title"],
                            "start": event["start"],
                            "end": event["end"],
                        }
                    )
            except Exception as exc:
                logger.warning("ICSFeedBackend: skipping event due to error: %s", exc)

        events.sort(key=lambda e: e["start"])
        return events

    def create_event(self, title: str, start: str, end: str) -> dict:
        """Not supported — ICSFeedBackend is read-only."""
        raise NotImplementedError(
            "ICSFeedBackend is read-only; use a writable backend to create events."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ics(self) -> bytes:
        """Return raw ICS bytes from file or HTTP source."""
        source = self._source
        if source.startswith("http://") or source.startswith("https://"):
            if self._http_fetch is not None:
                return self._http_fetch(source)
            req = urllib.request.Request(source, headers={"User-Agent": "Rex-ICS/1.0"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return cast(bytes, resp.read())
        return Path(source).read_bytes()

    def _parse_events(self, raw: bytes) -> Iterator[dict]:
        """Yield parsed event dicts from raw ICS bytes.

        Tries ``icalendar`` first; falls back to the stdlib parser.
        """
        try:
            from importlib.util import find_spec

            if find_spec("icalendar") is not None:
                yield from _parse_with_icalendar(raw)
                return
        except Exception:
            pass
        yield from _parse_stdlib(raw)


# ---------------------------------------------------------------------------
# icalendar-based parser (preferred)
# ---------------------------------------------------------------------------


def _parse_with_icalendar(raw: bytes) -> Iterator[dict]:
    """Parse ICS using the ``icalendar`` package."""
    import icalendar

    try:
        cal = icalendar.Calendar.from_ical(raw)
    except Exception as exc:
        logger.warning("ICSFeedBackend(icalendar): failed to parse calendar: %s", exc)
        return

    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        try:
            uid = str(component.get("UID", ""))
            summary = str(component.get("SUMMARY", "(no title)"))
            dtstart = component.get("DTSTART")
            dtend = component.get("DTEND")

            if dtstart is None:
                logger.warning("ICSFeedBackend(icalendar): VEVENT missing DTSTART — skipping")
                continue

            start_dt = _to_utc(dtstart.dt)
            end_dt = _to_utc(dtend.dt) if dtend else start_dt + timedelta(hours=1)

            if not uid:
                uid = _synthetic_uid(summary, start_dt)

            yield {
                "id": uid,
                "title": summary,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "_start_dt": start_dt,
            }
        except Exception as exc:
            logger.warning("ICSFeedBackend(icalendar): skipping malformed VEVENT: %s", exc)


# ---------------------------------------------------------------------------
# Stdlib-only line-by-line parser (fallback)
# ---------------------------------------------------------------------------


def _parse_stdlib(raw: bytes) -> Iterator[dict]:
    """Minimal RFC-5545 VEVENT parser using stdlib only."""
    text = raw.decode("utf-8", errors="replace")
    # Unfold continuation lines (RFC 5545 §3.1)
    text = text.replace("\r\n ", "").replace("\r\n\t", "").replace("\n ", "").replace("\n\t", "")

    in_vevent = False
    current: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.upper() == "BEGIN:VEVENT":
            in_vevent = True
            current = {}
            continue
        if line.upper() == "END:VEVENT":
            in_vevent = False
            event = _stdlib_build_event(current)
            if event is not None:
                yield event
            current = {}
            continue
        if not in_vevent:
            continue

        # Split at first colon, ignoring colons in value
        colon = line.find(":")
        if colon == -1:
            continue
        prop_part = line[:colon].upper()
        value = line[colon + 1 :]

        # Strip parameters (e.g. "DTSTART;TZID=America/New_York")
        prop_name = prop_part.split(";")[0]
        current[prop_name] = value

    if in_vevent and current:
        # File truncated inside a VEVENT — skip
        logger.warning("ICSFeedBackend(stdlib): truncated VEVENT block at end of file — skipping")


def _stdlib_build_event(props: dict[str, str]) -> dict | None:
    """Build an event dict from raw VEVENT property strings."""
    try:
        raw_start = props.get("DTSTART") or props.get("DTSTART;VALUE=DATE")
        raw_end = props.get("DTEND") or props.get("DTEND;VALUE=DATE")
        summary = props.get("SUMMARY", "(no title)")
        uid = props.get("UID", "")

        if not raw_start:
            logger.warning("ICSFeedBackend(stdlib): VEVENT missing DTSTART — skipping")
            return None

        start_dt = _parse_ics_datetime(raw_start)
        if start_dt is None:
            logger.warning("ICSFeedBackend(stdlib): unparseable DTSTART %r — skipping", raw_start)
            return None

        if raw_end:
            end_dt = _parse_ics_datetime(raw_end) or (start_dt + timedelta(hours=1))
        else:
            end_dt = start_dt + timedelta(hours=1)

        if not uid:
            uid = _synthetic_uid(summary, start_dt)

        return {
            "id": uid,
            "title": summary,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "_start_dt": start_dt,
        }
    except Exception as exc:
        logger.warning("ICSFeedBackend(stdlib): skipping malformed VEVENT: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _parse_ics_datetime(value: str) -> datetime | None:
    """Parse an ICS DTSTART/DTEND value string to a UTC-aware datetime.

    Handles:
    - ``20260401T100000Z``      — UTC
    - ``20260401T100000``       — assumed UTC (no TZID available in stdlib mode)
    - ``20260401``              — date-only (00:00 UTC)
    """
    value = value.strip()
    formats = [
        ("%Y%m%dT%H%M%SZ", True),  # UTC explicit
        ("%Y%m%dT%H%M%S", False),  # floating (treat as UTC)
        ("%Y%m%d", False),  # date-only
    ]
    for fmt, is_utc in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if is_utc or dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    return None


def _to_utc(dt: object) -> datetime:
    """Normalise a datetime or date object to a UTC-aware datetime."""
    from datetime import date as date_type

    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    if isinstance(dt, date_type):
        return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    raise TypeError(f"Cannot convert {type(dt)} to datetime")


def _synthetic_uid(summary: str, start_dt: datetime) -> str:
    """Generate a deterministic UID when the VEVENT has none."""
    key = f"{summary}|{start_dt.isoformat()}"
    return "synth-" + hashlib.sha1(key.encode()).hexdigest()[:16]


__all__ = ["ICSFeedBackend"]
