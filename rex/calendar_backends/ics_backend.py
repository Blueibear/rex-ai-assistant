"""ICS (iCalendar) read-only calendar backend.

Reads events from a local ``.ics`` file or an HTTPS URL.  Write operations
are not supported — this is a read-only source.

Configuration (in ``config/rex_config.json``):

.. code-block:: json

    {
      "calendar": {
        "backend": "ics",
        "ics": {
          "source": "/path/to/calendar.ics",
          "url_timeout": 15
        }
      }
    }

``source`` may be:
- An absolute or project-relative file path ending in ``.ics``
- An ``https://`` URL pointing to an ICS feed

When ``source`` is a URL, the feed is fetched with ``requests.get``
(already a project dependency) using ``url_timeout`` seconds.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rex.calendar_backends.base import CalendarBackend
from rex.calendar_backends.ics_parser import parse_ics
from rex.calendar_service import CalendarEvent

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15  # seconds


class ICSCalendarBackend(CalendarBackend):
    """Read-only backend that parses events from an ICS source."""

    def __init__(
        self,
        *,
        source: str,
        url_timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        """
        Args:
            source: File path (absolute or relative) or HTTPS URL to an ``.ics`` feed.
            url_timeout: Timeout in seconds for URL fetches.
        """
        self._source = source
        self._url_timeout = url_timeout
        self._events: list[CalendarEvent] | None = None
        self._connected = False

    def connect(self) -> bool:
        """Load and parse the ICS source."""
        try:
            text = self._read_source()
            self._events = parse_ics(text)
            self._connected = True
            logger.info(
                "ICS backend connected: %d event(s) from %s",
                len(self._events),
                self._source_label,
            )
            return True
        except Exception as exc:
            logger.error("ICS backend connect failed: %s", exc)
            self._connected = False
            return False

    def fetch_events(self) -> list[CalendarEvent]:
        """Return all parsed events.

        If not yet connected, attempts to connect first.
        """
        if not self._connected or self._events is None:
            self.connect()
        return list(self._events or [])

    def disconnect(self) -> None:
        self._connected = False
        self._events = None

    @property
    def is_connected(self) -> bool:  # noqa: D401
        return self._connected

    @property
    def backend_name(self) -> str:
        return "ics"

    def test_connection(self) -> tuple[bool, str | None]:
        """Validate the ICS source without storing events."""
        try:
            text = self._read_source()
            events = parse_ics(text)
            return True, f"OK: parsed {len(events)} event(s)"
        except FileNotFoundError:
            return False, f"ICS file not found: {self._source}"
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _is_url(self) -> bool:
        return self._source.startswith("https://") or self._source.startswith("http://")

    @property
    def _source_label(self) -> str:
        if self._is_url:
            return (
                f"URL ({self._source[:60]}...)"
                if len(self._source) > 60
                else f"URL ({self._source})"
            )
        return f"file ({self._source})"

    def _read_source(self) -> str:
        """Read the raw ICS text from the configured source."""
        if self._is_url:
            return self._fetch_url()
        return self._read_file()

    def _read_file(self) -> str:
        path = Path(self._source)
        if not path.is_absolute():
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"ICS file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _fetch_url(self) -> str:
        """Fetch ICS content from an HTTPS URL."""
        import requests

        if not self._source.startswith("https://"):
            raise ValueError(
                f"Only HTTPS URLs are supported for ICS feeds (got {self._source[:30]}...)"
            )

        response = requests.get(
            self._source,
            timeout=self._url_timeout,
            headers={"Accept": "text/calendar, text/plain"},
        )
        response.raise_for_status()
        return response.text


__all__ = ["ICSCalendarBackend"]
