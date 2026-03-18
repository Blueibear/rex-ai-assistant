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
import socket
from ipaddress import ip_address
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from rex.calendar_backends.base import CalendarBackend
from rex.calendar_backends.ics_parser import parse_ics
from rex.calendar_service import CalendarEvent

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15  # seconds
_ALLOWED_URL_SCHEMES = {"https"}


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
        if _looks_like_windows_path(self._source):
            return False
        parsed = urlparse(self._source)
        return bool(parsed.scheme)

    @property
    def _source_label(self) -> str:
        if self._is_url:
            safe = _sanitize_url_for_logs(self._source)
            return f"URL ({safe})"
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

        parsed = urlparse(self._source)
        scheme = (parsed.scheme or "").lower()
        if scheme not in _ALLOWED_URL_SCHEMES:
            raise ValueError(f"Only HTTPS URLs are supported for ICS feeds (got scheme: {scheme})")
        _validate_remote_host(parsed.hostname)

        response = requests.get(
            self._source,
            timeout=self._url_timeout,
            headers={"Accept": "text/calendar, text/plain"},
        )
        response.raise_for_status()
        return cast(str, response.text)


def _looks_like_windows_path(source: str) -> bool:
    return len(source) >= 3 and source[1] == ":" and source[2] in {"\\", "/"}


def _sanitize_url_for_logs(source: str) -> str:
    parsed = urlparse(source)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    return f"{parsed.scheme}://{host}{port}{path}"


def _validate_remote_host(hostname: str | None) -> None:
    if not hostname:
        raise ValueError("ICS URL is missing a hostname")

    lowered = hostname.strip().lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        raise ValueError("ICS URL host must not be localhost or local network")

    try:
        addresses = {ai[4][0] for ai in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve ICS URL host: {hostname}") from exc

    for addr in addresses:
        ip = ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("ICS URL host resolves to a local or reserved address")


__all__ = ["ICSCalendarBackend"]
