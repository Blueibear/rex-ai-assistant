"""Stub calendar backend that delegates to the existing mock CalendarService.

This backend preserves the original behaviour: events come from the JSON mock
file (``data/mock_calendar.json``) or from in-memory defaults.  It is selected
when no ``calendar.backend`` is configured (the default).
"""

from __future__ import annotations

from pathlib import Path

from rex.calendar_backends.base import CalendarBackend
from rex.calendar_service import CalendarEvent, CalendarService


class StubCalendarBackend(CalendarBackend):
    """Read events from the existing JSON mock store."""

    def __init__(
        self,
        *,
        mock_data_path: Path | str | None = None,
        mock_events: list[CalendarEvent] | None = None,
    ) -> None:
        self._inner = CalendarService(
            mock_data_path=mock_data_path,
            mock_events=mock_events,
        )
        self._connected = False

    def connect(self) -> bool:
        ok = self._inner.connect()
        self._connected = ok
        return ok

    def fetch_events(self) -> list[CalendarEvent]:
        return self._inner.get_all_events()

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:  # noqa: D401
        return self._connected

    @property
    def backend_name(self) -> str:
        return "stub"

    def test_connection(self) -> tuple[bool, str | None]:
        return True, None


__all__ = ["StubCalendarBackend"]
