"""Base interface for calendar backends.

All calendar backends (stub, ICS) must implement this interface so that
the CalendarService can swap backends transparently.
"""

from __future__ import annotations

import abc

from rex.calendar_service import CalendarEvent


class CalendarBackend(abc.ABC):
    """Abstract base class for read-only calendar backends.

    Concrete implementations:
    - ``StubCalendarBackend``: returns mock/default events (existing behaviour).
    - ``ICSCalendarBackend``: parses events from a local ``.ics`` file or HTTPS URL.
    """

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish a connection / validate the source."""

    @abc.abstractmethod
    def fetch_events(self) -> list[CalendarEvent]:
        """Return all events from this source."""

    def disconnect(self) -> None:  # noqa: B027
        """Release any held resources."""

    @property
    def is_connected(self) -> bool:  # noqa: D401
        """Whether the backend currently holds an active connection."""
        return False

    @property
    def backend_name(self) -> str:
        """Human-readable name for diagnostic output."""
        return self.__class__.__name__

    def test_connection(self) -> tuple[bool, str | None]:
        """Verify configuration without side effects.

        Returns:
            (ok, message) where *message* is ``None`` on success or an error string.
        """
        try:
            ok = self.connect()
            if ok:
                return True, None
            return False, "connect() returned False"
        except Exception as exc:
            return False, str(exc)


__all__ = [
    "CalendarBackend",
]
