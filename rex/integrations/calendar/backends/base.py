"""Transport-layer ABC for calendar backends.

All calendar backends must implement this interface so that CalendarService can
swap them transparently between stub, ICS-feed, and future OAuth backends.
"""

from __future__ import annotations

import abc


class CalendarBackend(abc.ABC):
    """Abstract base class for calendar read/write backends.

    Methods return plain :class:`dict` objects so that backends remain free of
    higher-level model dependencies.

    Concrete implementations:
    - ``StubCalendarBackend``: in-memory stub for offline development/testing.
    - ``ICSFeedBackend``: read-only ICS file or URL feed (US-210).
    """

    @abc.abstractmethod
    def get_upcoming(self, days: int = 7) -> list[dict]:
        """Return events occurring within the next *days* calendar days.

        Each dict must include at minimum:
        ``id``, ``title``, ``start``, ``end``.

        Args:
            days: Look-ahead window in calendar days.

        Returns:
            List of event dicts in chronological order.
        """

    @abc.abstractmethod
    def create_event(self, title: str, start: str, end: str) -> dict:
        """Create a new calendar event.

        Args:
            title: Event title / summary.
            start: ISO-8601 start datetime string (with timezone offset).
            end:   ISO-8601 end datetime string (with timezone offset).

        Returns:
            Dict representing the created event, including at minimum ``id``.

        Raises:
            Exception: On creation failure.
        """


__all__ = ["CalendarBackend"]
