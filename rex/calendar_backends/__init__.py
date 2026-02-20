"""Calendar backend adapters.

Provides pluggable calendar sources (stub/mock, ICS file/URL) so the
CalendarService can swap backends transparently, following the same
pattern used by ``rex.email_backends``.
"""

from rex.calendar_backends.base import CalendarBackend

__all__ = ["CalendarBackend"]
