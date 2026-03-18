"""Notification priority levels for the Rex assistant.

Defines the canonical :class:`NotificationPriority` enum used across routing,
delivery, and persistence layers.

Usage example::

    from rex.notification_priority import NotificationPriority

    priority = NotificationPriority.HIGH
    print(priority.value)   # "high"
    print(priority.label)   # "High"
"""

from __future__ import annotations

from enum import Enum


class NotificationPriority(str, Enum):
    """Priority levels for notifications.

    Values are lowercase strings so they can be stored directly in SQLite
    without any conversion and round-trip cleanly through JSON APIs.

    Members:
        critical: Requires immediate action; always bypasses quiet hours and
            digest queuing.
        high: Urgent; delivered immediately but subject to escalation rules.
        medium: Default priority; routed to digest queue unless configured
            otherwise.
        low: Informational; always queued for digest delivery.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def label(self) -> str:
        """Human-readable label (title-cased)."""
        return self.value.title()

    @classmethod
    def _missing_(cls, value: object) -> NotificationPriority:
        """Return MEDIUM for any unknown / legacy value (e.g. 'normal')."""
        return cls.MEDIUM

    @classmethod
    def from_str(cls, value: str | None) -> NotificationPriority:
        """Parse a string to a :class:`NotificationPriority`, defaulting to MEDIUM.

        This is the canonical way to deserialise a priority value read from
        the database or an external source.  Unrecognised values (including
        the legacy ``"normal"`` default) map to :attr:`MEDIUM`.

        Args:
            value: String representation of a priority level, or ``None``.

        Returns:
            The matching :class:`NotificationPriority`, or ``MEDIUM`` as a
            safe fallback.
        """
        if value is None:
            return cls.MEDIUM
        try:
            return cls(value.lower().strip())
        except ValueError:
            return cls.MEDIUM


__all__ = ["NotificationPriority"]
