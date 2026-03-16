"""NotificationRouter: priority-based notification dispatch.

Routing rules
-------------
- ``critical`` / ``high``   — dispatch immediately via desktop notification;
                               store to DB.
- ``medium``                 — dispatch via desktop notification unless quiet
                               hours are active; store to DB.
- ``low``                    — mark ``digest_eligible=True``, store to DB;
                               do not send desktop notification immediately.

Desktop notifications are sent via :func:`_send_desktop`.  The function tries
``plyer`` first, falls back to a no-op log if ``plyer`` is not installed so
the module works without optional dependencies.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from rex.notifications.models import Notification, NotificationStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Desktop notification helper
# ---------------------------------------------------------------------------


def _send_desktop(title: str, body: str) -> None:  # pragma: no cover
    """Send an OS-level desktop notification.

    Uses ``plyer`` when available; falls back to a log message otherwise.
    """
    try:
        from plyer import notification  # type: ignore[import-not-found]

        notification.notify(title=title, message=body, app_name="Rex")
    except Exception as exc:  # noqa: BLE001
        logger.info("Desktop notification (plyer unavailable): %s — %s | %s", title, body, exc)


# ---------------------------------------------------------------------------
# QuietHours protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class QuietHoursChecker(Protocol):
    """Minimal interface for quiet-hours gate injection."""

    def should_suppress(self, notification: Notification) -> bool:
        """Return ``True`` if *notification* should be suppressed right now."""
        ...


class _NeverSuppress:
    """Default no-op quiet-hours checker — never suppresses."""

    def should_suppress(self, notification: Notification) -> bool:  # noqa: ARG002
        return False


# ---------------------------------------------------------------------------
# NotificationRouter
# ---------------------------------------------------------------------------


class NotificationRouter:
    """Route a :class:`~rex.notifications.models.Notification` to the
    appropriate channel based on its priority.

    Args:
        store: :class:`~rex.notifications.models.NotificationStore` instance
            used to persist notifications.
        quiet_hours: Optional :class:`QuietHoursChecker` implementation.
            When ``None`` a no-op checker is used (no suppression).
    """

    def __init__(
        self,
        store: NotificationStore,
        quiet_hours: QuietHoursChecker | None = None,
    ) -> None:
        self._store = store
        self._quiet_hours: QuietHoursChecker = (
            quiet_hours if quiet_hours is not None else _NeverSuppress()
        )

    def route(self, notification: Notification) -> None:
        """Dispatch *notification* according to its priority.

        Side-effects:
            - Calls :func:`_send_desktop` for ``critical``, ``high``, and
              non-suppressed ``medium`` notifications.
            - Sets ``digest_eligible=True`` on ``low`` notifications before
              persisting.
            - Always calls :meth:`~NotificationStore.add` to persist the
              (possibly mutated) notification.
        """
        priority = notification.priority

        if priority in ("critical", "high"):
            _send_desktop(notification.title, notification.body)
            self._store.add(notification)

        elif priority == "medium":
            if not self._quiet_hours.should_suppress(notification):
                _send_desktop(notification.title, notification.body)
            self._store.add(notification)

        else:  # low
            # Mark digest-eligible before persisting.
            updated = notification.model_copy(update={"digest_eligible": True})
            self._store.add(updated)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "NotificationRouter",
    "QuietHoursChecker",
]
