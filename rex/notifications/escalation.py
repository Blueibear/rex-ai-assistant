"""EscalationEngine: promote unacknowledged notifications to higher priority.

:class:`EscalationEngine` scans all unread notifications in the store,
identifies those whose ``escalation_due_at`` timestamp has passed, promotes
their priority by one level, re-routes them through
:class:`~rex.notifications.router.NotificationRouter`, and reschedules the
next escalation check.

Priority promotion ladder
-------------------------
``low`` → ``medium`` → ``high`` → ``critical``

``critical`` notifications are not escalated further.

Escalation delay
----------------
Reads ``notifications_escalation_delay_minutes`` from ``config/rex_config.json``
(default: 30 minutes) and computes the next ``escalation_due_at`` as
``now + escalation_delay``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rex.notifications.models import NotificationPriority, NotificationStore
from rex.notifications.router import NotificationRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority ladder
# ---------------------------------------------------------------------------

_NEXT_PRIORITY: dict[NotificationPriority, NotificationPriority | None] = {
    "low": "medium",
    "medium": "high",
    "high": "critical",
    "critical": None,  # already at maximum
}

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rex_config.json"
_DEFAULT_DELAY_MINUTES = 30


def _load_escalation_delay(config: dict[str, object] | None) -> int:
    """Return escalation delay in minutes from *config* or rex_config.json."""
    if config is None:
        try:
            raw = _CONFIG_PATH.read_text(encoding="utf-8")
            data: dict[str, object] = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("EscalationEngine: could not read config: %s", exc)
            return _DEFAULT_DELAY_MINUTES
    else:
        data = config

    raw_val = data.get("notifications_escalation_delay_minutes", _DEFAULT_DELAY_MINUTES)
    try:
        return int(str(raw_val))
    except (TypeError, ValueError):
        return _DEFAULT_DELAY_MINUTES


# ---------------------------------------------------------------------------
# EscalationEngine
# ---------------------------------------------------------------------------


class EscalationEngine:
    """Promote unacknowledged notifications that have exceeded their deadline.

    Args:
        store: :class:`~rex.notifications.models.NotificationStore` to read
            and update.
        router: :class:`~rex.notifications.router.NotificationRouter` used to
            re-dispatch escalated notifications at their new priority.
        config: Optional pre-loaded config ``dict``.  When ``None`` the
            engine reads from ``config/rex_config.json`` at call time.
    """

    def __init__(
        self,
        store: NotificationStore,
        router: NotificationRouter,
        config: dict[str, object] | None = None,
    ) -> None:
        self._store = store
        self._router = router
        self._config = config

    def check_escalations(self) -> None:
        """Inspect all unread notifications and escalate overdue ones.

        For each unread notification whose ``escalation_due_at`` is in the
        past:

        1. Promote the priority by one level (``critical`` is skipped).
        2. Re-route through :class:`~rex.notifications.router.NotificationRouter`.
        3. Update ``escalation_due_at`` to ``now + escalation_delay`` and
           persist via :meth:`~NotificationStore.update`.
        """
        now = datetime.now(timezone.utc)
        delay_minutes = _load_escalation_delay(self._config)
        delay = timedelta(minutes=delay_minutes)
        unread = self._store.get_unread()

        for notification in unread:
            if notification.escalation_due_at is None:
                continue
            # Ensure comparison is timezone-aware.
            due = notification.escalation_due_at
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due > now:
                continue

            next_priority = _NEXT_PRIORITY.get(notification.priority)
            if next_priority is None:
                # Already critical — do not escalate further.
                continue

            escalated = notification.model_copy(
                update={
                    "priority": next_priority,
                    "escalation_due_at": now + delay,
                }
            )
            self._router.route(escalated)
            self._store.update(escalated)
            logger.info(
                "Escalated notification %r from %s to %s",
                notification.id,
                notification.priority,
                next_priority,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EscalationEngine",
]
