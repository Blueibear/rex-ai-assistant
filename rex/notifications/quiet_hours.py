"""QuietHoursGate: suppress non-exempt notifications during quiet hours.

:class:`QuietHoursGate` reads ``notifications_quiet_hours_start`` and
``notifications_quiet_hours_end`` from ``config/rex_config.json`` (or a
caller-supplied config dict) and implements the :class:`QuietHoursChecker`
protocol expected by :class:`~rex.notifications.router.NotificationRouter`.

Time-range handling
-------------------
Both values are ``"HH:MM"`` strings in 24-hour local time.  The gate handles
ranges that span midnight (e.g. start=23:00, end=07:00) correctly:

- If ``start < end``  the quiet period is a same-day window.
- If ``start >= end`` the quiet period spans midnight (overnight).

When the config keys are absent or malformed, :meth:`is_quiet_now` returns
``False`` (i.e. never suppresses).
"""

from __future__ import annotations

import json
import logging
from datetime import time
from pathlib import Path

from rex.notifications.models import Notification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rex_config.json"


def _load_quiet_hours_config(config: dict[str, object] | None) -> tuple[str, str]:
    """Return (start_str, end_str) from *config* or the default config file.

    Returns (``""``, ``""``) on any error so the gate stays disabled.
    """
    if config is None:
        try:
            raw = _CONFIG_PATH.read_text(encoding="utf-8")
            data: dict[str, object] = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("QuietHoursGate: could not read config: %s", exc)
            return ("", "")
    else:
        data = config

    start = data.get("notifications_quiet_hours_start", "")
    end = data.get("notifications_quiet_hours_end", "")
    return (str(start), str(end))


def _parse_time(value: str) -> time | None:
    """Parse ``"HH:MM"`` string into :class:`datetime.time`, or ``None``."""
    try:
        parts = value.strip().split(":")
        if len(parts) != 2:
            return None
        hour, minute = int(parts[0]), int(parts[1])
        return time(hour, minute)
    except (ValueError, AttributeError):
        return None


def _is_in_quiet_window(current: time, start: time, end: time) -> bool:
    """Return ``True`` if *current* falls within the [start, end) quiet window.

    Handles overnight windows (start >= end) correctly.
    """
    if start < end:
        # Same-day window e.g. 22:00 → 23:30
        return start <= current < end
    elif start > end:
        # Overnight window e.g. 23:00 → 07:00
        return current >= start or current < end
    else:
        # start == end: zero-length window — always quiet (full day suppression)
        return True


# ---------------------------------------------------------------------------
# QuietHoursGate
# ---------------------------------------------------------------------------


class QuietHoursGate:
    """Gate that suppresses non-exempt notifications during quiet hours.

    Implements the :class:`~rex.notifications.router.QuietHoursChecker`
    Protocol so it can be injected into
    :class:`~rex.notifications.router.NotificationRouter`.

    Args:
        config: Optional pre-loaded config ``dict``.  When ``None`` (default)
            the gate reads from ``config/rex_config.json`` at call time.
        clock: Optional callable returning the current :class:`datetime.time`
            in local timezone.  Defaults to ``datetime.now().time()``.
            Useful for unit testing without mocking system time.
    """

    def __init__(
        self,
        config: dict[str, object] | None = None,
        clock: object | None = None,
    ) -> None:
        self._config = config
        # Accept a callable; default to datetime.now().time()
        import datetime as _dt

        self._clock: object = clock if clock is not None else _dt.datetime.now

    def _current_time(self) -> time:
        import datetime as _dt

        if callable(self._clock):
            result = self._clock()
            if isinstance(result, _dt.datetime):
                return result.time()
            if isinstance(result, time):
                return result
        return _dt.datetime.now().time()

    def is_quiet_now(self) -> bool:
        """Return ``True`` if the current local time is within quiet hours.

        Returns ``False`` if quiet hours are not configured or are malformed.
        """
        start_str, end_str = _load_quiet_hours_config(self._config)
        start = _parse_time(start_str)
        end = _parse_time(end_str)
        if start is None or end is None:
            return False
        current = self._current_time()
        return _is_in_quiet_window(current, start, end)

    def should_suppress(self, notification: Notification) -> bool:
        """Return ``True`` if *notification* should be suppressed right now.

        A notification is suppressed when quiet hours are active **and** the
        notification is not exempt (``quiet_hours_exempt=False``).
        """
        return self.is_quiet_now() and not notification.quiet_hours_exempt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "QuietHoursGate",
]
