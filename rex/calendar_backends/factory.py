"""Factory for creating calendar backends from configuration.

Reads the ``calendar`` section of ``config/rex_config.json`` and returns
the appropriate :class:`CalendarBackend` instance.

Configuration examples:

Stub (default — no config needed):

.. code-block:: json

    {}

ICS read-only:

.. code-block:: json

    {
      "calendar": {
        "backend": "ics",
        "ics": {
          "source": "/home/user/cal.ics",
          "url_timeout": 15
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rex.calendar_backends.base import CalendarBackend

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("config/rex_config.json")


def _load_calendar_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``calendar`` section from the config dict or file."""
    if config is not None:
        return config.get("calendar", {})  # type: ignore[no-any-return]

    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / _CONFIG_PATH
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("calendar", {})  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug("Could not load calendar config: %s", exc)

    return {}


def create_calendar_backend(
    config: dict[str, Any] | None = None,
) -> CalendarBackend:
    """Create a CalendarBackend from configuration.

    Args:
        config: Full config dict (with a ``calendar`` key).  If ``None``,
                loads from ``config/rex_config.json``.

    Returns:
        A configured :class:`CalendarBackend` instance.
    """
    cal_config = _load_calendar_config(config)
    backend_name = cal_config.get("backend", "stub").lower().strip()

    if backend_name == "ics":
        return _create_ics_backend(cal_config)

    # Default: stub
    return _create_stub_backend()


def _create_ics_backend(cal_config: dict[str, Any]) -> CalendarBackend:
    from rex.calendar_backends.ics_backend import ICSCalendarBackend

    ics_config = cal_config.get("ics", {})
    source = ics_config.get("source", "")
    if not source:
        logger.warning(
            "calendar.backend is 'ics' but calendar.ics.source is empty; "
            "falling back to stub backend"
        )
        return _create_stub_backend()

    url_timeout = int(ics_config.get("url_timeout", 15))
    return ICSCalendarBackend(source=source, url_timeout=url_timeout)


def _create_stub_backend() -> CalendarBackend:
    from rex.calendar_backends.stub import StubCalendarBackend

    return StubCalendarBackend()


def get_backend_names() -> list[str]:
    """Return list of supported backend names."""
    return ["stub", "ics"]


__all__ = ["create_calendar_backend", "get_backend_names"]
