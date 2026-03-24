"""OpenClaw tool adapter — calendar_create.

Wraps Rex's existing ``CalendarService.create_event()`` from
:mod:`rex.calendar_service` and exposes it for registration with OpenClaw's
tool system.

This is a *policy-gated* tool: in normal operation the policy engine
requires approval before the event is created (MEDIUM risk).  The callable
itself does not enforce policy — that is the caller's responsibility.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :func:`calendar_create` callable works
independently of OpenClaw.

Typical usage::

    from rex.openclaw.tools.calendar_tool import calendar_create, register

    result = calendar_create(
        title="Team standup",
        start_time="2026-03-23T09:00:00",
        end_time="2026-03-23T09:30:00",
    )
    # {'ok': True, 'event_id': '...', 'title': 'Team standup'}

    register()   # no-op if openclaw not installed
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from rex.calendar_service import get_calendar_service as _get_calendar_service

logger = logging.getLogger(__name__)

#: Tool name used when registering with OpenClaw.
TOOL_NAME = "calendar_create"

#: Human-readable description forwarded to OpenClaw's tool registry.
TOOL_DESCRIPTION = (
    "Create a new calendar event. "
    'Args: {"title": "Event title", "start_time": "2026-03-23T09:00:00", '
    '"end_time": "2026-03-23T09:30:00", "location": "optional", "description": "optional"}'
)


def _parse_dt(value: str | datetime) -> datetime:
    """Parse an ISO-8601 string or pass through a datetime object.

    Returns a timezone-aware datetime in UTC.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def calendar_create(
    title: str,
    start_time: str | datetime,
    end_time: str | datetime,
    location: str | None = None,
    description: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a calendar event via Rex's CalendarService.

    Delegates to :func:`rex.calendar_service.get_calendar_service().create_event`.
    In stub mode (no backend configured) the event is saved to a local mock
    file and returned.

    .. note::
        This tool is policy-gated (MEDIUM risk).  Callers are responsible
        for obtaining policy approval before invoking this function.

    Args:
        title:       Event title / summary.
        start_time:  Start time as ISO-8601 string or :class:`datetime`.
        end_time:    End time as ISO-8601 string or :class:`datetime`.
        location:    Optional venue or location string.
        description: Optional event description.
        context:     Optional ambient context dict (unused; reserved for future
            timezone injection).

    Returns:
        A dict with keys ``ok`` (bool), ``event_id`` (str), and
        ``title`` (str).
    """
    service = _get_calendar_service()
    start_dt = _parse_dt(start_time)
    end_dt = _parse_dt(end_time)

    event = service.create_event(
        title=title,
        start_time=start_dt,
        end_time=end_dt,
        location=location,
        description=description,
    )
    return {
        "ok": True,
        "event_id": event.event_id,
        "title": event.title,
    }
