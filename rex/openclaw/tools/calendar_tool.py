"""OpenClaw tool adapter — calendar_create.

Wraps Rex's existing ``CalendarService.create_event()`` from
:mod:`rex.calendar_service` and exposes it for registration with OpenClaw's
tool system.

This is a *policy-gated* tool: in normal operation the policy engine
requires approval before the event is created (MEDIUM risk).  The callable
itself does not enforce policy — that is the caller's responsibility.

Per-user isolation (issue #303): the dispatcher-injected ``_user_id`` is
required and validated **before** any account or service resolution; missing
or invalid identity fails closed.  The event is created only in the
requesting user's own calendar store.

Typical usage::

    from rex.openclaw.tools.calendar_tool import calendar_create

    result = calendar_create(
        title="Team standup",
        start_time="2026-03-23T09:00:00",
        end_time="2026-03-23T09:30:00",
        _user_id="alice",
    )
    # {'ok': True, 'event_id': '...', 'title': 'Team standup'}
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from rex.assistant_errors import IntegrationNotConfiguredError
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
        dt = dt.replace(tzinfo=UTC)
    return dt


def calendar_create(
    title: str = "",
    start_time: str | datetime = "",
    end_time: str | datetime = "",
    location: str | None = None,
    description: str | None = None,
    context: dict[str, Any] | None = None,
    account_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a calendar event via Rex's CalendarService as the dispatching user.

    Delegates to :func:`rex.calendar_service.get_calendar_service().create_event`.
    The dispatcher-injected ``_user_id`` is required and validated **before**
    any account or credential resolution; missing or invalid identity fails
    closed.  An optional ``account_id`` routes through a specific account
    only after ownership validation — a named user can never fall back to
    the global default or first configured account.

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
        account_id:  Optional explicit account to create in (must belong to
            the requesting user).
        **kwargs:    Absorbs dispatcher-injected keys such as ``transcript``
            and ``_user_id`` without raising TypeError.

    Returns:
        A dict with keys ``ok`` (bool), and on success ``event_id`` (str)
        and ``title`` (str); on refusal ``error`` (str).
    """
    from rex.calendar_accounts import require_user_id

    try:
        user_id = require_user_id(kwargs.get("_user_id"))
    except PermissionError as exc:
        logger.warning("calendar_create tool refused: %s", exc)
        return {
            "ok": False,
            "error": "calendar_create requires a valid user identity",
        }

    service = _get_calendar_service()
    start_dt = _parse_dt(start_time)
    end_dt = _parse_dt(end_time)

    # Audit metadata: requesting user and selected account only — never
    # event bodies, attendee lists, or credential references.
    logger.info("calendar_create tool: user=%s account=%s", user_id, account_id or "(default)")
    try:
        event = service.create_event(
            title=title,
            start_time=start_dt,
            end_time=end_dt,
            location=location,
            description=description,
            user_id=user_id,
            account_id=account_id,
        )
    except PermissionError as exc:
        logger.warning("calendar_create tool refused for user=%s: %s", user_id, exc)
        return {"ok": False, "error": str(exc)}
    except IntegrationNotConfiguredError:
        logger.warning("calendar_create tool: no calendar account for user=%s", user_id)
        return {"ok": False, "error": "No calendar account is available for this user"}

    return {
        "ok": True,
        "event_id": event.event_id,
        "title": event.title,
    }
