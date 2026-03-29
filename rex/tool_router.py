"""Tool router — dispatches named tool calls to their real handlers.

Only tools listed in EXECUTABLE_TOOLS are accepted.  Any other name raises
UnknownToolError immediately, rather than silently returning None or a
misleading error message.

Full handler implementations for weather_now, web_search, send_email, and
calendar_create_event are added in US-186 and US-187.  This module provides
the dispatch skeleton and the time_now / home_assistant_call_service handlers.
"""

from __future__ import annotations

import logging
from typing import Any

from rex.tool_catalog import EXECUTABLE_TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownToolError(Exception):
    """Raised when a tool name is not in the authoritative catalog.

    This is distinct from a generic KeyError or ValueError so callers can
    handle the "tool does not exist" case explicitly.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"Tool '{tool_name}' is not in EXECUTABLE_TOOLS. "
            f"Known tools: {sorted(EXECUTABLE_TOOLS)}"
        )


# ---------------------------------------------------------------------------
# Internal handler stubs (to be fleshed out in US-186 / US-187)
# ---------------------------------------------------------------------------


def _handle_time_now(args: dict[str, Any]) -> str:
    """Return the current time, optionally for a given timezone."""
    from datetime import datetime, timezone

    location = args.get("location", "local")
    now = datetime.now(tz=timezone.utc)
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')} (requested for: {location})"


def _handle_weather_now(args: dict[str, Any]) -> str:
    """Stub — full implementation in US-186."""
    return "[integration not configured]"


def _handle_web_search(args: dict[str, Any]) -> str:
    """Stub — full implementation in US-186."""
    return "[integration not configured]"


def _handle_send_email(args: dict[str, Any]) -> str:
    """Stub — full implementation in US-187."""
    return "[integration not configured]"


def _handle_calendar_create_event(args: dict[str, Any]) -> str:
    """Stub — full implementation in US-187."""
    return "[integration not configured]"


def _handle_home_assistant_call_service(args: dict[str, Any]) -> str:
    """Call a Home Assistant service endpoint."""
    domain = args.get("domain", "")
    service = args.get("service", "")
    entity_id = args.get("entity_id", "")

    try:
        from rex.config import AppConfig

        cfg = AppConfig()
        ha_url = getattr(cfg, "home_assistant_url", None)
        if not ha_url:
            return "[integration not configured]"
    except Exception:
        return "[integration not configured]"

    try:
        import requests

        from rex.credentials import get_credential_manager

        token = get_credential_manager().get_token("home_assistant")
        if not token:
            return "[integration not configured]"

        url = f"{ha_url.rstrip('/')}/api/services/{domain}/{service}"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"entity_id": entity_id} if entity_id else {},
            timeout=10,
        )
        resp.raise_for_status()
        return f"Home Assistant: {domain}.{service} called on {entity_id}"
    except Exception as exc:
        logger.warning("home_assistant_call_service failed: %s", exc)
        return f"[Home Assistant error: {exc}]"


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {
    "time_now": _handle_time_now,
    "weather_now": _handle_weather_now,
    "web_search": _handle_web_search,
    "send_email": _handle_send_email,
    "calendar_create_event": _handle_calendar_create_event,
    "home_assistant_call_service": _handle_home_assistant_call_service,
}

# Sanity-check at import time: every EXECUTABLE_TOOL must have a handler.
assert set(_HANDLERS) == set(EXECUTABLE_TOOLS), (
    f"Handler table mismatch.\n"
    f"  Missing handlers: {EXECUTABLE_TOOLS - set(_HANDLERS)}\n"
    f"  Extra handlers: {set(_HANDLERS) - EXECUTABLE_TOOLS}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_tool(tool_name: str, args: dict[str, Any] | None = None) -> str:
    """Execute a tool by name and return its string result.

    Args:
        tool_name: Must be a member of EXECUTABLE_TOOLS.
        args: Keyword arguments forwarded to the tool handler.

    Returns:
        A non-empty string result from the handler.

    Raises:
        UnknownToolError: If ``tool_name`` is not in EXECUTABLE_TOOLS.
    """
    if tool_name not in EXECUTABLE_TOOLS:
        raise UnknownToolError(tool_name)

    handler = _HANDLERS[tool_name]
    result: str = handler(args or {})
    logger.debug("execute_tool(%r) → %r", tool_name, result[:80])
    return result


__all__ = [
    "execute_tool",
    "UnknownToolError",
]
