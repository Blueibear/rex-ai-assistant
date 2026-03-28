"""OpenClaw tool adapter — time_now.

Wraps Rex's existing ``time_now`` implementation from :mod:`rex.openclaw.tool_executor`
and exposes it for use from the ToolServer HTTP endpoint (US-007/US-008).

Typical usage::

    from rex.openclaw.tools.time_tool import time_now

    result = time_now("Edinburgh, Scotland")
    # {'local_time': '2026-03-22 14:30', 'date': '2026-03-22', 'timezone': 'Europe/London'}
"""

from __future__ import annotations

import logging
from typing import Any

from rex.openclaw.tool_executor import execute_tool

logger = logging.getLogger(__name__)

#: Tool name used when registering with OpenClaw.
TOOL_NAME = "time_now"

#: Human-readable description forwarded to OpenClaw's tool registry.
TOOL_DESCRIPTION = (
    "Get the current local date and time for a given location. "
    'Args: {"location": "City, Region or Country"}'
)


def time_now(
    location: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the current local time for *location*.

    Delegates to :func:`rex.openclaw.tool_executor.execute_tool` so that the existing
    city-to-timezone lookup, geolocation fallback, and audit logging all apply.

    Args:
        location: City / region string.  When *None*, the context
            ``"location"`` key is used as fallback.
        context: Optional default context dict forwarded to the tool router.
            Recognised keys: ``"location"``, ``"timezone"``.

    Returns:
        A dict with keys ``local_time``, ``date``, and ``timezone`` on
        success, or ``{"error": {...}}`` on failure.
    """
    args: dict[str, Any] = {}
    if location is not None:
        args["location"] = location

    default_context: dict[str, Any] = context or {}

    return execute_tool(
        {"tool": TOOL_NAME, "args": args},
        default_context,
        skip_policy_check=True,
        skip_credential_check=True,
        skip_audit_log=True,
    )
