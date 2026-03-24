"""OpenClaw tool adapter — weather_now.

Wraps Rex's existing ``weather_now`` implementation from :mod:`rex.openclaw.tool_executor`
and exposes it for use from the ToolServer HTTP endpoint (US-007/US-008).

Requires ``OPENWEATHERMAP_API_KEY`` to be set in the environment.

Typical usage::

    from rex.openclaw.tools.weather_tool import weather_now

    result = weather_now("Edinburgh, Scotland")
    # {'temperature': 12.3, 'description': 'cloudy', ...}
"""

from __future__ import annotations

import logging
from typing import Any

from rex.openclaw.tool_executor import execute_tool

logger = logging.getLogger(__name__)

#: Tool name used when registering with OpenClaw.
TOOL_NAME = "weather_now"

#: Human-readable description forwarded to OpenClaw's tool registry.
TOOL_DESCRIPTION = (
    "Get the current weather conditions for a given location. "
    'Requires OPENWEATHERMAP_API_KEY. Args: {"location": "City, Region or Country"}'
)


def weather_now(
    location: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return current weather conditions for *location*.

    Delegates to :func:`rex.openclaw.tool_executor.execute_tool` so that the existing
    geolocation fallback and error handling all apply.

    Args:
        location: City / region string.  When *None*, the context
            ``"location"`` key and geolocation cache are used as fallbacks.
        context: Optional default context dict forwarded to the tool router.
            Recognised keys: ``"location"``, ``"timezone"``.

    Returns:
        A dict with weather fields (temperature, description, etc.) on
        success, or ``{"error": {...}}`` on failure (including when the API
        key is not configured).
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


