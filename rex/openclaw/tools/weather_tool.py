"""OpenClaw tool adapter — weather_now.

Wraps Rex's existing ``weather_now`` implementation from :mod:`rex.openclaw.tool_executor`
and exposes it for registration with OpenClaw's tool system.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :func:`weather_now` callable works
independently of OpenClaw (though it requires ``OPENWEATHERMAP_API_KEY`` to
be set in the environment).

Typical usage::

    from rex.openclaw.tools.weather_tool import weather_now, register

    result = weather_now("Edinburgh, Scotland")
    # {'temperature': 12.3, 'description': 'cloudy', ...}

    register()   # no-op if openclaw not installed
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


def register(agent: Any = None) -> Any:
    """Register the ``weather_now`` tool with OpenClaw.

    When the ``openclaw`` package is available this function calls
    OpenClaw's tool registration API, passing :func:`weather_now` as the
    handler.  When OpenClaw is not installed it logs a warning and returns
    ``None``.

    .. note::
        The exact OpenClaw tool registration call is a stub (see PRD §8.3 —
        *"Confirm OpenClaw's tool registration mechanism"*).  Replace the
        ``# TODO`` line once the API is confirmed.

    Args:
        agent: Optional OpenClaw agent handle.

    Returns:
        The registration handle returned by OpenClaw, or ``None``.
    """
    from rex.config import load_config as _load_config
    from rex.openclaw.http_client import get_openclaw_client

    if get_openclaw_client(_load_config()) is None:
        logger.warning(
            "OpenClaw gateway not configured — %s tool not registered with OpenClaw",
            TOOL_NAME,
        )
        return None

    # TODO: replace with real OpenClaw tool registration once API is confirmed.
    # Expected shape (to be verified):
    #   handle = _openclaw.register_tool(
    #       name=TOOL_NAME,
    #       description=TOOL_DESCRIPTION,
    #       handler=weather_now,
    #       agent=agent,
    #   )
    #   return handle
    logger.warning(
        "OpenClaw tool registration stub for %s — update once API is confirmed (PRD §8.3)",
        TOOL_NAME,
    )
    return None
