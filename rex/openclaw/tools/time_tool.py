"""OpenClaw tool adapter — time_now.

Wraps Rex's existing ``time_now`` implementation from :mod:`rex.tool_router`
and exposes it for registration with OpenClaw's tool system.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :func:`time_now` callable works
independently of OpenClaw.

Typical usage::

    from rex.openclaw.tools.time_tool import time_now, register

    result = time_now("Edinburgh, Scotland")
    # {'local_time': '2026-03-22 14:30', 'date': '2026-03-22', 'timezone': 'Europe/London'}

    register()   # no-op if openclaw not installed
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from rex.tool_router import execute_tool

logger = logging.getLogger(__name__)

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw  # type: ignore[import-not-found]
else:
    _openclaw = None  # type: ignore[assignment]

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

    Delegates to :func:`rex.tool_router.execute_tool` so that the existing
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


def register(agent: Any = None) -> Any:
    """Register the ``time_now`` tool with OpenClaw.

    When the ``openclaw`` package is available this function calls
    OpenClaw's tool registration API, passing :func:`time_now` as the
    handler.  When OpenClaw is not installed it logs a warning and returns
    ``None``.

    .. note::
        The exact OpenClaw tool registration call is a stub (see PRD §8.3 —
        *"Confirm OpenClaw's tool registration mechanism"*).  Replace the
        ``# TODO`` line once the API is confirmed.

    Args:
        agent: Optional OpenClaw agent handle.  Some registration APIs
            attach tools to a specific agent; others use a global registry.

    Returns:
        The registration handle returned by OpenClaw, or ``None``.
    """
    if not OPENCLAW_AVAILABLE:
        logger.warning(
            "openclaw package not installed — %s tool not registered with OpenClaw",
            TOOL_NAME,
        )
        return None

    # TODO: replace with real OpenClaw tool registration once API is confirmed.
    # Expected shape (to be verified):
    #   handle = _openclaw.register_tool(
    #       name=TOOL_NAME,
    #       description=TOOL_DESCRIPTION,
    #       handler=time_now,
    #       agent=agent,
    #   )
    #   return handle
    logger.warning(
        "OpenClaw tool registration stub for %s — update once API is confirmed (PRD §8.3)",
        TOOL_NAME,
    )
    return None
