"""OpenClaw tool adapter — home_assistant_call_service.

Wraps Rex's :class:`~rex.ha_bridge.HABridge` to expose Home Assistant
service calls for registration with OpenClaw's tool system.

This is a *policy-gated* tool: in normal operation the policy engine
requires approval before the service call is attempted (MEDIUM risk).  The
callable itself does not enforce policy — that is the caller's
responsibility.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :func:`ha_call_service` callable works
independently of OpenClaw.

Typical usage::

    from rex.openclaw.tools.ha_tool import ha_call_service, register

    result = ha_call_service("light", "turn_on", "light.living_room")
    # {'success': True, 'message': 'Light.turn_on light.living_room.', 'entity_id': 'light.living_room'}

    register()   # no-op if openclaw not installed
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from rex.ha_bridge import HABridge as _HABridge
from rex.ha_bridge import IntentMatch as _IntentMatch

logger = logging.getLogger(__name__)

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw  # type: ignore[import-not-found]
else:
    _openclaw = None  # type: ignore[assignment]

#: Tool name used when registering with OpenClaw.
TOOL_NAME = "home_assistant_call_service"

#: Human-readable description forwarded to OpenClaw's tool registry.
TOOL_DESCRIPTION = (
    "Call a Home Assistant service to control smart home devices. "
    'Args: {"domain": "light", "service": "turn_on", "entity_id": "light.living_room", '
    '"data": {"brightness_pct": 80}}'
)


def _get_ha_bridge() -> _HABridge:
    """Return a new HABridge instance (injectable in tests)."""
    return _HABridge()


def ha_call_service(
    domain: str,
    service: str,
    entity_id: str,
    data: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a Home Assistant service via Rex's HABridge.

    Constructs an :class:`~rex.ha_bridge.IntentMatch` from the supplied
    arguments and delegates to :meth:`~rex.ha_bridge.HABridge._execute_intent`.

    .. note::
        This tool is policy-gated (MEDIUM risk).  Callers are responsible
        for obtaining policy approval before invoking this function.

    Args:
        domain:    Home Assistant domain (e.g. ``"light"``, ``"switch"``).
        service:   Service to call (e.g. ``"turn_on"``, ``"turn_off"``).
        entity_id: Target entity (e.g. ``"light.living_room"``).
        data:      Optional extra service-call parameters merged into the
            request body alongside ``entity_id``.
        context:   Optional ambient context dict (unused; reserved for future
            injection).

    Returns:
        A dict with keys ``success`` (bool), ``message`` (str), and
        ``entity_id`` (str).
    """
    bridge = _get_ha_bridge()
    if not bridge.enabled:
        return {
            "success": False,
            "message": "Home Assistant bridge is not configured.",
            "entity_id": entity_id,
        }

    intent_data: dict[str, Any] = {"entity_id": entity_id}
    if data:
        intent_data.update(data)

    intent = _IntentMatch(
        domain=domain,
        service=service,
        entity_id=entity_id,
        data=intent_data,
        description=f"{domain}.{service} {entity_id}",
        source="openclaw",
    )
    success, message = bridge._execute_intent(intent)
    return {"success": success, "message": message, "entity_id": entity_id}


def register(agent: Any = None) -> Any:
    """Register the ``home_assistant_call_service`` tool with OpenClaw.

    When the ``openclaw`` package is available this function calls
    OpenClaw's tool registration API, passing :func:`ha_call_service` as the
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
    #       handler=ha_call_service,
    #       agent=agent,
    #   )
    #   return handle
    logger.warning(
        "OpenClaw tool registration stub for %s — update once API is confirmed (PRD §8.3)",
        TOOL_NAME,
    )
    return None
