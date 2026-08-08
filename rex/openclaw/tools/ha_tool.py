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

import hashlib
import logging
import uuid
from typing import Any

from rex.ha.mutation_service import HAMutation, HAMutationService
from rex.ha_bridge import HABridge as _HABridge
from rex.response.builder import home_assistant_status_message

logger = logging.getLogger(__name__)

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


class _BridgeMutationClient:
    def __init__(self, bridge: _HABridge) -> None:
        self._bridge = bridge

    def call_service(self, domain: str, service: str, data: dict[str, Any]) -> None:
        self._bridge._request("POST", f"/api/services/{domain}/{service}", json=data)

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        value = self._bridge._request("GET", f"/api/states/{entity_id}")
        return value if isinstance(value, dict) else None


def _get_mutation_service(bridge: _HABridge) -> HAMutationService:
    token = str(getattr(bridge, "_token", ""))
    secret = hashlib.sha256(f"askrex-ha-confirmation:{token}".encode()).digest()
    return HAMutationService(_BridgeMutationClient(bridge), confirmation_secret=secret)


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

    ambient = context or {}
    user_id = str(ambient.get("user_id") or ambient.get("user") or "")
    request_id = str(ambient.get("request_id") or uuid.uuid4())
    confirmation_token = ambient.get("confirmation_token")
    result = _get_mutation_service(bridge).execute(
        HAMutation(
            user_id=user_id,
            domain=domain,
            service=service,
            entity_id=entity_id,
            parameters=dict(data or {}),
            request_id=request_id,
            confirmation_token=(
                str(confirmation_token) if confirmation_token is not None else None
            ),
        )
    )
    return {
        **result.to_dict(),
        "message": home_assistant_status_message(
            result.status.value,
            entity_id=result.entity_id,
            detail=result.detail,
            expected=result.expected,
        ),
    }
