"""HA event subscriber — US-P5-007.

Routes ``"ha.command"`` events published on Rex's EventBridge to the Home
Assistant bridge, so that external components can trigger HA service calls
via the event bus rather than direct method calls.

Event payload format::

    {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living_room",
        "data": {"brightness_pct": 80}   # optional
    }

Typical usage::

    from rex.openclaw.ha_event_subscriber import HaEventSubscriber
    from rex.openclaw.event_bridge import EventBridge

    bridge = EventBridge()
    subscriber = HaEventSubscriber(bus=bridge)
    subscriber.subscribe()

    # Now publishing an ha.command event triggers an HA service call:
    bridge.publish("ha.command", {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living_room",
    })
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from rex.event_bus import EventBus as _EventBus
from rex.openclaw.event_bridge import EventBridge as _EventBridge
from rex.openclaw.tools.ha_tool import ha_call_service

logger = logging.getLogger(__name__)

#: Event type that triggers HA service calls.
HA_COMMAND_EVENT = "ha.command"


class HaEventSubscriber:
    """Subscribe to ``"ha.command"`` events and dispatch them to HABridge.

    Accepts an explicit bus (EventBridge or EventBus) for injection in tests.
    Defaults to the global EventBridge singleton.

    Args:
        bus: Optional bus to subscribe on. Defaults to a fresh EventBridge
            (which wraps the global Rex event bus singleton).
    """

    def __init__(self, bus: _EventBridge | _EventBus | None = None) -> None:
        self._bus: _EventBridge | _EventBus = bus if bus is not None else _EventBridge()
        self._handler: Callable[..., None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self) -> None:
        """Register the handler for ``"ha.command"`` events on the bus.

        Idempotent — calling multiple times has no additional effect.
        """
        if self._handler is not None:
            return
        self._handler = self._on_ha_command
        self._bus.subscribe(HA_COMMAND_EVENT, self._handler)
        logger.debug("[HaEventSubscriber] Subscribed to '%s'", HA_COMMAND_EVENT)

    def unsubscribe(self) -> None:
        """Remove the handler from the bus.

        Idempotent — safe to call even if not subscribed.
        """
        if self._handler is None:
            return
        self._bus.unsubscribe(HA_COMMAND_EVENT, self._handler)
        self._handler = None
        logger.debug("[HaEventSubscriber] Unsubscribed from '%s'", HA_COMMAND_EVENT)

    @property
    def is_subscribed(self) -> bool:
        """True when currently subscribed to the event bus."""
        return self._handler is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_ha_command(self, *args: Any) -> None:
        """Handle an ``"ha.command"`` event.

        Supports both legacy (event_type, payload) and rich (Event) forms.
        """
        # Legacy form: (event_type: str, payload: dict)
        # Rich form:   (event: Event)
        payload: dict[str, Any] = {}
        if len(args) == 2:
            # legacy: (event_type, payload)
            payload = args[1] if isinstance(args[1], dict) else {}
        elif len(args) == 1:
            raw = args[0]
            if isinstance(raw, dict):
                payload = raw
            elif hasattr(raw, "payload"):
                payload = raw.payload if isinstance(raw.payload, dict) else {}

        domain = payload.get("domain", "")
        service = payload.get("service", "")
        entity_id = payload.get("entity_id", "")
        data: dict[str, Any] = payload.get("data") or {}

        if not (domain and service and entity_id):
            logger.warning(
                "[HaEventSubscriber] Invalid ha.command payload — missing domain/service/entity_id: %r",
                payload,
            )
            return

        logger.info(
            "[HaEventSubscriber] Dispatching ha.command: %s.%s %s",
            domain,
            service,
            entity_id,
        )
        result = ha_call_service(
            domain=domain,
            service=service,
            entity_id=entity_id,
            data=data or None,
        )
        if not result.get("success"):
            logger.warning(
                "[HaEventSubscriber] ha.command failed: %s", result.get("message")
            )
