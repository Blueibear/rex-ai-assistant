"""Event trigger system for Rex AI Assistant.

Allows workflows (or arbitrary callables) to be registered as triggers for
specific event types.  When a matching event is published on the event bus the
registered trigger is invoked.

Usage::

    from rex.event_triggers import EventTriggerRegistry, get_trigger_registry

    registry = get_trigger_registry()
    registry.register("alarm.fired", my_workflow_launcher)

    # Later, publishing an event on the bus will invoke my_workflow_launcher.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from rex.event_bus import Event, EventBus, get_event_bus

logger = logging.getLogger(__name__)

# Type alias for a trigger callable: receives the Event and returns anything.
TriggerFn = Callable[[Event], Any]


class EventTriggerRegistry:
    """Registry that maps event types to trigger callables.

    When :meth:`attach` is called the registry subscribes itself to the event
    bus so that published events are forwarded to registered triggers.
    """

    def __init__(self, bus: EventBus | None = None) -> None:
        self._bus: EventBus = bus if bus is not None else get_event_bus()
        self._lock = threading.RLock()
        self._triggers: defaultdict[str, list[TriggerFn]] = defaultdict(list)
        self._attached = False
        # Stable lambda so we can unsubscribe the same object later.
        self._bus_handler: TriggerFn = lambda event: self._handle_event(event)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, event_type: str, trigger: TriggerFn) -> None:
        """Register *trigger* to be called whenever *event_type* is published.

        If the registry is already attached to the bus the trigger will receive
        events immediately.  Call :meth:`attach` first if you want events to
        flow.

        Args:
            event_type: The event type string to listen for.
            trigger: A callable that accepts a single :class:`~rex.event_bus.Event`.
        """
        with self._lock:
            self._triggers[event_type].append(trigger)
        logger.debug("Registered trigger for event type '%s': %s", event_type, trigger)

    def unregister(self, event_type: str, trigger: TriggerFn) -> bool:
        """Remove a previously registered trigger.

        Returns:
            ``True`` if the trigger was found and removed, ``False`` otherwise.
        """
        with self._lock:
            try:
                self._triggers[event_type].remove(trigger)
                logger.debug("Unregistered trigger for event type '%s': %s", event_type, trigger)
                return True
            except ValueError:
                return False

    def list_triggers(self, event_type: str | None = None) -> dict[str, list[TriggerFn]]:
        """Return a snapshot of registered triggers.

        Args:
            event_type: If given, return only triggers for that event type.
        """
        with self._lock:
            if event_type is not None:
                return {event_type: list(self._triggers.get(event_type, []))}
            return {et: list(fns) for et, fns in self._triggers.items()}

    # ------------------------------------------------------------------
    # Bus attachment
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Subscribe this registry to the event bus.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        with self._lock:
            if self._attached:
                return
            self._bus.subscribe("*", self._bus_handler)
            self._attached = True
            logger.debug("EventTriggerRegistry attached to event bus")

    def detach(self) -> None:
        """Unsubscribe this registry from the event bus."""
        with self._lock:
            if not self._attached:
                return
            self._bus.unsubscribe("*", self._bus_handler)
            self._attached = False
            logger.debug("EventTriggerRegistry detached from event bus")

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _handle_event(self, event: Event) -> None:
        """Called by the event bus for every published event."""
        with self._lock:
            fns = list(self._triggers.get(event.event_type, []))

        if not fns:
            return

        logger.debug("Event '%s' matched %d trigger(s)", event.event_type, len(fns))
        for fn in fns:
            try:
                fn(event)
            except Exception:
                logger.exception(
                    "Trigger '%s' raised an error for event '%s'",
                    getattr(fn, "__name__", repr(fn)),
                    event.event_type,
                )


# ---------------------------------------------------------------------------
# Global registry helpers
# ---------------------------------------------------------------------------

_REGISTRY: EventTriggerRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_trigger_registry() -> EventTriggerRegistry:
    """Return (and lazily create) the global :class:`EventTriggerRegistry`."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = EventTriggerRegistry()
    return _REGISTRY


def set_trigger_registry(registry: EventTriggerRegistry) -> None:
    """Replace the global registry (useful in tests)."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = registry


__all__ = [
    "EventTriggerRegistry",
    "TriggerFn",
    "get_trigger_registry",
    "set_trigger_registry",
]
