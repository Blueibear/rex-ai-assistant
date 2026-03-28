"""OpenClaw event bridge — US-P4-015.

EventBridge is Rex's internal event bus.  It implements
:class:`~rex.contracts.event_bus.EventBusProtocol` by delegating to Rex's
existing :class:`~rex.event_bus.EventBus` singleton.

OpenClaw event bridging (WebSocket) is out of scope for HTTP integration.
All subscribe/publish operations run locally within the Rex process.

# FUTURE: WebSocket bridge to OpenClaw gateway events

Typical usage::

    from rex.openclaw.event_bridge import EventBridge

    bridge = EventBridge()

    # Subscribe to an event (legacy or rich style)
    unsubscribe = bridge.subscribe("email.unread", lambda et, payload: print(payload))

    # Publish an event (simple form)
    bridge.publish("email.unread", {"count": 3, "emails": []})

    # Full delegation: all operations pass through to Rex EventBus
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Callable

from rex.openclaw.event_bus import EventBus as _EventBus
from rex.openclaw.event_bus import get_event_bus as _get_event_bus

logger = logging.getLogger(__name__)


class EventBridge:
    """Rex's internal event bus, exposed as an OpenClaw-compatible adapter.

    Implements :class:`~rex.contracts.event_bus.EventBusProtocol` by
    delegating all operations to an underlying :class:`~rex.event_bus.EventBus`
    instance.

    EventBridge is Rex's internal event bus.  OpenClaw event bridging
    (WebSocket) is out of scope for HTTP integration.

    When no ``bus`` is supplied the global Rex singleton (via
    :func:`~rex.event_bus.get_event_bus`) is used, ensuring that all
    existing Rex subscribers continue to receive events published through
    the bridge.

    Args:
        bus: Optional explicit :class:`~rex.event_bus.EventBus` instance.
            Defaults to the global Rex event bus singleton.
    """

    def __init__(self, bus: _EventBus | None = None) -> None:
        self._bus: _EventBus = bus if bus is not None else _get_event_bus()

    # ------------------------------------------------------------------
    # EventBusProtocol implementation
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, fn: Callable[..., None]) -> Callable[[], None] | None:
        """Register a handler for *event_type*.

        Delegates to :meth:`~rex.event_bus.EventBus.subscribe`.

        Args:
            event_type: The event name to subscribe to.  Use ``"*"`` for
                wildcard (all events).
            fn: Either a legacy ``(event_type, payload)`` callback or a
                rich ``(Event)`` handler.

        Returns:
            For legacy callbacks: an unsubscribe callable.
            For rich handlers: ``None``.
        """
        result: Callable[[], None] | None = self._bus.subscribe(event_type, fn)
        return result

    def unsubscribe(self, event_type: str, handler: Callable[..., None]) -> bool:
        """Remove *handler* from *event_type* subscriptions.

        Delegates to :meth:`~rex.event_bus.EventBus.unsubscribe`.

        Args:
            event_type: The event name the handler was subscribed to.
            handler: The exact callable that was passed to :meth:`subscribe`.

        Returns:
            ``True`` if the handler was found and removed, ``False`` otherwise.
        """
        return self._bus.unsubscribe(event_type, handler)

    def publish(self, *args: Any, **kwargs: Any) -> Any:
        """Publish an event.

        Delegates to :meth:`~rex.event_bus.EventBus.publish`.

        Supports two call forms:

        - ``publish(event_type: str, payload: dict)`` — simple form
        - ``publish(event: Event)`` — rich form

        Returns:
            The :class:`~rex.event_bus.Event` object that was dispatched
            (simple form), or ``None`` (rich form).
        """
        return self._bus.publish(*args, **kwargs)

    def get_metrics(self) -> dict[str, int]:
        """Return counters: ``published_events``, ``handler_errors``.

        Delegates to :meth:`~rex.event_bus.EventBus.get_metrics`.
        """
        return self._bus.get_metrics()

    def get_stats(self) -> dict[str, Any]:
        """Return extended statistics including per-event-type subscriber counts.

        Delegates to :meth:`~rex.event_bus.EventBus.get_stats`.
        """
        return self._bus.get_stats()

    def iter_subscribers(self, event_type: str) -> Iterable[Callable[..., None]]:
        """Yield all handlers for *event_type*.

        Delegates to :meth:`~rex.event_bus.EventBus.iter_subscribers`.
        """
        return self._bus.iter_subscribers(event_type)

    def get_subscription_count(self, event_type: str) -> int:
        """Return the total number of handlers subscribed to *event_type*.

        Delegates to :meth:`~rex.event_bus.EventBus.get_subscription_count`.
        """
        return self._bus.get_subscription_count(event_type)

    def clear_subscriptions(self, event_type: str | None = None) -> None:
        """Remove subscriptions.

        Delegates to :meth:`~rex.event_bus.EventBus.clear_subscriptions`.

        Args:
            event_type: When provided, clear only this event type.
                When ``None``, clear all subscriptions.
        """
        self._bus.clear_subscriptions(event_type)
