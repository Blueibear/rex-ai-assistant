"""Protocol defining the event bus interface for Rex.

This contract captures the public API of ``rex.event_bus`` so that an
OpenClaw-backed adapter can be substituted transparently.  It covers both
the *simple* (legacy) callback API and the *rich* ``Event``-object API that
the existing bus supports.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, runtime_checkable


# Two handler shapes the bus supports:
# 1) Legacy:  callback(event_type: str, payload: dict) -> None
# 2) Rich:    handler(Event) -> None
# We use ``Callable[..., None]`` here so the Protocol stays dependency-free.
AnyHandler = Callable[..., None]


@runtime_checkable
class EventBusProtocol(Protocol):
    """Structural protocol for the Rex event bus.

    Covers the full public API of ``rex.event_bus.EventBus``:

    - ``subscribe`` / ``unsubscribe`` — add and remove handlers
    - ``publish`` — fire an event (simple or rich form)
    - ``get_metrics`` / ``get_stats`` — introspection
    - ``iter_subscribers`` / ``get_subscription_count`` — iteration helpers
    - ``clear_subscriptions`` — teardown

    Both the *legacy* callback style (``callback(event_type, payload)``) and
    the *rich* handler style (``handler(Event)``) must be preserved by any
    implementation.

    Note: The ``EventQueue`` helper class is a separate concern and is not
    included in this protocol.  It wraps an ``EventBusProtocol`` instance.
    """

    def subscribe(self, event_type: str, fn: AnyHandler) -> Callable[[], None] | None:
        """Register a handler for *event_type*.

        Args:
            event_type: The event name to subscribe to.  Use ``"*"`` for all
                events (wildcard).
            fn: Either a legacy ``(event_type, payload)`` callback or a rich
                ``(Event)`` handler.  Implementors must detect the signature
                and store accordingly.

        Returns:
            For legacy callbacks: an unsubscribe callable.
            For rich handlers: ``None`` (use :meth:`unsubscribe` instead).
        """
        ...

    def unsubscribe(self, event_type: str, handler: AnyHandler) -> bool:
        """Remove *handler* from *event_type* subscriptions.

        Args:
            event_type: The event name the handler was subscribed to.
            handler: The exact callable that was passed to :meth:`subscribe`.

        Returns:
            ``True`` if the handler was found and removed, ``False`` otherwise.
        """
        ...

    def publish(self, *args: Any, **kwargs: Any) -> Any:
        """Publish an event.

        Supports two call forms:

        - ``publish(event_type: str, payload: dict)`` — simple form
        - ``publish(event: Event)`` — rich form

        Returns the ``Event`` object that was dispatched (simple form), or
        ``None`` (rich form).
        """
        ...

    def get_metrics(self) -> dict[str, int]:
        """Return counters: ``event_count``, ``error_count``, etc."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Return extended statistics including per-event-type subscriber counts."""
        ...

    def iter_subscribers(self, event_type: str) -> Iterable[AnyHandler]:
        """Yield all handlers (both legacy and rich) for *event_type*."""
        ...

    def get_subscription_count(self, event_type: str) -> int:
        """Return the total number of handlers subscribed to *event_type*."""
        ...

    def clear_subscriptions(self, event_type: str | None = None) -> None:
        """Remove all subscriptions.

        Args:
            event_type: When provided, clear only subscriptions for this event
                type.  When ``None``, clear all subscriptions on the bus.
        """
        ...
