"""Simple in-process event bus for Rex services."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, DefaultDict, Iterable

EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class Event:
    """Represents a published event."""

    event_type: str
    payload: dict[str, Any]
    timestamp: datetime


class EventBus:
    """Lightweight publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, list[EventCallback]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: EventCallback) -> Callable[[], None]:
        """Subscribe to an event type and return an unsubscribe callable."""
        self._subscribers[event_type].append(callback)

        def _unsubscribe() -> None:
            self._subscribers[event_type].remove(callback)

        return _unsubscribe

    def publish(self, event_type: str, payload: dict[str, Any]) -> Event:
        """Publish an event to all subscribers."""
        event = Event(
            event_type=event_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
        )
        for callback in self._iter_callbacks(event_type):
            callback(event.event_type, event.payload)
        return event

    def _iter_callbacks(self, event_type: str) -> Iterable[EventCallback]:
        callbacks = list(self._subscribers.get(event_type, []))
        callbacks.extend(self._subscribers.get("*", []))
        return callbacks


_EVENT_BUS: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the global event bus instance."""
    global _EVENT_BUS
    if _EVENT_BUS is None:
        _EVENT_BUS = EventBus()
    return _EVENT_BUS


def set_event_bus(event_bus: EventBus) -> None:
    """Set the global event bus instance."""
    global _EVENT_BUS
    _EVENT_BUS = event_bus


__all__ = ["Event", "EventBus", "get_event_bus", "set_event_bus"]
