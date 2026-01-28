"""
Event bus module for Rex AI Assistant.

Provides a publish-subscribe event system for internal communication
between components (scheduler, email, calendar, workflows, etc.).
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """An event in the system."""

    event_type: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        """String representation of the event."""
        return f"Event(type={self.event_type}, timestamp={self.timestamp.isoformat()})"


class EventBus:
    """
    Event bus for publish-subscribe messaging.

    Features:
    - Subscribe handlers to specific event types
    - Publish events to all subscribed handlers
    - Thread-safe operation
    - Wildcard subscriptions (subscribe to all events with '*')
    - Error handling to prevent one handler from affecting others
    """

    def __init__(self):
        """Initialize the event bus."""
        self._subscriptions: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.RLock()
        self._event_count = 0
        self._error_count = 0

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Event type to subscribe to (use '*' for all events)
            handler: Callable that takes an Event and returns None
        """
        with self._lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []

            self._subscriptions[event_type].append(handler)
            logger.debug(f"Subscribed handler {handler.__name__} to event type: {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]) -> bool:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: Event type to unsubscribe from
            handler: Handler to remove

        Returns:
            True if handler was removed, False if not found
        """
        with self._lock:
            if event_type in self._subscriptions:
                try:
                    self._subscriptions[event_type].remove(handler)
                    logger.debug(f"Unsubscribed handler {handler.__name__} from event type: {event_type}")
                    return True
                except ValueError:
                    pass
            return False

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribed handlers.

        Args:
            event: Event to publish
        """
        with self._lock:
            self._event_count += 1

            # Get handlers for this specific event type
            handlers = self._subscriptions.get(event.event_type, []).copy()

            # Add wildcard handlers
            handlers.extend(self._subscriptions.get('*', []))

        logger.debug(f"Publishing event: {event.event_type} to {len(handlers)} handler(s)")

        # Execute handlers outside the lock to prevent deadlocks
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in event handler {handler.__name__} for event {event.event_type}: {e}", exc_info=True)

    def get_subscription_count(self, event_type: str) -> int:
        """
        Get the number of handlers subscribed to an event type.

        Args:
            event_type: Event type to check

        Returns:
            Number of subscribed handlers
        """
        with self._lock:
            return len(self._subscriptions.get(event_type, []))

    def get_stats(self) -> dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            return {
                'total_events': self._event_count,
                'error_count': self._error_count,
                'subscription_types': len(self._subscriptions),
                'subscriptions': {
                    event_type: len(handlers)
                    for event_type, handlers in self._subscriptions.items()
                }
            }

    def clear_subscriptions(self, event_type: Optional[str] = None) -> None:
        """
        Clear subscriptions.

        Args:
            event_type: If specified, clear only this type. If None, clear all.
        """
        with self._lock:
            if event_type is not None:
                self._subscriptions.pop(event_type, None)
                logger.debug(f"Cleared subscriptions for event type: {event_type}")
            else:
                self._subscriptions.clear()
                logger.debug("Cleared all subscriptions")


# Global event bus instance
_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:
                _event_bus = EventBus()
    return _event_bus


def set_event_bus(event_bus: EventBus) -> None:
    """Set the global event bus instance (for testing)."""
    global _event_bus
    with _event_bus_lock:
        _event_bus = event_bus
