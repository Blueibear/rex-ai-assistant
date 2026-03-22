# OPENCLAW-REPLACE: This module will be replaced by OpenClaw. Do not add new features.

"""
Event bus module for Rex AI Assistant.

Provides a lightweight publish-subscribe event system for internal communication
between components (scheduler, email, calendar, workflows, etc.).

Compatibility goals:
- Supports the "simple" API used by older code:
    bus.publish(event_type: str, payload: dict) -> Event
    unsubscribe = bus.subscribe(event_type: str, callback(event_type, payload)) -> callable
- Also supports the "rich" API used by newer code:
    bus.publish(Event(...)) -> None
    bus.subscribe(event_type: str, handler(Event)) -> None
    bus.unsubscribe(event_type: str, handler(Event)) -> bool
- Thread-safe, wildcard subscriptions with "*"
- Errors in one handler do not break others
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, overload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """Represents a published event."""

    event_type: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"Event(type={self.event_type}, timestamp={self.timestamp.isoformat()})"


# Two callback shapes we support:
# 1) legacy: callback(event_type: str, payload: dict) -> None
LegacyCallback = Callable[[str, dict[str, Any]], None]
# 2) newer: handler(event: Event) -> None
EventHandler = Callable[[Event], None]


class EventBus:
    """
    Event bus for publish-subscribe messaging.

    Features:
    - Thread-safe subscribe/publish/unsubscribe
    - Wildcard subscriptions: subscribe to "*" to receive all events
    - Error isolation: one bad handler won't stop the others
    - Dual API support (legacy + newer)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Store both styles separately to avoid signature confusion
        self._legacy_subscribers: defaultdict[str, list[LegacyCallback]] = defaultdict(list)
        self._handlers: defaultdict[str, list[EventHandler]] = defaultdict(list)
        self._subscriptions = self._handlers

        self._event_count = 0
        self._error_count = 0

    # -------------------------
    # Subscribe / Unsubscribe
    # -------------------------

    @overload
    def subscribe(self, event_type: str, callback: LegacyCallback) -> Callable[[], None]: ...
    @overload
    def subscribe(self, event_type: str, handler: EventHandler) -> None: ...

    def subscribe(self, event_type: str, fn: Callable[..., Any]):  # type: ignore[misc]
        """
        Subscribe to an event type.

        Legacy mode:
            unsubscribe = subscribe("type", callback(event_type, payload))
        New mode:
            subscribe("type", handler(event))
        """
        with self._lock:
            # Best-effort signature detection:
            # - If function expects 2+ positional args, treat as legacy callback.
            # - Otherwise treat as Event handler.
            is_legacy = False
            try:
                code = getattr(fn, "__code__", None)
                if code is not None and code.co_argcount >= 2:
                    is_legacy = True
            except Exception:
                is_legacy = False

            if is_legacy:
                cb = fn
                if cb not in self._legacy_subscribers[event_type]:
                    self._legacy_subscribers[event_type].append(cb)

                def _unsubscribe() -> None:
                    self._safe_remove_legacy(event_type, cb)

                return _unsubscribe

            handler = fn
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
            return None

    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """
        Unsubscribe a newer-style handler(Event) from an event type.

        Returns True if removed, False otherwise.
        """
        with self._lock:
            try:
                self._handlers[event_type].remove(handler)
                return True
            except (KeyError, ValueError):
                return False

    def _safe_remove_legacy(self, event_type: str, callback: LegacyCallback) -> None:
        with self._lock:
            try:
                self._legacy_subscribers[event_type].remove(callback)
            except (KeyError, ValueError):
                return

    # -------------------------
    # Publish
    # -------------------------

    @overload
    def publish(self, event_type: str, payload: dict[str, Any]) -> Event: ...
    @overload
    def publish(self, event: Event) -> None: ...

    def publish(self, arg1: Any, arg2: Any = None):  # type: ignore[misc]
        """
        Publish an event.

        Legacy mode:
            event = publish("type", {"k":"v"}) -> Event
        New mode:
            publish(Event(...)) -> None
        """
        if isinstance(arg1, Event):
            event = arg1
            self._publish_event(event)
            return None

        event_type = str(arg1)
        payload = arg2 if isinstance(arg2, dict) else {}
        event = Event(event_type=event_type, payload=payload, timestamp=datetime.now(timezone.utc))
        self._publish_event(event)
        return event

    def _publish_event(self, event: Event) -> None:
        # Snapshot handlers under lock, then execute outside lock
        with self._lock:
            self._event_count += 1
            legacy_callbacks = list(self._legacy_subscribers.get(event.event_type, []))
            legacy_callbacks.extend(self._legacy_subscribers.get("*", []))

            handlers = list(self._handlers.get(event.event_type, []))
            handlers.extend(self._handlers.get("*", []))

        total = len(legacy_callbacks) + len(handlers)
        logger.debug("Publishing event %s to %d subscriber(s)", event.event_type, total)

        for cb in legacy_callbacks:
            try:
                cb(event.event_type, event.payload)
            except Exception as e:
                self._record_error(cb, event, e)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._record_error(handler, event, e)

    def _record_error(self, fn: Callable[..., Any], event: Event, exc: Exception) -> None:
        with self._lock:
            self._error_count += 1
        name = getattr(fn, "__name__", repr(fn))
        logger.error(
            "Error in event handler %s for event %s: %s",
            name,
            event.event_type,
            exc,
            exc_info=True,
        )

    def get_metrics(self) -> dict[str, int]:
        """Return event bus metrics."""
        with self._lock:
            return {
                "published_events": self._event_count,
                "handler_errors": self._error_count,
            }

    # -------------------------
    # Introspection / Maintenance
    # -------------------------

    def iter_subscribers(self, event_type: str) -> Iterable[Callable[..., Any]]:
        """Return a snapshot iterable of all subscribers for an event type (including both styles)."""
        with self._lock:
            yield from self._legacy_subscribers.get(event_type, [])
            yield from self._handlers.get(event_type, [])

    def get_subscription_count(self, event_type: str) -> int:
        """Count subscribers for a specific event type (not including wildcard)."""
        with self._lock:
            return len(self._legacy_subscribers.get(event_type, [])) + len(
                self._handlers.get(event_type, [])
            )

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            keys = set(self._legacy_subscribers.keys()) | set(self._handlers.keys())
            return {
                "total_events": self._event_count,
                "error_count": self._error_count,
                "subscription_types": len(keys),
                "subscriptions": {
                    et: {
                        "legacy": len(self._legacy_subscribers.get(et, [])),
                        "handlers": len(self._handlers.get(et, [])),
                    }
                    for et in sorted(keys)
                },
            }

    def clear_subscriptions(self, event_type: str | None = None) -> None:
        """
        Clear subscriptions.

        Args:
            event_type: If specified, clear only this type. If None, clear all.
        """
        with self._lock:
            if event_type is None:
                self._legacy_subscribers.clear()
                self._handlers.clear()
                logger.debug("Cleared all subscriptions")
                return

            self._legacy_subscribers.pop(event_type, None)
            self._handlers.pop(event_type, None)
            logger.debug("Cleared subscriptions for event type: %s", event_type)


class EventQueue:
    """
    Bounded, sequentially-processed event queue backed by an EventBus.

    Events are placed in a bounded queue to prevent unbounded memory growth.
    A single worker thread processes events in order, ensuring sequential delivery.
    When the queue is full, new events are dropped with a warning.
    """

    DEFAULT_MAXSIZE = 1000

    def __init__(self, bus: EventBus, maxsize: int = DEFAULT_MAXSIZE) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self._bus = bus
        self._maxsize = maxsize
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=maxsize)
        self._worker: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._dropped_count = 0
        self._processed_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(
                target=self._process_events,
                name="EventQueue-worker",
                daemon=True,
            )
            self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker thread, waiting up to *timeout* seconds."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        # Wake up the worker so it can exit
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    # ------------------------------------------------------------------
    # Enqueueing
    # ------------------------------------------------------------------

    def enqueue(self, event: Event) -> bool:
        """
        Add *event* to the queue.

        Returns True if the event was queued, False if the queue was full
        and the event was dropped.
        """
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_count += 1
            logger.warning(
                "EventQueue overflow (maxsize=%d): dropping event %s",
                self._maxsize,
                event.event_type,
            )
            return False

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _process_events(self) -> None:
        """Worker loop — runs in a dedicated daemon thread."""
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if item is None:
                # Sentinel: time to exit
                self._queue.task_done()
                break

            try:
                self._bus._publish_event(item)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "EventQueue: unhandled error publishing %s: %s",
                    item.event_type,
                    exc,
                    exc_info=True,
                )
            finally:
                self._queue.task_done()
                with self._lock:
                    self._processed_count += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def qsize(self) -> int:
        """Approximate number of events currently in the queue."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """True when the queue has reached its maxsize."""
        return self._queue.full()

    @property
    def is_running(self) -> bool:
        """True while the worker thread is active."""
        return self._running

    def get_metrics(self) -> dict[str, int]:
        """Return queue metrics."""
        with self._lock:
            return {
                "processed": self._processed_count,
                "dropped": self._dropped_count,
                "queued": self._queue.qsize(),
                "maxsize": self._maxsize,
            }

    def join(self) -> None:
        """Block until all currently-queued events have been processed."""
        self._queue.join()


# Global event bus instance
_EVENT_BUS: EventBus | None = None
_EVENT_BUS_LOCK = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the global event bus instance."""
    global _EVENT_BUS
    if _EVENT_BUS is None:
        with _EVENT_BUS_LOCK:
            if _EVENT_BUS is None:
                _EVENT_BUS = EventBus()
    return _EVENT_BUS


def set_event_bus(event_bus: EventBus) -> None:
    """Set the global event bus instance (for testing)."""
    global _EVENT_BUS
    with _EVENT_BUS_LOCK:
        _EVENT_BUS = event_bus


__all__ = ["Event", "EventBus", "EventQueue", "get_event_bus", "set_event_bus"]
