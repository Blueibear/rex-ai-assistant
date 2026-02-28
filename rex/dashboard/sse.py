"""Server-Sent Events broadcaster for dashboard notifications."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class NotificationEvent:
    """Dashboard notification event payload."""

    type: str
    notification_id: str
    user_id: str | None
    unread_count: int | None = None


class _Subscriber:
    def __init__(self, max_events: int) -> None:
        self.queue: queue.Queue[NotificationEvent] = queue.Queue(maxsize=max_events)
        self.closed = False


class NotificationBroadcaster:
    """Thread-safe in-process notification event broadcaster."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[_Subscriber] = set()

    def subscribe(self, *, max_events: int = 100) -> _Subscriber:
        """Create and register a subscriber with a bounded queue."""
        subscriber = _Subscriber(max_events=max_events)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        """Remove and close a subscriber."""
        with self._lock:
            self._subscribers.discard(subscriber)
        subscriber.closed = True

    def publish(self, event: NotificationEvent) -> None:
        """Publish an event to all active subscribers without blocking."""
        with self._lock:
            subscribers = list(self._subscribers)

        stale: list[_Subscriber] = []
        for subscriber in subscribers:
            if subscriber.closed:
                stale.append(subscriber)
                continue

            try:
                subscriber.queue.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.queue.get_nowait()
                    subscriber.queue.put_nowait(event)
                except (queue.Empty, queue.Full):
                    stale.append(subscriber)

        if stale:
            with self._lock:
                for subscriber in stale:
                    self._subscribers.discard(subscriber)
                    subscriber.closed = True

    def stream(
        self,
        subscriber: _Subscriber,
        *,
        timeout: float = 15.0,
        keepalive_interval: float = 15.0,
    ):
        """Yield SSE payload chunks for a subscriber."""
        last_keepalive = time.monotonic()
        try:
            while not subscriber.closed:
                try:
                    event = subscriber.queue.get(timeout=timeout)
                except queue.Empty:
                    now = time.monotonic()
                    if now - last_keepalive >= keepalive_interval:
                        last_keepalive = now
                        yield ": keep-alive\n\n"
                    continue

                last_keepalive = time.monotonic()
                payload = json.dumps(
                    {
                        "type": event.type,
                        "notification_id": event.notification_id,
                        "user_id": event.user_id,
                        "unread_count": event.unread_count,
                    }
                )
                yield f"event: notification\ndata: {payload}\n\n"
        finally:
            self.unsubscribe(subscriber)


_broadcaster: NotificationBroadcaster | None = None


def get_broadcaster() -> NotificationBroadcaster:
    """Get singleton broadcaster instance."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = NotificationBroadcaster()
    return _broadcaster


def set_broadcaster(broadcaster: NotificationBroadcaster | None) -> None:
    """Set singleton broadcaster (primarily for tests)."""
    global _broadcaster
    _broadcaster = broadcaster


__all__ = [
    "NotificationBroadcaster",
    "NotificationEvent",
    "get_broadcaster",
    "set_broadcaster",
]
