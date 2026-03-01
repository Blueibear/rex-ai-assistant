"""Server-Sent Events (SSE) broadcaster for dashboard notifications.

Provides an in-process, thread-safe broadcaster that pushes notification
events to connected SSE clients. Uses stdlib only (queue, threading, time).

Typical usage:

    from rex.dashboard.sse import get_broadcaster, NotificationEvent

    broadcaster = get_broadcaster()
    subscriber = broadcaster.subscribe()

    def sse_stream():
        for chunk in broadcaster.stream(subscriber, timeout=15.0):
            yield chunk

    broadcaster.publish(
        NotificationEvent(
            type="notification",
            notification_id="notif_123",
            user_id="james",
            unread_count=5,
        )
    )

Design notes:
- Each subscriber has its own bounded queue. When full, the oldest event is dropped.
- stream() yields SSE-formatted strings and periodic keep-alive comments.
- publish() accepts either a NotificationEvent or a JSON-serializable dict payload.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NotificationEvent:
    """Dashboard notification event payload."""

    type: str
    notification_id: str
    user_id: str | None
    unread_count: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "unread_count": self.unread_count,
        }


class _Subscriber:
    def __init__(self, max_events: int) -> None:
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_events)
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

    def publish(self, event: NotificationEvent | dict[str, Any]) -> None:
        """Publish an event to all active subscribers without blocking."""
        if isinstance(event, NotificationEvent):
            payload = event.to_payload()
        elif isinstance(event, dict):
            payload = event
        else:
            raise TypeError("event must be NotificationEvent or dict[str, Any]")

        with self._lock:
            subscribers = list(self._subscribers)

        stale: list[_Subscriber] = []
        for subscriber in subscribers:
            if subscriber.closed:
                stale.append(subscriber)
                continue

            try:
                subscriber.queue.put_nowait(payload)
                continue
            except queue.Full:
                pass

            # Drop the oldest event, then retry once.
            try:
                subscriber.queue.get_nowait()
            except queue.Empty:
                stale.append(subscriber)
                continue

            try:
                subscriber.queue.put_nowait(payload)
            except queue.Full:
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
    ) -> Iterator[str]:
        """Yield SSE payload chunks for a subscriber."""
        last_keepalive = time.monotonic()
        try:
            while not subscriber.closed:
                try:
                    payload = subscriber.queue.get(timeout=timeout)
                except queue.Empty:
                    now = time.monotonic()
                    if now - last_keepalive >= keepalive_interval:
                        last_keepalive = now
                        yield ": keep-alive\n\n"
                    continue

                last_keepalive = time.monotonic()
                data = json.dumps(payload)
                yield f"event: notification\ndata: {data}\n\n"
        finally:
            self.unsubscribe(subscriber)

    def shutdown(self) -> None:
        """Close and remove all subscribers."""
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            subscriber.closed = True

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


_broadcaster: NotificationBroadcaster | None = None
_broadcaster_lock = threading.Lock()


def get_broadcaster() -> NotificationBroadcaster:
    """Get singleton broadcaster instance (lazy, thread-safe)."""
    global _broadcaster
    if _broadcaster is None:
        with _broadcaster_lock:
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
