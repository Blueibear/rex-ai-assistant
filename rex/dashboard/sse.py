"""Server-Sent Events (SSE) broadcaster for dashboard notifications.

Provides an in-process, thread-safe broadcaster that pushes notification
events to connected SSE clients.  The broadcaster uses stdlib only
(``queue``, ``threading``, ``time``) and has no external dependencies.

Usage::

    from rex.dashboard.sse import get_broadcaster

    broadcaster = get_broadcaster()

    # Subscribe (returns a generator suitable for Flask SSE responses)
    def sse_stream():
        for event in broadcaster.subscribe(timeout=30.0, max_events=None):
            yield event

    # Publish (called from DashboardStore.write or routes)
    broadcaster.publish({"id": "notif_1", "title": "Hello", ...})

Design notes:

- Each subscriber gets its own ``queue.Queue`` that receives copies of
  published events.  When the subscriber disconnects (generator is
  garbage-collected or explicitly closed) the queue is automatically
  removed.
- ``subscribe()`` yields SSE-formatted ``data: ...\\n\\n`` strings.
- The broadcaster is a singleton obtained via ``get_broadcaster()``.
- ``max_events`` and ``timeout`` parameters on ``subscribe`` are
  intended for testing; production clients typically leave them unset.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class SSEBroadcaster:
    """Thread-safe in-process broadcaster for SSE notification events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[str | None]] = []

    def publish(self, event_data: dict[str, Any]) -> int:
        """Publish an event to all connected subscribers.

        Args:
            event_data: Dictionary payload to send as a JSON SSE event.

        Returns:
            Number of subscribers that received the event.
        """
        payload = f"data: {json.dumps(event_data)}\n\n"
        delivered = 0
        with self._lock:
            dead: list[queue.Queue[str | None]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                    delivered += 1
                except queue.Full:
                    # Subscriber is too slow; drop them
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)
        if delivered:
            logger.debug("SSE broadcast delivered to %d subscriber(s)", delivered)
        return delivered

    def subscribe(
        self,
        *,
        timeout: float = 30.0,
        max_events: int | None = None,
        initial_data: dict[str, Any] | None = None,
    ):
        """Yield SSE-formatted event strings.

        This is a generator intended to be returned from a Flask SSE
        endpoint.

        Args:
            timeout: Seconds to wait for an event before sending a
                keep-alive comment.  Also controls how quickly the
                generator notices a ``max_events`` limit or shutdown.
            max_events: Maximum number of data events to yield before
                stopping.  ``None`` means unlimited (production default).
            initial_data: Optional initial payload sent as the first
                event (e.g. current unread count).
        """
        q: queue.Queue[str | None] = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        try:
            # Send initial data event if provided
            if initial_data is not None:
                yield f"data: {json.dumps(initial_data)}\n\n"

            event_count = 0
            while True:
                if max_events is not None and event_count >= max_events:
                    break
                try:
                    item = q.get(timeout=timeout)
                except queue.Empty:
                    # Send keep-alive comment
                    yield f": keepalive {int(time.time())}\n\n"
                    continue

                if item is None:
                    # Sentinel: shutdown
                    break

                yield item
                event_count += 1
        finally:
            with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def shutdown(self) -> None:
        """Signal all subscribers to disconnect."""
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        """Return the current number of active subscribers."""
        with self._lock:
            return len(self._subscribers)


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_broadcaster: SSEBroadcaster | None = None
_broadcaster_lock = threading.Lock()


def get_broadcaster() -> SSEBroadcaster:
    """Get the global SSE broadcaster instance (lazy singleton)."""
    global _broadcaster
    if _broadcaster is None:
        with _broadcaster_lock:
            if _broadcaster is None:
                _broadcaster = SSEBroadcaster()
    return _broadcaster


def set_broadcaster(broadcaster: SSEBroadcaster | None) -> None:
    """Replace the global SSE broadcaster (for testing)."""
    global _broadcaster
    _broadcaster = broadcaster


__all__ = [
    "SSEBroadcaster",
    "get_broadcaster",
    "set_broadcaster",
]
