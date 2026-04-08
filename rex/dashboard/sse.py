"""Server-Sent Events status broker for Rex (US-062).

Provides a lightweight pub-sub mechanism for streaming Rex status changes to
connected browser clients.  Status events flow from the voice loop (or any
other component) through :func:`emit_status`, then out to all subscribed SSE
clients.

Public API
----------
- :data:`RexStatus`          — enum of valid status strings
- :func:`emit_status`        — publish a new status to all clients
- :func:`get_current_status` — return the most recently emitted status
- :func:`subscribe`          — context manager; yields a per-client queue
- :func:`unsubscribe`        — remove a client queue from the broker
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RexStatus:
    """Enumeration of valid Rex status strings."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    DONE = "done"
    ERROR = "error"


_lock = threading.Lock()
_clients: list[queue.Queue[str]] = []
_current_status: str = RexStatus.IDLE


def emit_status(status: str) -> None:
    """Publish *status* to all subscribed SSE clients.

    Args:
        status: One of the :class:`RexStatus` constants (or any string).
    """
    global _current_status
    with _lock:
        _current_status = status
        for client_q in _clients:
            try:
                client_q.put_nowait(status)
            except queue.Full:
                pass  # drop if client is slow
    logger.debug("emit_status: %s (%d clients)", status, len(_clients))


def get_current_status() -> str:
    """Return the most recently emitted status."""
    return _current_status


def subscribe() -> queue.Queue[str]:
    """Register a new SSE client and return its event queue."""
    client_q: queue.Queue[str] = queue.Queue(maxsize=20)
    with _lock:
        _clients.append(client_q)
    logger.debug("SSE client subscribed (total: %d)", len(_clients))
    return client_q


def unsubscribe(client_q: queue.Queue[str]) -> None:
    """Remove *client_q* from the broker."""
    with _lock:
        try:
            _clients.remove(client_q)
        except ValueError:
            pass
    logger.debug("SSE client unsubscribed (total: %d)", len(_clients))


@contextmanager
def subscription() -> Generator[queue.Queue[str], None, None]:
    """Context manager that subscribes and auto-unsubscribes on exit."""
    client_q = subscribe()
    try:
        yield client_q
    finally:
        unsubscribe(client_q)


def _reset_for_tests() -> None:
    """Reset broker state — for use in tests only."""
    global _current_status, _clients
    with _lock:
        _clients = []
        _current_status = RexStatus.IDLE


__all__ = [
    "RexStatus",
    "emit_status",
    "get_current_status",
    "subscribe",
    "subscription",
    "unsubscribe",
]
