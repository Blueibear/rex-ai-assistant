"""
Tests for EventQueue (US-067: Event queue stability).

Acceptance criteria:
- events queued safely
- queue overflow prevented
- events processed sequentially
- Typecheck passes
"""

from __future__ import annotations

import threading
import time

import pytest

from rex.openclaw.event_bus import Event, EventBus, EventQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_queue(maxsize: int = 100) -> tuple[EventQueue, list[Event]]:
    """Return a started EventQueue and the list it delivers events into."""
    bus = EventBus()
    received: list[Event] = []

    def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("test", handler)
    eq = EventQueue(bus, maxsize=maxsize)
    eq.start()
    return eq, received


# ---------------------------------------------------------------------------
# Basic queuing
# ---------------------------------------------------------------------------


def test_enqueue_returns_true_when_space_available() -> None:
    eq, received = _make_queue()
    event = Event(event_type="test", payload={"n": 1})
    result = eq.enqueue(event)
    eq.join()
    eq.stop()
    assert result is True
    assert len(received) == 1
    assert received[0].payload["n"] == 1


def test_events_are_delivered_to_bus_handlers() -> None:
    eq, received = _make_queue()
    for i in range(5):
        eq.enqueue(Event(event_type="test", payload={"n": i}))
    eq.join()
    eq.stop()
    assert len(received) == 5


# ---------------------------------------------------------------------------
# Sequential processing
# ---------------------------------------------------------------------------


def test_events_processed_sequentially() -> None:
    """Handler should see events in the order they were enqueued."""
    bus = EventBus()
    order: list[int] = []
    lock = threading.Lock()

    def handler(event: Event) -> None:
        with lock:
            order.append(event.payload["n"])

    bus.subscribe("test", handler)
    eq = EventQueue(bus, maxsize=50)
    eq.start()

    for i in range(20):
        eq.enqueue(Event(event_type="test", payload={"n": i}))

    eq.join()
    eq.stop()

    assert order == list(range(20))


# ---------------------------------------------------------------------------
# Overflow prevention
# ---------------------------------------------------------------------------


def test_overflow_returns_false_and_drops_event() -> None:
    """When the queue is full, enqueue returns False and drops the event."""
    bus = EventBus()
    barrier = threading.Event()
    received: list[Event] = []

    def slow_handler(event: Event) -> None:
        # Block until released so the queue fills up
        barrier.wait()
        received.append(event)

    bus.subscribe("test", slow_handler)
    maxsize = 3
    eq = EventQueue(bus, maxsize=maxsize)
    eq.start()

    # First event goes to the worker (not in queue), remaining fill the queue
    results = []
    for _ in range(maxsize + 5):  # clearly more than maxsize
        results.append(eq.enqueue(Event(event_type="test", payload={})))
        time.sleep(0.001)  # tiny sleep so worker picks up first event

    # At least one enqueue should have been dropped
    assert False in results

    # Cleanup
    barrier.set()
    eq.join()
    eq.stop()


def test_metrics_track_dropped_events() -> None:
    """get_metrics() should report dropped events when overflow occurs."""
    bus = EventBus()
    barrier = threading.Event()

    def slow_handler(event: Event) -> None:
        barrier.wait()

    bus.subscribe("test", slow_handler)
    maxsize = 2
    eq = EventQueue(bus, maxsize=maxsize)
    eq.start()

    # Flood the queue
    for _ in range(maxsize + 10):
        eq.enqueue(Event(event_type="test", payload={}))
        time.sleep(0.001)

    metrics = eq.get_metrics()
    barrier.set()
    eq.join()
    eq.stop()

    assert metrics["dropped"] > 0
    assert metrics["maxsize"] == maxsize


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_stop_idempotent() -> None:
    bus = EventBus()
    eq = EventQueue(bus)
    eq.start()
    eq.start()  # second start is a no-op
    assert eq.is_running is True
    eq.stop()
    eq.stop()  # second stop is a no-op
    assert eq.is_running is False


def test_invalid_maxsize_raises() -> None:
    bus = EventBus()
    with pytest.raises(ValueError):
        EventQueue(bus, maxsize=0)
    with pytest.raises(ValueError):
        EventQueue(bus, maxsize=-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_track_processed_count() -> None:
    eq, received = _make_queue()
    n = 10
    for i in range(n):
        eq.enqueue(Event(event_type="test", payload={"n": i}))
    eq.join()
    eq.stop()

    metrics = eq.get_metrics()
    assert metrics["processed"] == n
    assert metrics["dropped"] == 0


def test_qsize_reflects_queue_depth() -> None:
    bus = EventBus()
    barrier = threading.Event()

    def slow_handler(event: Event) -> None:
        barrier.wait()

    bus.subscribe("test", slow_handler)
    eq = EventQueue(bus, maxsize=50)
    eq.start()

    # Let the worker pick up the first event, then queue a few more
    eq.enqueue(Event(event_type="test", payload={}))
    time.sleep(0.05)  # worker picks up the first event

    for _ in range(5):
        eq.enqueue(Event(event_type="test", payload={}))

    assert eq.qsize > 0

    barrier.set()
    eq.join()
    eq.stop()
