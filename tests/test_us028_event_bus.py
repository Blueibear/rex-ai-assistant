"""
US-028: Event bus

Acceptance criteria:
- events published
- subscribers receive events
- event propagation works
- Typecheck passes
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from rex.event_bus import Event, EventBus, get_event_bus, set_event_bus


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Give each test a fresh event bus."""
    set_event_bus(EventBus())
    yield
    set_event_bus(EventBus())


# -----------------------------------------------------------------------
# Events published
# -----------------------------------------------------------------------


def test_publish_legacy_api_returns_event():
    bus = EventBus()
    result = bus.publish("test.event", {"key": "value"})
    assert isinstance(result, Event)
    assert result.event_type == "test.event"
    assert result.payload == {"key": "value"}


def test_publish_rich_api_with_event_object():
    bus = EventBus()
    event = Event(event_type="rich.event", payload={"x": 1})
    bus.publish(event)
    metrics = bus.get_metrics()
    assert metrics["published_events"] == 1


def test_publish_increments_event_count():
    bus = EventBus()
    bus.publish("a.event", {})
    bus.publish("b.event", {})
    assert bus.get_metrics()["published_events"] == 2


def test_publish_with_empty_payload():
    bus = EventBus()
    result = bus.publish("empty.payload", {})
    assert result.payload == {}


def test_publish_event_has_timestamp():
    bus = EventBus()
    result = bus.publish("ts.event", {})
    from datetime import datetime

    assert isinstance(result.timestamp, datetime)


# -----------------------------------------------------------------------
# Subscribers receive events
# -----------------------------------------------------------------------


def test_subscriber_receives_event_legacy():
    bus = EventBus()
    received: list[tuple[str, dict[str, Any]]] = []

    def callback(event_type: str, payload: dict[str, Any]) -> None:
        received.append((event_type, payload))

    bus.subscribe("greet.event", callback)
    bus.publish("greet.event", {"msg": "hello"})

    assert len(received) == 1
    assert received[0] == ("greet.event", {"msg": "hello"})


def test_subscriber_receives_event_rich_api():
    bus = EventBus()
    received: list[Event] = []

    def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("user.action", handler)
    bus.publish(Event(event_type="user.action", payload={"id": 42}))

    assert len(received) == 1
    assert received[0].event_type == "user.action"
    assert received[0].payload["id"] == 42


def test_multiple_subscribers_all_receive_event():
    bus = EventBus()
    count: list[int] = [0]

    def h1(event: Event) -> None:
        count[0] += 1

    def h2(event: Event) -> None:
        count[0] += 10

    bus.subscribe("shared.event", h1)
    bus.subscribe("shared.event", h2)
    bus.publish(Event(event_type="shared.event", payload={}))

    assert count[0] == 11


def test_subscriber_only_receives_matching_event_type():
    bus = EventBus()
    received: list[str] = []

    def handler(event: Event) -> None:
        received.append(event.event_type)

    bus.subscribe("target.event", handler)
    bus.publish(Event(event_type="other.event", payload={}))
    bus.publish(Event(event_type="target.event", payload={}))

    assert received == ["target.event"]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received: list[Event] = []

    def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("unsub.event", handler)
    bus.publish(Event(event_type="unsub.event", payload={}))
    assert len(received) == 1

    result = bus.unsubscribe("unsub.event", handler)
    assert result is True

    bus.publish(Event(event_type="unsub.event", payload={}))
    assert len(received) == 1  # no new event received


def test_legacy_unsubscribe_callable_stops_delivery():
    bus = EventBus()
    received: list[Any] = []

    def callback(event_type: str, payload: dict[str, Any]) -> None:
        received.append(event_type)

    unsub = bus.subscribe("legacy.event", callback)
    bus.publish("legacy.event", {})
    assert len(received) == 1

    unsub()
    bus.publish("legacy.event", {})
    assert len(received) == 1  # no new event


# -----------------------------------------------------------------------
# Event propagation works
# -----------------------------------------------------------------------


def test_wildcard_subscription_receives_all_events():
    bus = EventBus()
    received: list[str] = []

    def wildcard_handler(event: Event) -> None:
        received.append(event.event_type)

    bus.subscribe("*", wildcard_handler)
    bus.publish(Event(event_type="alpha.event", payload={}))
    bus.publish(Event(event_type="beta.event", payload={}))
    bus.publish(Event(event_type="gamma.event", payload={}))

    assert received == ["alpha.event", "beta.event", "gamma.event"]


def test_wildcard_and_specific_both_receive_event():
    bus = EventBus()
    wildcard_hits: list[Event] = []
    specific_hits: list[Event] = []

    bus.subscribe("*", lambda e: wildcard_hits.append(e))
    bus.subscribe("specific.event", lambda e: specific_hits.append(e))

    bus.publish(Event(event_type="specific.event", payload={}))

    assert len(wildcard_hits) == 1
    assert len(specific_hits) == 1


def test_bad_handler_does_not_break_propagation():
    bus = EventBus()
    good_received: list[Event] = []

    def bad_handler(event: Event) -> None:
        raise RuntimeError("handler failed")

    def good_handler(event: Event) -> None:
        good_received.append(event)

    bus.subscribe("prop.event", bad_handler)
    bus.subscribe("prop.event", good_handler)
    bus.publish(Event(event_type="prop.event", payload={}))

    assert len(good_received) == 1
    assert bus.get_metrics()["handler_errors"] == 1


def test_event_propagation_is_thread_safe():
    bus = EventBus()
    received: list[Event] = []
    lock = threading.Lock()

    def handler(event: Event) -> None:
        with lock:
            received.append(event)

    bus.subscribe("thread.event", handler)

    threads = [
        threading.Thread(
            target=bus.publish, args=(Event(event_type="thread.event", payload={"i": i}),)
        )
        for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == 20


def test_global_event_bus_is_singleton():
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2


def test_global_event_bus_propagates_events():
    received: list[Event] = []

    def handler(event: Event) -> None:
        received.append(event)

    bus = get_event_bus()
    bus.subscribe("global.event", handler)
    bus.publish(Event(event_type="global.event", payload={"src": "global"}))

    assert len(received) == 1
    assert received[0].payload["src"] == "global"
