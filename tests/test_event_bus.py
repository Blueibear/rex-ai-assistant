"""
Tests for the event bus.

This test module supports BOTH event bus variants that have appeared in the
codebase:

Variant A (legacy/simple bus):
- EventBus.publish(event_type: str, payload: dict) -> Event
- EventBus.subscribe(event_type: str, callback(event_type, payload)) -> unsubscribe callable
- Wildcard subscription via "*"

Variant B (newer/thread-safe bus):
- EventBus.publish(event: Event) -> None
- EventBus.subscribe(event_type: str, handler(Event)) -> None
- Event has (event_type, payload, timestamp) and __repr__
- Wildcard subscription via "*"
- unsubscribe(event_type, handler) -> bool
- get_stats(), get_subscription_count(), clear_subscriptions()

The tests auto-detect the available API at runtime and skip incompatible tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from rex.event_bus import EventBus


def _event_class():
    try:
        from rex.event_bus import Event  # type: ignore

        return Event
    except Exception:
        return None


def _is_variant_a() -> bool:
    """
    Detect legacy API by checking if publish accepts (event_type, payload).
    """
    bus = EventBus()
    try:
        bus.publish("test.event", {"ok": True})  # type: ignore[arg-type]
        return True
    except TypeError:
        return False
    except Exception:
        # If it accepted the signature but failed for some other reason, treat as A.
        return True


def _is_variant_b() -> bool:
    """
    Detect newer API by checking if publish accepts a single Event object.
    """
    Event = _event_class()
    if Event is None:
        return False
    bus = EventBus()
    try:
        bus.publish(Event(event_type="test.event", payload={}))  # type: ignore[arg-type]
        return True
    except TypeError:
        return False
    except Exception:
        return True


# -------------------------------------------------------------------
# Variant A tests (legacy/simple publish-subscribe)
# -------------------------------------------------------------------


@pytest.mark.skipif(
    not _is_variant_a(),
    reason="Legacy EventBus.publish(event_type, payload) API not available in this build.",
)
def test_event_bus_publish_and_subscribe_variant_a() -> None:
    bus = EventBus()
    received: list[tuple[str, dict[str, object]]] = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        received.append((event_type, payload))

    bus.subscribe("email.triaged", handler)  # type: ignore[arg-type]
    bus.publish("email.triaged", {"count": 2})  # type: ignore[arg-type]

    assert received == [("email.triaged", {"count": 2})]


@pytest.mark.skipif(
    not _is_variant_a(),
    reason="Legacy EventBus.publish(event_type, payload) API not available in this build.",
)
def test_event_bus_wildcard_variant_a() -> None:
    bus = EventBus()
    received: list[tuple[str, dict[str, object]]] = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        received.append((event_type, payload))

    bus.subscribe("*", handler)  # type: ignore[arg-type]
    bus.publish("calendar.created", {"event_id": "123"})  # type: ignore[arg-type]

    assert received == [("calendar.created", {"event_id": "123"})]


# -------------------------------------------------------------------
# Variant B fixtures and tests (newer/thread-safe Event object bus)
# -------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_event_creation_variant_b() -> None:
    Event = _event_class()
    assert Event is not None

    event = Event(event_type="test.event", payload={"key": "value"})
    assert event.event_type == "test.event"
    assert event.payload == {"key": "value"}
    assert isinstance(event.timestamp, datetime)


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_event_repr_variant_b() -> None:
    Event = _event_class()
    assert Event is not None

    event = Event(event_type="test.event", payload={})
    repr_str = repr(event)
    assert "test.event" in repr_str
    assert event.timestamp.isoformat() in repr_str


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_event_bus_initialization_variant_b(event_bus: EventBus) -> None:
    assert hasattr(event_bus, "_subscriptions")
    assert getattr(event_bus, "_event_count", 0) == 0
    assert getattr(event_bus, "_error_count", 0) == 0


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_subscribe_variant_b(event_bus: EventBus) -> None:
    def handler(event):
        return None

    event_bus.subscribe("test.event", handler)  # type: ignore[arg-type]
    assert event_bus.get_subscription_count("test.event") == 1  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_unsubscribe_variant_b(event_bus: EventBus) -> None:
    def handler(event):
        return None

    event_bus.subscribe("test.event", handler)  # type: ignore[arg-type]
    assert event_bus.get_subscription_count("test.event") == 1  # type: ignore[attr-defined]

    result = event_bus.unsubscribe("test.event", handler)  # type: ignore[attr-defined]
    assert result is True
    assert event_bus.get_subscription_count("test.event") == 0  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_publish_to_handler_variant_b(event_bus: EventBus) -> None:
    Event = _event_class()
    assert Event is not None

    received_events: list[Any] = []

    def handler(event):
        received_events.append(event)

    event_bus.subscribe("test.event", handler)  # type: ignore[arg-type]
    event_bus.publish(Event(event_type="test.event", payload={"data": "test"}))  # type: ignore[arg-type]

    assert len(received_events) == 1
    assert received_events[0].event_type == "test.event"
    assert received_events[0].payload["data"] == "test"


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_wildcard_subscription_variant_b(event_bus: EventBus) -> None:
    Event = _event_class()
    assert Event is not None

    received_events: list[Any] = []

    def wildcard_handler(event):
        received_events.append(event)

    event_bus.subscribe("*", wildcard_handler)  # type: ignore[arg-type]

    event_bus.publish(Event(event_type="test.event1", payload={}))  # type: ignore[arg-type]
    event_bus.publish(Event(event_type="test.event2", payload={}))  # type: ignore[arg-type]

    assert len(received_events) == 2
    assert received_events[0].event_type == "test.event1"
    assert received_events[1].event_type == "test.event2"


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_handler_error_handling_variant_b(event_bus: EventBus) -> None:
    Event = _event_class()
    assert Event is not None

    good_handler_events: list[Any] = []

    def bad_handler(event):
        raise ValueError("Handler error")

    def good_handler(event):
        good_handler_events.append(event)

    event_bus.subscribe("test.event", bad_handler)  # type: ignore[arg-type]
    event_bus.subscribe("test.event", good_handler)  # type: ignore[arg-type]

    event_bus.publish(Event(event_type="test.event", payload={}))  # type: ignore[arg-type]

    assert len(good_handler_events) == 1
    stats = event_bus.get_stats()  # type: ignore[attr-defined]
    assert stats["error_count"] == 1


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_duplicate_subscription_prevented_variant_b(event_bus: EventBus) -> None:
    """Subscribing the same handler twice should register it only once."""

    def handler(event):
        return None

    event_bus.subscribe("test.event", handler)  # type: ignore[arg-type]
    event_bus.subscribe("test.event", handler)  # type: ignore[arg-type]
    assert event_bus.get_subscription_count("test.event") == 1  # type: ignore[attr-defined]


@pytest.mark.skipif(
    not _is_variant_a(),
    reason="Legacy EventBus.publish(event_type, payload) API not available in this build.",
)
def test_duplicate_subscription_prevented_variant_a() -> None:
    """Legacy: subscribing the same callback twice should register it only once."""
    bus = EventBus()
    received: list[tuple[str, dict[str, object]]] = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        received.append((event_type, payload))

    bus.subscribe("test.event", handler)  # type: ignore[arg-type]
    bus.subscribe("test.event", handler)  # type: ignore[arg-type]
    bus.publish("test.event", {"x": 1})  # type: ignore[arg-type]
    # Should only receive the event once, not twice
    assert len(received) == 1


@pytest.mark.skipif(
    not _is_variant_b(),
    reason="Newer EventBus.publish(Event) API not available in this build.",
)
def test_clear_subscriptions_variant_b(event_bus: EventBus) -> None:
    def handler(event):
        return None

    event_bus.subscribe("test.event1", handler)  # type: ignore[arg-type]
    event_bus.subscribe("test.event2", handler)  # type: ignore[arg-type]

    event_bus.clear_subscriptions("test.event1")  # type: ignore[attr-defined]
    assert event_bus.get_subscription_count("test.event1") == 0  # type: ignore[attr-defined]
    assert event_bus.get_subscription_count("test.event2") == 1  # type: ignore[attr-defined]

    event_bus.clear_subscriptions()  # type: ignore[attr-defined]
    assert event_bus.get_subscription_count("test.event2") == 0  # type: ignore[attr-defined]
