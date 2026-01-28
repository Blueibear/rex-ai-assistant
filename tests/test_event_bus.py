"""Tests for event bus module."""

from datetime import datetime

import pytest

from rex.event_bus import Event, EventBus


@pytest.fixture
def event_bus():
    """Create a test event bus instance."""
    return EventBus()


def test_event_creation():
    """Test creating an event."""
    event = Event(
        event_type="test.event",
        payload={"key": "value"}
    )

    assert event.event_type == "test.event"
    assert event.payload == {"key": "value"}
    assert isinstance(event.timestamp, datetime)


def test_event_repr():
    """Test event string representation."""
    event = Event(
        event_type="test.event",
        payload={}
    )

    repr_str = repr(event)
    assert "test.event" in repr_str
    assert event.timestamp.isoformat() in repr_str


def test_event_bus_initialization(event_bus):
    """Test event bus initializes correctly."""
    assert event_bus._subscriptions == {}
    assert event_bus._event_count == 0
    assert event_bus._error_count == 0


def test_subscribe(event_bus):
    """Test subscribing to events."""
    def handler(event):
        pass

    event_bus.subscribe("test.event", handler)

    assert event_bus.get_subscription_count("test.event") == 1


def test_subscribe_multiple_handlers(event_bus):
    """Test subscribing multiple handlers to same event type."""
    def handler1(event):
        pass

    def handler2(event):
        pass

    event_bus.subscribe("test.event", handler1)
    event_bus.subscribe("test.event", handler2)

    assert event_bus.get_subscription_count("test.event") == 2


def test_subscribe_different_event_types(event_bus):
    """Test subscribing to different event types."""
    def handler1(event):
        pass

    def handler2(event):
        pass

    event_bus.subscribe("test.event1", handler1)
    event_bus.subscribe("test.event2", handler2)

    assert event_bus.get_subscription_count("test.event1") == 1
    assert event_bus.get_subscription_count("test.event2") == 1


def test_unsubscribe(event_bus):
    """Test unsubscribing from events."""
    def handler(event):
        pass

    event_bus.subscribe("test.event", handler)
    assert event_bus.get_subscription_count("test.event") == 1

    result = event_bus.unsubscribe("test.event", handler)
    assert result is True
    assert event_bus.get_subscription_count("test.event") == 0


def test_unsubscribe_nonexistent(event_bus):
    """Test unsubscribing a handler that doesn't exist."""
    def handler(event):
        pass

    result = event_bus.unsubscribe("test.event", handler)
    assert result is False


def test_publish_to_handler(event_bus):
    """Test publishing events to handlers."""
    received_events = []

    def handler(event):
        received_events.append(event)

    event_bus.subscribe("test.event", handler)

    event = Event(event_type="test.event", payload={"data": "test"})
    event_bus.publish(event)

    assert len(received_events) == 1
    assert received_events[0].event_type == "test.event"
    assert received_events[0].payload["data"] == "test"


def test_publish_to_multiple_handlers(event_bus):
    """Test publishing to multiple handlers."""
    handler1_events = []
    handler2_events = []

    def handler1(event):
        handler1_events.append(event)

    def handler2(event):
        handler2_events.append(event)

    event_bus.subscribe("test.event", handler1)
    event_bus.subscribe("test.event", handler2)

    event = Event(event_type="test.event", payload={})
    event_bus.publish(event)

    assert len(handler1_events) == 1
    assert len(handler2_events) == 1


def test_publish_no_handlers(event_bus):
    """Test publishing event with no handlers."""
    event = Event(event_type="test.event", payload={})
    # Should not raise error
    event_bus.publish(event)

    stats = event_bus.get_stats()
    assert stats['total_events'] == 1


def test_wildcard_subscription(event_bus):
    """Test wildcard subscription receives all events."""
    received_events = []

    def wildcard_handler(event):
        received_events.append(event)

    event_bus.subscribe("*", wildcard_handler)

    event1 = Event(event_type="test.event1", payload={})
    event2 = Event(event_type="test.event2", payload={})

    event_bus.publish(event1)
    event_bus.publish(event2)

    assert len(received_events) == 2
    assert received_events[0].event_type == "test.event1"
    assert received_events[1].event_type == "test.event2"


def test_wildcard_and_specific_subscription(event_bus):
    """Test that specific and wildcard handlers both receive events."""
    wildcard_events = []
    specific_events = []

    def wildcard_handler(event):
        wildcard_events.append(event)

    def specific_handler(event):
        specific_events.append(event)

    event_bus.subscribe("*", wildcard_handler)
    event_bus.subscribe("test.event", specific_handler)

    event = Event(event_type="test.event", payload={})
    event_bus.publish(event)

    # Both handlers should have received the event
    assert len(wildcard_events) == 1
    assert len(specific_events) == 1


def test_handler_error_handling(event_bus):
    """Test that handler errors don't affect other handlers."""
    good_handler_events = []

    def bad_handler(event):
        raise ValueError("Handler error")

    def good_handler(event):
        good_handler_events.append(event)

    event_bus.subscribe("test.event", bad_handler)
    event_bus.subscribe("test.event", good_handler)

    event = Event(event_type="test.event", payload={})
    event_bus.publish(event)

    # Good handler should still have received event
    assert len(good_handler_events) == 1

    # Error count should have increased
    stats = event_bus.get_stats()
    assert stats['error_count'] == 1


def test_get_subscription_count(event_bus):
    """Test getting subscription count."""
    def handler(event):
        pass

    assert event_bus.get_subscription_count("test.event") == 0

    event_bus.subscribe("test.event", handler)
    assert event_bus.get_subscription_count("test.event") == 1


def test_get_stats(event_bus):
    """Test getting event bus statistics."""
    def handler(event):
        pass

    event_bus.subscribe("test.event1", handler)
    event_bus.subscribe("test.event2", handler)

    event1 = Event(event_type="test.event1", payload={})
    event2 = Event(event_type="test.event2", payload={})

    event_bus.publish(event1)
    event_bus.publish(event2)

    stats = event_bus.get_stats()

    assert stats['total_events'] == 2
    assert stats['error_count'] == 0
    assert stats['subscription_types'] == 2
    assert stats['subscriptions']['test.event1'] == 1
    assert stats['subscriptions']['test.event2'] == 1


def test_clear_subscriptions_specific(event_bus):
    """Test clearing subscriptions for specific event type."""
    def handler(event):
        pass

    event_bus.subscribe("test.event1", handler)
    event_bus.subscribe("test.event2", handler)

    event_bus.clear_subscriptions("test.event1")

    assert event_bus.get_subscription_count("test.event1") == 0
    assert event_bus.get_subscription_count("test.event2") == 1


def test_clear_subscriptions_all(event_bus):
    """Test clearing all subscriptions."""
    def handler(event):
        pass

    event_bus.subscribe("test.event1", handler)
    event_bus.subscribe("test.event2", handler)

    event_bus.clear_subscriptions()

    assert event_bus.get_subscription_count("test.event1") == 0
    assert event_bus.get_subscription_count("test.event2") == 0


def test_event_payload_types(event_bus):
    """Test events with various payload types."""
    received_events = []

    def handler(event):
        received_events.append(event)

    event_bus.subscribe("test.event", handler)

    # String payload
    event1 = Event(event_type="test.event", payload={"data": "string"})
    event_bus.publish(event1)

    # Number payload
    event2 = Event(event_type="test.event", payload={"data": 123})
    event_bus.publish(event2)

    # List payload
    event3 = Event(event_type="test.event", payload={"data": [1, 2, 3]})
    event_bus.publish(event3)

    # Nested dict payload
    event4 = Event(event_type="test.event", payload={"data": {"nested": "value"}})
    event_bus.publish(event4)

    assert len(received_events) == 4
    assert received_events[0].payload["data"] == "string"
    assert received_events[1].payload["data"] == 123
    assert received_events[2].payload["data"] == [1, 2, 3]
    assert received_events[3].payload["data"]["nested"] == "value"


def test_concurrent_publishing(event_bus):
    """Test that event bus handles concurrent publishing safely."""
    import threading

    received_events = []
    lock = threading.Lock()

    def handler(event):
        with lock:
            received_events.append(event)

    event_bus.subscribe("test.event", handler)

    def publish_events(count):
        for i in range(count):
            event = Event(event_type="test.event", payload={"i": i})
            event_bus.publish(event)

    threads = []
    for _ in range(5):
        thread = threading.Thread(target=publish_events, args=(10,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Should have received 50 events total (5 threads * 10 events each)
    assert len(received_events) == 50
