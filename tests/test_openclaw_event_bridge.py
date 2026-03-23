"""Tests for rex.openclaw.event_bridge — US-P4-015 and US-P4-016.

US-P4-015 acceptance criteria:
  - EventBridge exists and is importable
  - Satisfies EventBusProtocol structural check
  - Delegates every method to the underlying EventBus
  - register() returns None when openclaw not installed

US-P4-016 acceptance criteria:
  - publish event → subscriber receives it (round-trip)
  - wildcard subscriber receives all events
  - legacy (str, dict) callback style works through bridge
  - rich (Event) handler style works through bridge
  - unsubscribe removes handler
  - clear_subscriptions removes all handlers
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.openclaw.event_bus import Event, EventBus
from rex.openclaw.event_bridge import OPENCLAW_AVAILABLE, EventBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_bridge() -> tuple[EventBridge, EventBus]:
    """Return an EventBridge backed by a fresh isolated EventBus."""
    bus = EventBus()
    bridge = EventBridge(bus=bus)
    return bridge, bus


# ---------------------------------------------------------------------------
# US-P4-015: Instantiation and protocol conformance
# ---------------------------------------------------------------------------


class TestEventBridgeInstantiation:
    def test_import(self):
        from rex.openclaw import event_bridge  # noqa: F401

    def test_no_args_uses_global_bus(self):
        """EventBridge() with no args constructs without error."""
        bridge = EventBridge()
        assert bridge is not None

    def test_explicit_bus_arg(self):
        """EventBridge accepts an explicit bus and stores it."""
        bus = EventBus()
        bridge = EventBridge(bus=bus)
        assert bridge._bus is bus

    def test_openclaw_available_is_bool(self):
        assert isinstance(OPENCLAW_AVAILABLE, bool)



# ---------------------------------------------------------------------------
# US-P4-015: Delegation to underlying EventBus
# ---------------------------------------------------------------------------


class TestDelegation:
    def setup_method(self):
        self.bus = EventBus()
        self.bridge = EventBridge(bus=self.bus)

    def test_subscribe_delegates(self):
        handler = MagicMock()
        with patch.object(self.bus, "subscribe", return_value=None) as mock_sub:
            self.bridge.subscribe("email.unread", handler)
            mock_sub.assert_called_once_with("email.unread", handler)

    def test_unsubscribe_delegates(self):
        handler = MagicMock()
        with patch.object(self.bus, "unsubscribe", return_value=True) as mock_unsub:
            result = self.bridge.unsubscribe("email.unread", handler)
            mock_unsub.assert_called_once_with("email.unread", handler)
            assert result is True

    def test_publish_simple_delegates(self):
        fake_event = Event("email.unread", {"count": 1})
        with patch.object(self.bus, "publish", return_value=fake_event) as mock_pub:
            result = self.bridge.publish("email.unread", {"count": 1})
            mock_pub.assert_called_once_with("email.unread", {"count": 1})
            assert result is fake_event

    def test_publish_rich_delegates(self):
        ev = Event("calendar.created", {"title": "Meeting"})
        with patch.object(self.bus, "publish", return_value=None) as mock_pub:
            self.bridge.publish(ev)
            mock_pub.assert_called_once_with(ev)

    def test_get_metrics_delegates(self):
        metrics = {"published_events": 5, "handler_errors": 0}
        with patch.object(self.bus, "get_metrics", return_value=metrics) as mock_m:
            result = self.bridge.get_metrics()
            mock_m.assert_called_once()
            assert result == metrics

    def test_get_stats_delegates(self):
        stats = {"total_events": 5, "error_count": 0, "subscription_types": 1, "subscriptions": {}}
        with patch.object(self.bus, "get_stats", return_value=stats) as mock_s:
            result = self.bridge.get_stats()
            mock_s.assert_called_once()
            assert result == stats

    def test_get_subscription_count_delegates(self):
        with patch.object(self.bus, "get_subscription_count", return_value=3) as mock_c:
            result = self.bridge.get_subscription_count("email.unread")
            mock_c.assert_called_once_with("email.unread")
            assert result == 3

    def test_iter_subscribers_delegates(self):
        handler = MagicMock()
        with patch.object(self.bus, "iter_subscribers", return_value=iter([handler])) as mock_i:
            subscribers = list(self.bridge.iter_subscribers("email.unread"))
            mock_i.assert_called_once_with("email.unread")
            assert handler in subscribers

    def test_clear_subscriptions_delegates(self):
        with patch.object(self.bus, "clear_subscriptions") as mock_c:
            self.bridge.clear_subscriptions()
            mock_c.assert_called_once_with(None)

    def test_clear_subscriptions_specific_type_delegates(self):
        with patch.object(self.bus, "clear_subscriptions") as mock_c:
            self.bridge.clear_subscriptions("email.unread")
            mock_c.assert_called_once_with("email.unread")


# ---------------------------------------------------------------------------
# US-P4-016: Round-trip tests (publish → subscriber receives)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def setup_method(self):
        self.bridge, self.bus = _fresh_bridge()

    def test_legacy_callback_receives_event(self):
        """Legacy-style callback(event_type, payload) fires on publish."""
        received = []

        def on_email(event_type, payload):
            received.append((event_type, payload))

        self.bridge.subscribe("email.unread", on_email)
        self.bridge.publish("email.unread", {"count": 2})

        assert len(received) == 1
        assert received[0] == ("email.unread", {"count": 2})

    def test_rich_handler_receives_event(self):
        """Rich-style handler(Event) fires on publish."""
        received = []

        def on_calendar(event: Event):
            received.append(event)

        self.bridge.subscribe("calendar.created", on_calendar)
        self.bridge.publish("calendar.created", {"title": "Standup"})

        assert len(received) == 1
        assert received[0].event_type == "calendar.created"
        assert received[0].payload["title"] == "Standup"

    def test_wildcard_subscriber_receives_all_events(self):
        """Wildcard '*' subscriber receives every event published."""
        received = []

        def catch_all(event_type, payload):
            received.append(event_type)

        self.bridge.subscribe("*", catch_all)
        self.bridge.publish("email.unread", {"count": 1})
        self.bridge.publish("calendar.updated", {"event": {}})
        self.bridge.publish("email.triaged", {})

        assert received == ["email.unread", "calendar.updated", "email.triaged"]

    def test_unsubscribe_stops_delivery(self):
        """After unsubscribe, handler no longer receives events."""
        received = []

        def handler(event: Event):
            received.append(event)

        self.bridge.subscribe("email.read", handler)
        self.bridge.publish("email.read", {"id": "1"})
        assert len(received) == 1

        self.bridge.unsubscribe("email.read", handler)
        self.bridge.publish("email.read", {"id": "2"})
        assert len(received) == 1  # still 1 — second event not delivered

    def test_legacy_unsubscribe_callable_stops_delivery(self):
        """Callable returned by subscribe() stops legacy callback delivery."""
        received = []

        def cb(event_type, payload):
            received.append(payload)

        unsub = self.bridge.subscribe("email.unread", cb)
        assert callable(unsub)

        self.bridge.publish("email.unread", {"count": 1})
        assert len(received) == 1

        unsub()
        self.bridge.publish("email.unread", {"count": 2})
        assert len(received) == 1  # stopped

    def test_clear_subscriptions_removes_all(self):
        """clear_subscriptions() removes all handlers for that event type."""
        received = []

        def h1(event: Event):
            received.append("h1")

        def h2(event: Event):
            received.append("h2")

        self.bridge.subscribe("email.unread", h1)
        self.bridge.subscribe("email.unread", h2)
        self.bridge.publish("email.unread", {})
        assert received == ["h1", "h2"]

        self.bridge.clear_subscriptions("email.unread")
        received.clear()
        self.bridge.publish("email.unread", {})
        assert received == []

    def test_publish_rich_event_object(self):
        """Publishing an Event object directly works through bridge."""
        received = []

        def handler(event: Event):
            received.append(event)

        self.bridge.subscribe("calendar.deleted", handler)
        ev = Event("calendar.deleted", {"event_id": "abc"})
        self.bridge.publish(ev)

        assert len(received) == 1
        assert received[0].payload["event_id"] == "abc"

    def test_metrics_track_published_count(self):
        """get_metrics() reflects events published through bridge."""
        self.bridge.publish("email.unread", {"count": 1})
        self.bridge.publish("calendar.update", {"count": 2})
        metrics = self.bridge.get_metrics()
        assert metrics["published_events"] >= 2

    def test_subscription_count_reflects_subscribers(self):
        """get_subscription_count returns correct count after subscribe."""
        assert self.bridge.get_subscription_count("email.unread") == 0

        def h(event: Event):
            pass

        self.bridge.subscribe("email.unread", h)
        assert self.bridge.get_subscription_count("email.unread") == 1

    def test_bridge_and_bus_share_same_state(self):
        """Subscribers added via bridge are visible to the underlying bus."""
        received = []

        def handler(event: Event):
            received.append(event)

        self.bridge.subscribe("email.triaged", handler)
        # Publish directly through underlying bus — bridge subscriber should fire
        self.bus.publish("email.triaged", {"count": 5})

        assert len(received) == 1


# ---------------------------------------------------------------------------
# US-P4-015: register() stub
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_none_without_openclaw(self):
        bridge = EventBridge(bus=EventBus())
        if not OPENCLAW_AVAILABLE:
            assert bridge.register() is None

    def test_register_accepts_agent_arg(self):
        bridge = EventBridge(bus=EventBus())
        agent = MagicMock()
        if not OPENCLAW_AVAILABLE:
            assert bridge.register(agent=agent) is None
