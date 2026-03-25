"""Tests for US-P5-007: Wire HA event subscriptions through OpenClaw.

Acceptance criteria:
  - HaEventSubscriber subscribes to "ha.command" events via EventBridge
  - Publishing an ha.command event with valid payload calls ha_call_service
  - Subscribing multiple times is idempotent
  - unsubscribe() removes the handler and stops dispatching
  - Invalid payloads (missing required fields) are logged and ignored
  - is_subscribed property reflects subscription state
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from rex.openclaw.event_bus import EventBus
from rex.openclaw.event_bridge import EventBridge
from rex.openclaw.ha_event_subscriber import HA_COMMAND_EVENT, HaEventSubscriber

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_bridge() -> EventBridge:
    """Return an EventBridge wrapping a fresh isolated EventBus."""
    return EventBridge(bus=EventBus())


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestHaEventSubscriberInstantiation:
    def test_import(self):
        from rex.openclaw import ha_event_subscriber  # noqa: F401

    def test_constructs_without_args(self):
        with patch("rex.openclaw.ha_event_subscriber._EventBridge") as mock_cls:
            mock_cls.return_value = MagicMock()
            subscriber = HaEventSubscriber()
        assert subscriber is not None

    def test_accepts_explicit_bus(self):
        bridge = _fresh_bridge()
        subscriber = HaEventSubscriber(bus=bridge)
        assert subscriber._bus is bridge

    def test_accepts_raw_event_bus(self):
        bus = EventBus()
        subscriber = HaEventSubscriber(bus=bus)
        assert subscriber._bus is bus

    def test_not_subscribed_initially(self):
        subscriber = HaEventSubscriber(bus=_fresh_bridge())
        assert subscriber.is_subscribed is False

    def test_ha_command_event_constant(self):
        assert HA_COMMAND_EVENT == "ha.command"


# ---------------------------------------------------------------------------
# subscribe / unsubscribe
# ---------------------------------------------------------------------------


class TestSubscribeUnsubscribe:
    def test_subscribe_sets_is_subscribed(self):
        subscriber = HaEventSubscriber(bus=_fresh_bridge())
        subscriber.subscribe()
        assert subscriber.is_subscribed is True

    def test_subscribe_idempotent(self):
        bridge = _fresh_bridge()
        subscriber = HaEventSubscriber(bus=bridge)
        subscriber.subscribe()
        subscriber.subscribe()
        # Only one handler registered — subscription count should be 1
        assert bridge.get_subscription_count(HA_COMMAND_EVENT) == 1

    def test_unsubscribe_clears_is_subscribed(self):
        subscriber = HaEventSubscriber(bus=_fresh_bridge())
        subscriber.subscribe()
        subscriber.unsubscribe()
        assert subscriber.is_subscribed is False

    def test_unsubscribe_idempotent(self):
        subscriber = HaEventSubscriber(bus=_fresh_bridge())
        subscriber.unsubscribe()  # not subscribed yet — must not raise
        assert subscriber.is_subscribed is False

    def test_subscribe_registers_on_bus(self):
        bridge = _fresh_bridge()
        subscriber = HaEventSubscriber(bus=bridge)
        assert bridge.get_subscription_count(HA_COMMAND_EVENT) == 0
        subscriber.subscribe()
        assert bridge.get_subscription_count(HA_COMMAND_EVENT) == 1

    def test_unsubscribe_deregisters_from_bus(self):
        bridge = _fresh_bridge()
        subscriber = HaEventSubscriber(bus=bridge)
        subscriber.subscribe()
        subscriber.unsubscribe()
        assert bridge.get_subscription_count(HA_COMMAND_EVENT) == 0


# ---------------------------------------------------------------------------
# Event dispatch — legacy (event_type, payload) form
# ---------------------------------------------------------------------------


class TestDispatchLegacyForm:
    """Events published as (event_type, payload) dict trigger ha_call_service."""

    def _make_subscriber(self) -> tuple[HaEventSubscriber, EventBridge]:
        bridge = _fresh_bridge()
        subscriber = HaEventSubscriber(bus=bridge)
        subscriber.subscribe()
        return subscriber, bridge

    def test_turn_on_light_dispatched(self):
        _sub, bridge = self._make_subscriber()

        with patch(
            "rex.openclaw.ha_event_subscriber.ha_call_service",
            return_value={"success": True, "message": "Done.", "entity_id": "light.living_room"},
        ) as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {"domain": "light", "service": "turn_on", "entity_id": "light.living_room"},
            )

        mock_call.assert_called_once_with(
            domain="light",
            service="turn_on",
            entity_id="light.living_room",
            data=None,
        )

    def test_turn_off_switch_dispatched(self):
        _sub, bridge = self._make_subscriber()

        with patch(
            "rex.openclaw.ha_event_subscriber.ha_call_service",
            return_value={"success": True, "message": "Done.", "entity_id": "switch.garage"},
        ) as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {"domain": "switch", "service": "turn_off", "entity_id": "switch.garage"},
            )

        mock_call.assert_called_once_with(
            domain="switch",
            service="turn_off",
            entity_id="switch.garage",
            data=None,
        )

    def test_extra_data_forwarded(self):
        _sub, bridge = self._make_subscriber()

        with patch(
            "rex.openclaw.ha_event_subscriber.ha_call_service",
            return_value={"success": True, "message": "Done.", "entity_id": "light.bedroom"},
        ) as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {
                    "domain": "light",
                    "service": "turn_on",
                    "entity_id": "light.bedroom",
                    "data": {"brightness_pct": 60},
                },
            )

        mock_call.assert_called_once_with(
            domain="light",
            service="turn_on",
            entity_id="light.bedroom",
            data={"brightness_pct": 60},
        )

    def test_failed_service_call_logs_warning(self, caplog):
        import logging

        _sub, bridge = self._make_subscriber()

        with patch(
            "rex.openclaw.ha_event_subscriber.ha_call_service",
            return_value={"success": False, "message": "Entity not found.", "entity_id": "light.x"},
        ):
            with caplog.at_level(logging.WARNING):
                bridge.publish(
                    HA_COMMAND_EVENT,
                    {"domain": "light", "service": "turn_on", "entity_id": "light.x"},
                )

        assert any("failed" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Invalid payload — missing required fields
# ---------------------------------------------------------------------------


class TestInvalidPayload:
    def _make_subscriber(self) -> tuple[HaEventSubscriber, EventBridge]:
        bridge = _fresh_bridge()
        subscriber = HaEventSubscriber(bus=bridge)
        subscriber.subscribe()
        return subscriber, bridge

    def test_missing_domain_ignored(self):
        _sub, bridge = self._make_subscriber()

        with patch("rex.openclaw.ha_event_subscriber.ha_call_service") as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {"service": "turn_on", "entity_id": "light.x"},
            )

        mock_call.assert_not_called()

    def test_missing_service_ignored(self):
        _sub, bridge = self._make_subscriber()

        with patch("rex.openclaw.ha_event_subscriber.ha_call_service") as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {"domain": "light", "entity_id": "light.x"},
            )

        mock_call.assert_not_called()

    def test_missing_entity_id_ignored(self):
        _sub, bridge = self._make_subscriber()

        with patch("rex.openclaw.ha_event_subscriber.ha_call_service") as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {"domain": "light", "service": "turn_on"},
            )

        mock_call.assert_not_called()

    def test_empty_payload_ignored(self):
        _sub, bridge = self._make_subscriber()

        with patch("rex.openclaw.ha_event_subscriber.ha_call_service") as mock_call:
            bridge.publish(HA_COMMAND_EVENT, {})

        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Unsubscribed — no dispatch
# ---------------------------------------------------------------------------


class TestUnsubscribedNoDispatch:
    def test_no_dispatch_after_unsubscribe(self):
        bridge = _fresh_bridge()
        subscriber = HaEventSubscriber(bus=bridge)
        subscriber.subscribe()
        subscriber.unsubscribe()

        with patch("rex.openclaw.ha_event_subscriber.ha_call_service") as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {"domain": "light", "service": "turn_on", "entity_id": "light.x"},
            )

        mock_call.assert_not_called()

    def test_no_dispatch_when_never_subscribed(self):
        bridge = _fresh_bridge()
        HaEventSubscriber(bus=bridge)  # not subscribed

        with patch("rex.openclaw.ha_event_subscriber.ha_call_service") as mock_call:
            bridge.publish(
                HA_COMMAND_EVENT,
                {"domain": "switch", "service": "turn_off", "entity_id": "switch.x"},
            )

        mock_call.assert_not_called()
