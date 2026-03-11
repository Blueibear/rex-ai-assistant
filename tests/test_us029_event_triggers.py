"""Tests for US-029: Event triggers.

Acceptance criteria:
- triggers registered
- events trigger workflows
- errors logged
- Typecheck passes
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from rex.event_bus import Event, EventBus, set_event_bus
from rex.event_triggers import EventTriggerRegistry, get_trigger_registry, set_trigger_registry


@pytest.fixture(autouse=True)
def isolated_bus_and_registry():
    """Each test gets a fresh EventBus and EventTriggerRegistry."""
    bus = EventBus()
    set_event_bus(bus)
    registry = EventTriggerRegistry(bus=bus)
    set_trigger_registry(registry)
    yield bus, registry
    # Detach after test to avoid cross-test pollution
    registry.detach()


# ---------------------------------------------------------------------------
# Trigger registration
# ---------------------------------------------------------------------------


def test_register_trigger_stores_callable(isolated_bus_and_registry):
    """Triggers can be registered for an event type."""
    _, registry = isolated_bus_and_registry
    fn = MagicMock()
    registry.register("alarm.fired", fn)
    triggers = registry.list_triggers("alarm.fired")
    assert "alarm.fired" in triggers
    assert fn in triggers["alarm.fired"]


def test_register_multiple_triggers_for_same_event(isolated_bus_and_registry):
    """Multiple triggers can be registered for the same event type."""
    _, registry = isolated_bus_and_registry
    fn1, fn2 = MagicMock(), MagicMock()
    registry.register("data.ready", fn1)
    registry.register("data.ready", fn2)
    triggers = registry.list_triggers("data.ready")
    assert fn1 in triggers["data.ready"]
    assert fn2 in triggers["data.ready"]


def test_register_triggers_for_different_events(isolated_bus_and_registry):
    """Triggers for different event types are stored independently."""
    _, registry = isolated_bus_and_registry
    fn_a, fn_b = MagicMock(), MagicMock()
    registry.register("event.a", fn_a)
    registry.register("event.b", fn_b)
    assert fn_a in registry.list_triggers("event.a")["event.a"]
    assert fn_b in registry.list_triggers("event.b")["event.b"]
    assert fn_b not in registry.list_triggers("event.a").get("event.a", [])


def test_list_triggers_all(isolated_bus_and_registry):
    """list_triggers() with no argument returns all registered triggers."""
    _, registry = isolated_bus_and_registry
    fn1, fn2 = MagicMock(), MagicMock()
    registry.register("x", fn1)
    registry.register("y", fn2)
    all_triggers = registry.list_triggers()
    assert "x" in all_triggers
    assert "y" in all_triggers


# ---------------------------------------------------------------------------
# Events trigger workflows / callables
# ---------------------------------------------------------------------------


def test_event_triggers_registered_callable(isolated_bus_and_registry):
    """Publishing an event fires the registered trigger."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    received: list[Event] = []
    registry.register("test.fired", received.append)

    bus.publish("test.fired", {"key": "value"})

    assert len(received) == 1
    assert received[0].event_type == "test.fired"
    assert received[0].payload == {"key": "value"}


def test_event_triggers_multiple_callables(isolated_bus_and_registry):
    """All registered triggers fire when an event is published."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    fn1, fn2 = MagicMock(), MagicMock()
    registry.register("multi.event", fn1)
    registry.register("multi.event", fn2)

    bus.publish("multi.event", {})

    fn1.assert_called_once()
    fn2.assert_called_once()


def test_unregistered_event_type_does_not_fire_trigger(isolated_bus_and_registry):
    """Trigger registered for one event does not fire for a different event."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    fn = MagicMock()
    registry.register("specific.event", fn)

    bus.publish("other.event", {})

    fn.assert_not_called()


def test_trigger_receives_event_object(isolated_bus_and_registry):
    """The trigger callable receives the full Event object."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    captured: list[Any] = []
    registry.register("capture.me", lambda e: captured.append(e))

    bus.publish("capture.me", {"msg": "hello"})

    assert len(captured) == 1
    evt = captured[0]
    assert isinstance(evt, Event)
    assert evt.event_type == "capture.me"
    assert evt.payload["msg"] == "hello"


def test_trigger_not_fired_before_attach(isolated_bus_and_registry):
    """If attach() has not been called, triggers do not fire."""
    bus, registry = isolated_bus_and_registry
    # Do NOT call registry.attach()

    fn = MagicMock()
    registry.register("never.fires", fn)

    bus.publish("never.fires", {})

    fn.assert_not_called()


def test_trigger_not_fired_after_detach(isolated_bus_and_registry):
    """After detach(), triggers stop receiving events."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    fn = MagicMock()
    registry.register("detach.test", fn)

    bus.publish("detach.test", {})
    assert fn.call_count == 1

    registry.detach()
    bus.publish("detach.test", {})
    assert fn.call_count == 1  # should not have increased


def test_attach_is_idempotent(isolated_bus_and_registry):
    """Calling attach() multiple times registers only one handler."""
    bus, registry = isolated_bus_and_registry
    registry.attach()
    registry.attach()  # should be a no-op

    received: list[Event] = []
    registry.register("idempotent.test", received.append)

    bus.publish("idempotent.test", {})

    # If attach registered twice, received would contain 2 entries
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Unregister
# ---------------------------------------------------------------------------


def test_unregister_removes_trigger(isolated_bus_and_registry):
    """unregister() removes the trigger so it no longer fires."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    fn = MagicMock()
    registry.register("remove.me", fn)
    removed = registry.unregister("remove.me", fn)

    assert removed is True
    bus.publish("remove.me", {})
    fn.assert_not_called()


def test_unregister_unknown_trigger_returns_false(isolated_bus_and_registry):
    """unregister() returns False when the trigger was never registered."""
    _, registry = isolated_bus_and_registry
    fn = MagicMock()
    result = registry.unregister("nonexistent", fn)
    assert result is False


# ---------------------------------------------------------------------------
# Error handling / logging
# ---------------------------------------------------------------------------


def test_error_in_trigger_is_logged(isolated_bus_and_registry, caplog):
    """Errors raised by a trigger are caught and logged, not propagated."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    def bad_trigger(event: Event) -> None:
        raise RuntimeError("trigger exploded")

    registry.register("error.event", bad_trigger)

    with caplog.at_level(logging.ERROR, logger="rex.event_triggers"):
        bus.publish("error.event", {})

    assert any("trigger exploded" in r.message or "bad_trigger" in r.message for r in caplog.records)


def test_error_in_one_trigger_does_not_stop_others(isolated_bus_and_registry):
    """When one trigger raises, subsequent triggers still execute."""
    bus, registry = isolated_bus_and_registry
    registry.attach()

    fn_good = MagicMock()

    def bad_trigger(event: Event) -> None:
        raise ValueError("boom")

    registry.register("mixed.event", bad_trigger)
    registry.register("mixed.event", fn_good)

    bus.publish("mixed.event", {})

    fn_good.assert_called_once()


def test_global_registry_returns_same_instance():
    """get_trigger_registry() returns the same instance on repeated calls."""
    r1 = get_trigger_registry()
    r2 = get_trigger_registry()
    assert r1 is r2


def test_set_trigger_registry_replaces_global(isolated_bus_and_registry):
    """set_trigger_registry() swaps out the global registry."""
    bus, _ = isolated_bus_and_registry
    new_registry = EventTriggerRegistry(bus=bus)
    set_trigger_registry(new_registry)
    assert get_trigger_registry() is new_registry
