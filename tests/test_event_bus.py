"""Tests for the event bus."""

from __future__ import annotations

from rex.event_bus import EventBus


def test_event_bus_publish_and_subscribe():
    bus = EventBus()
    received = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        received.append((event_type, payload))

    bus.subscribe("email.triaged", handler)
    bus.publish("email.triaged", {"count": 2})

    assert received == [("email.triaged", {"count": 2})]


def test_event_bus_wildcard():
    bus = EventBus()
    received = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        received.append((event_type, payload))

    bus.subscribe("*", handler)
    bus.publish("calendar.created", {"event_id": "123"})

    assert received == [("calendar.created", {"event_id": "123"})]
