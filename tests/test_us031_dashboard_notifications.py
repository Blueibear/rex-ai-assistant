"""Tests for US-031: Dashboard notifications streamed via SSE.

Acceptance criteria:
- SSE endpoint works
- notifications streamed
- disconnect handled
- Typecheck passes
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from rex.dashboard.sse import (
    NotificationBroadcaster,
    NotificationEvent,
    get_broadcaster,
    set_broadcaster,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_broadcaster():
    """Isolate the global broadcaster singleton for each test."""
    set_broadcaster(None)
    yield
    set_broadcaster(None)


def _make_broadcaster() -> NotificationBroadcaster:
    return NotificationBroadcaster()


# ---------------------------------------------------------------------------
# NotificationEvent
# ---------------------------------------------------------------------------


def test_notification_event_to_payload():
    event = NotificationEvent(
        type="notification",
        notification_id="notif_abc",
        user_id="alice",
        unread_count=3,
    )
    payload = event.to_payload()
    assert payload["type"] == "notification"
    assert payload["notification_id"] == "notif_abc"
    assert payload["user_id"] == "alice"
    assert payload["unread_count"] == 3


def test_notification_event_optional_unread():
    event = NotificationEvent(
        type="notification",
        notification_id="x",
        user_id=None,
    )
    payload = event.to_payload()
    assert payload["unread_count"] is None
    assert payload["user_id"] is None


# ---------------------------------------------------------------------------
# NotificationBroadcaster – subscribe / unsubscribe
# ---------------------------------------------------------------------------


def test_subscribe_creates_subscriber():
    bc = _make_broadcaster()
    sub = bc.subscribe()
    assert not sub.closed
    assert bc.subscriber_count == 1


def test_unsubscribe_closes_subscriber():
    bc = _make_broadcaster()
    sub = bc.subscribe()
    bc.unsubscribe(sub)
    assert sub.closed
    assert bc.subscriber_count == 0


def test_multiple_subscribers():
    bc = _make_broadcaster()
    s1 = bc.subscribe()
    s2 = bc.subscribe()
    assert bc.subscriber_count == 2
    bc.unsubscribe(s1)
    assert bc.subscriber_count == 1
    bc.unsubscribe(s2)
    assert bc.subscriber_count == 0


# ---------------------------------------------------------------------------
# NotificationBroadcaster – publish
# ---------------------------------------------------------------------------


def test_publish_event_object_reaches_subscriber():
    bc = _make_broadcaster()
    sub = bc.subscribe()
    event = NotificationEvent(
        type="notification",
        notification_id="n1",
        user_id="bob",
        unread_count=1,
    )
    bc.publish(event)
    payload: dict[str, Any] = sub.queue.get(timeout=1)
    assert payload["notification_id"] == "n1"
    assert payload["user_id"] == "bob"


def test_publish_dict_reaches_subscriber():
    bc = _make_broadcaster()
    sub = bc.subscribe()
    bc.publish({"type": "test", "data": "hello"})
    payload = sub.queue.get(timeout=1)
    assert payload["data"] == "hello"


def test_publish_to_multiple_subscribers():
    bc = _make_broadcaster()
    s1 = bc.subscribe()
    s2 = bc.subscribe()
    bc.publish({"msg": "broadcast"})
    p1 = s1.queue.get(timeout=1)
    p2 = s2.queue.get(timeout=1)
    assert p1["msg"] == "broadcast"
    assert p2["msg"] == "broadcast"


def test_publish_invalid_type_raises():
    bc = _make_broadcaster()
    with pytest.raises(TypeError):
        bc.publish("not a dict or event")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# NotificationBroadcaster – stream (SSE endpoint works / notifications streamed)
# ---------------------------------------------------------------------------


def test_stream_yields_init_like_event_on_publish():
    """stream() yields an SSE chunk when an event is published."""
    bc = _make_broadcaster()
    sub = bc.subscribe()
    bc.publish({"type": "notification", "notification_id": "n2", "user_id": "u1"})
    # Close subscriber after one item so stream terminates
    chunks = []
    for chunk in bc.stream(sub, timeout=0.5, keepalive_interval=999.0):
        chunks.append(chunk)
        sub.closed = True  # Signal end after first event
        break
    assert chunks, "Expected at least one SSE chunk"
    first = chunks[0]
    assert "event: notification" in first
    assert "data:" in first
    payload = json.loads(first.split("data: ", 1)[1].strip())
    assert payload["notification_id"] == "n2"


def test_stream_yields_keepalive_on_timeout():
    """stream() emits a keep-alive comment when no events arrive within keepalive_interval."""
    bc = _make_broadcaster()
    sub = bc.subscribe()
    chunks = []
    for chunk in bc.stream(sub, timeout=0.05, keepalive_interval=0.0):
        chunks.append(chunk)
        sub.closed = True
        break
    assert any(": keep-alive" in c for c in chunks)


def test_stream_multiple_events_in_order():
    bc = _make_broadcaster()
    sub = bc.subscribe()
    for i in range(3):
        bc.publish({"seq": i})

    received = []
    for chunk in bc.stream(sub, timeout=0.2, keepalive_interval=999.0):
        if "data:" in chunk:
            payload = json.loads(chunk.split("data: ", 1)[1].strip())
            received.append(payload["seq"])
        if len(received) == 3:
            sub.closed = True
            break
    assert received == [0, 1, 2]


# ---------------------------------------------------------------------------
# Disconnect handling
# ---------------------------------------------------------------------------


def test_stream_stops_when_subscriber_closed():
    """stream() exits cleanly when subscriber.closed is set externally."""
    bc = _make_broadcaster()
    sub = bc.subscribe()

    results: list[str] = []

    def _run():
        for chunk in bc.stream(sub, timeout=0.1, keepalive_interval=999.0):
            results.append(chunk)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.05)
    bc.unsubscribe(sub)  # Marks closed=True
    t.join(timeout=1.0)
    assert not t.is_alive(), "stream() should have terminated after unsubscribe"


def test_shutdown_closes_all_subscribers():
    bc = _make_broadcaster()
    subs = [bc.subscribe() for _ in range(3)]
    assert bc.subscriber_count == 3
    bc.shutdown()
    for sub in subs:
        assert sub.closed
    assert bc.subscriber_count == 0


def test_stream_unsubscribes_subscriber_on_exit():
    """stream() removes the subscriber from the broadcaster when the generator is exhausted."""
    bc = _make_broadcaster()
    sub = bc.subscribe()
    # Publish one event then mark closed so stream exits
    bc.publish({"type": "x"})
    sub.closed = True
    # Drain the generator
    list(bc.stream(sub, timeout=0.05, keepalive_interval=999.0))
    assert bc.subscriber_count == 0


# ---------------------------------------------------------------------------
# get_broadcaster singleton
# ---------------------------------------------------------------------------


def test_get_broadcaster_returns_singleton():
    b1 = get_broadcaster()
    b2 = get_broadcaster()
    assert b1 is b2


def test_set_broadcaster_replaces_singleton():
    custom = _make_broadcaster()
    set_broadcaster(custom)
    assert get_broadcaster() is custom


# ---------------------------------------------------------------------------
# Queue overflow – oldest event dropped, stream continues
# ---------------------------------------------------------------------------


def test_queue_full_drops_oldest_event():
    bc = _make_broadcaster()
    sub = bc.subscribe(max_events=2)
    # Fill the queue and then overflow
    bc.publish({"seq": 0})
    bc.publish({"seq": 1})
    bc.publish({"seq": 2})  # Should drop seq=0 and add seq=2
    received = []
    for chunk in bc.stream(sub, timeout=0.2, keepalive_interval=999.0):
        if "data:" in chunk:
            payload = json.loads(chunk.split("data: ", 1)[1].strip())
            received.append(payload["seq"])
        if len(received) == 2:
            sub.closed = True
            break
    assert 0 not in received, "Oldest event should have been dropped"
    assert 2 in received
