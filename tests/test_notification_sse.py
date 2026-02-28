"""Tests for the real-time SSE notification stream (Phase 8.1).

Covers:
- SSE broadcaster unit tests (publish, subscribe, shutdown)
- ``/api/notifications/stream`` endpoint auth, content-type, initial payload
- Broadcaster integration: persisting a notification triggers an SSE event
- Graceful disconnect handling

All tests are offline and deterministic.  No network calls are made.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time

import pytest

from rex.dashboard.sse import SSEBroadcaster, set_broadcaster
from rex.dashboard_store import DashboardStore, set_dashboard_store

# Skip all tests if Flask is not available
pytest.importorskip("flask")


# ------------------------------------------------------------------
# Broadcaster unit tests
# ------------------------------------------------------------------


class TestSSEBroadcaster:
    """Unit tests for the SSEBroadcaster class."""

    def test_publish_with_no_subscribers(self):
        """Publish returns 0 when there are no subscribers."""
        b = SSEBroadcaster()
        assert b.publish({"type": "test"}) == 0

    def test_subscribe_and_publish(self):
        """A subscriber receives published events."""
        b = SSEBroadcaster()
        received: list[str] = []

        def reader():
            for event in b.subscribe(timeout=1.0, max_events=1):
                received.append(event)

        t = threading.Thread(target=reader)
        t.start()
        # Give the subscriber thread time to register
        time.sleep(0.1)

        count = b.publish({"type": "notification", "id": "n1"})
        t.join(timeout=3.0)

        assert count == 1
        assert len(received) == 1
        # First event should be a data line (no initial_data set)
        assert received[0].startswith("data: ")
        payload = json.loads(received[0].removeprefix("data: ").strip())
        assert payload["type"] == "notification"
        assert payload["id"] == "n1"

    def test_subscribe_with_initial_data(self):
        """subscribe() yields initial_data as the first event."""
        b = SSEBroadcaster()
        events = list(
            b.subscribe(
                timeout=0.1,
                max_events=0,
                initial_data={"type": "init", "unread_count": 5},
            )
        )
        assert len(events) == 1
        payload = json.loads(events[0].removeprefix("data: ").strip())
        assert payload["type"] == "init"
        assert payload["unread_count"] == 5

    def test_max_events_limits_output(self):
        """subscribe() stops after max_events data events."""
        b = SSEBroadcaster()
        received: list[str] = []

        def reader():
            for event in b.subscribe(timeout=1.0, max_events=2):
                received.append(event)

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.1)

        for i in range(5):
            b.publish({"i": i})
            time.sleep(0.05)

        t.join(timeout=3.0)
        # Should have received exactly 2 data events
        data_events = [e for e in received if e.startswith("data: ")]
        assert len(data_events) == 2

    def test_shutdown_disconnects_subscribers(self):
        """shutdown() causes all subscriber generators to exit."""
        b = SSEBroadcaster()
        received: list[str] = []

        def reader():
            for event in b.subscribe(timeout=5.0, max_events=None):
                received.append(event)

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.1)

        b.shutdown()
        t.join(timeout=3.0)
        assert not t.is_alive()

    def test_subscriber_count(self):
        """subscriber_count reflects active subscribers."""
        b = SSEBroadcaster()
        assert b.subscriber_count == 0

        events = []

        def reader():
            for event in b.subscribe(timeout=5.0, max_events=1):
                events.append(event)

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.1)
        assert b.subscriber_count == 1

        b.publish({"done": True})
        t.join(timeout=3.0)
        assert b.subscriber_count == 0

    def test_multiple_subscribers(self):
        """Multiple subscribers each receive published events."""
        b = SSEBroadcaster()
        results = {1: [], 2: []}

        def reader(idx):
            for event in b.subscribe(timeout=1.0, max_events=1):
                results[idx].append(event)

        t1 = threading.Thread(target=reader, args=(1,))
        t2 = threading.Thread(target=reader, args=(2,))
        t1.start()
        t2.start()
        time.sleep(0.1)

        b.publish({"msg": "hello"})
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        assert len(results[1]) == 1
        assert len(results[2]) == 1


# ------------------------------------------------------------------
# Flask endpoint tests
# ------------------------------------------------------------------


@pytest.fixture
def sse_app_client(monkeypatch, tmp_path):
    """Create a test Flask client with SSE support."""
    monkeypatch.setenv("REX_TESTING", "true")
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "1")

    scheduler_path = tmp_path / "scheduler" / "jobs.json"
    scheduler_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "rex_config.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {"llm_provider": "echo", "llm_model": "test"},
                "runtime": {"log_level": "DEBUG"},
            }
        )
    )

    from rex import scheduler as scheduler_module

    scheduler_module._SCHEDULER = None

    from rex.scheduler import Scheduler, set_scheduler

    test_scheduler = Scheduler(jobs_file=scheduler_path)
    set_scheduler(test_scheduler)

    if "flask_proxy" in sys.modules:
        del sys.modules["flask_proxy"]

    module = importlib.import_module("flask_proxy")
    app = module.app
    app.config["TESTING"] = True

    # Use a fresh broadcaster for each test
    test_broadcaster = SSEBroadcaster()
    set_broadcaster(test_broadcaster)

    with app.test_client() as client:
        yield client, tmp_path, test_broadcaster

    set_broadcaster(None)


@pytest.fixture
def sse_auth_headers(sse_app_client):
    """Get auth headers and token by logging in."""
    client, _, _ = sse_app_client
    response = client.post(
        "/api/dashboard/login",
        json={},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )
    if response.status_code == 200:
        data = response.get_json()
        return {"Authorization": f"Bearer {data['token']}"}
    return {}


@pytest.fixture
def sse_auth_token(sse_app_client):
    """Get just the auth token (for query param auth)."""
    client, _, _ = sse_app_client
    response = client.post(
        "/api/dashboard/login",
        json={},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )
    if response.status_code == 200:
        return response.get_json()["token"]
    return ""


class TestSSEEndpointAuth:
    """Test that /api/notifications/stream requires authentication."""

    def test_stream_requires_auth(self, sse_app_client):
        """Stream endpoint returns 401 when unauthenticated."""
        client, _, _ = sse_app_client
        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"
        try:
            response = client.get(
                "/api/notifications/stream?max_events=0&_testing=1",
                environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            )
            assert response.status_code == 401
        finally:
            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"

    def test_stream_accessible_with_auth(self, sse_app_client, sse_auth_headers):
        """Stream endpoint returns 200 with valid auth."""
        client, _, _ = sse_app_client
        response = client.get(
            "/api/notifications/stream?max_events=0&_testing=1",
            headers=sse_auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200

    def test_stream_accessible_with_token_query_param(self, sse_app_client, sse_auth_token):
        """Stream endpoint accepts token as query parameter (for EventSource)."""
        client, _, _ = sse_app_client
        response = client.get(
            f"/api/notifications/stream?max_events=0&_testing=1&token={sse_auth_token}",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200


class TestSSEEndpointResponse:
    """Test SSE stream response format and initial data."""

    def test_content_type_is_event_stream(self, sse_app_client, sse_auth_headers):
        """Response Content-Type is text/event-stream."""
        client, _, _ = sse_app_client
        response = client.get(
            "/api/notifications/stream?max_events=0&_testing=1",
            headers=sse_auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.content_type

    def test_initial_payload_contains_unread_count(
        self, sse_app_client, sse_auth_headers, tmp_path
    ):
        """The initial SSE event includes the current unread count."""
        client, _, broadcaster = sse_app_client
        store = DashboardStore(db_path=tmp_path / "sse_init.db")
        store.write(notification_id="sse_n1", title="Test", body="body")
        set_dashboard_store(store)

        try:
            response = client.get(
                "/api/notifications/stream?max_events=0&_testing=1",
                headers=sse_auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert response.status_code == 200
            # The streamed body should contain the init event
            body = response.get_data(as_text=True)
            assert "data: " in body
            # Parse the first data line
            for line in body.strip().split("\n"):
                if line.startswith("data: "):
                    payload = json.loads(line.removeprefix("data: "))
                    assert payload["type"] == "init"
                    assert payload["unread_count"] >= 1
                    break
            else:
                pytest.fail("No data event found in SSE stream")
        finally:
            set_dashboard_store(None)

    def test_no_cache_headers(self, sse_app_client, sse_auth_headers):
        """SSE responses include no-cache headers."""
        client, _, _ = sse_app_client
        response = client.get(
            "/api/notifications/stream?max_events=0&_testing=1",
            headers=sse_auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.headers.get("Cache-Control") == "no-cache"


class TestSSEBroadcastIntegration:
    """Test that persisting a notification triggers an SSE event."""

    def test_store_write_publishes_to_broadcaster(self, tmp_path):
        """DashboardStore.write() publishes an event to the SSE broadcaster."""
        broadcaster = SSEBroadcaster()
        set_broadcaster(broadcaster)

        received: list[str] = []

        def reader():
            for event in broadcaster.subscribe(timeout=2.0, max_events=1):
                received.append(event)

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.1)

        store = DashboardStore(db_path=tmp_path / "broadcast_test.db")
        store.write(
            notification_id="bc_1",
            priority="urgent",
            title="Broadcast Test",
            body="test body",
        )

        t.join(timeout=5.0)
        set_broadcaster(None)

        assert len(received) == 1
        payload = json.loads(received[0].removeprefix("data: ").strip())
        assert payload["type"] == "notification"
        assert payload["id"] == "bc_1"
        assert payload["priority"] == "urgent"
        assert payload["title"] == "Broadcast Test"

    def test_store_write_without_broadcaster_does_not_fail(self, tmp_path):
        """DashboardStore.write() succeeds even if broadcaster is absent."""
        set_broadcaster(None)
        store = DashboardStore(db_path=tmp_path / "no_broadcaster.db")
        nid = store.write(title="No Crash", body="test")
        assert nid is not None
