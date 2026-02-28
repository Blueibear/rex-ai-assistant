"""Tests for dashboard notification SSE streaming and broadcaster behavior."""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from rex.dashboard.auth import SessionManager
from rex.dashboard.sse import NotificationBroadcaster, NotificationEvent, set_broadcaster
from rex.dashboard_store import DashboardStore, set_dashboard_store

pytest.importorskip("flask")


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    monkeypatch.setenv("REX_TESTING", "true")
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-password")

    from rex import scheduler as scheduler_module

    scheduler_module._SCHEDULER = None

    from rex.scheduler import Scheduler, set_scheduler

    jobs_path = tmp_path / "scheduler" / "jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    set_scheduler(Scheduler(jobs_file=jobs_path))

    store = DashboardStore(db_path=tmp_path / "dashboard_notifications.db")
    set_dashboard_store(store)

    set_broadcaster(NotificationBroadcaster())

    from rex.dashboard import auth as dashboard_auth

    dashboard_auth._session_manager = SessionManager()

    if "flask_proxy" in sys.modules:
        del sys.modules["flask_proxy"]

    module = importlib.import_module("flask_proxy")
    app = module.app

    with app.test_client() as client:
        yield client, store, app

    set_dashboard_store(None)
    set_broadcaster(None)


@pytest.fixture
def auth_header(app_client):
    client, _, _ = app_client
    response = client.post(
        "/api/dashboard/login",
        json={"password": "test-password"},
        environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
    )
    assert response.status_code == 200
    token = response.get_json()["token"]
    return {"Authorization": f"Bearer {token}"}, token


def _next_chunk(response):
    return next(response.response).decode("utf-8")


class TestNotificationBroadcaster:
    def test_publish_subscribe_and_keepalive(self):
        broadcaster = NotificationBroadcaster()
        subscriber = broadcaster.subscribe(max_events=2)

        broadcaster.publish(
            NotificationEvent(
                type="created",
                notification_id="n1",
                user_id="james",
                unread_count=1,
            )
        )

        stream = broadcaster.stream(subscriber, timeout=0.01, keepalive_interval=0.0)
        first = next(stream)
        assert "event: notification" in first
        assert '"notification_id": "n1"' in first

        keepalive = next(stream)
        assert keepalive.startswith(": keep-alive")

    def test_drops_oldest_when_queue_is_full(self):
        broadcaster = NotificationBroadcaster()
        subscriber = broadcaster.subscribe(max_events=1)

        broadcaster.publish(
            NotificationEvent(type="created", notification_id="n1", user_id="james", unread_count=1)
        )
        broadcaster.publish(
            NotificationEvent(type="created", notification_id="n2", user_id="james", unread_count=2)
        )

        stream = broadcaster.stream(subscriber, timeout=0.01, keepalive_interval=1.0)
        first = next(stream)
        assert '"notification_id": "n2"' in first


class TestNotificationSSEEndpoint:
    def test_stream_requires_auth(self, app_client):
        client, _, _ = app_client

        response = client.get(
            "/api/notifications/stream",
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
        )

        assert response.status_code == 401

    def test_stream_sends_init_and_headers(self, app_client, auth_header):
        client, store, _ = app_client
        headers, _token = auth_header

        store.write(title="Welcome", body="Message", user_id="james")

        response = client.get(
            "/api/notifications/stream?timeout=0.01",
            headers=headers,
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            buffered=False,
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["X-Accel-Buffering"] == "no"

        try:
            first = _next_chunk(response)
            assert first.startswith("event: init")
            assert "unread_count" in first
        finally:
            response.close()

    def test_stream_supports_query_token_only_in_safe_contexts(self, app_client, auth_header):
        client, _store, app = app_client
        _headers, token = auth_header

        insecure_client = app.test_client(use_cookies=False)
        insecure = insecure_client.get(
            f"/api/notifications/stream?token={token}",
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            buffered=False,
        )
        assert insecure.status_code == 401
        insecure.close()

        secure = client.get(
            f"/api/notifications/stream?token={token}",
            headers={"X-Forwarded-Proto": "https", "Origin": "http://localhost"},
            environ_overrides={"REMOTE_ADDR": "8.8.8.8", "HTTP_HOST": "localhost"},
            buffered=False,
        )
        assert secure.status_code == 200
        secure.close()

    def test_store_write_triggers_user_scoped_sse_event(self, app_client, auth_header):
        client, store, _ = app_client
        headers, _token = auth_header

        response = client.get(
            "/api/notifications/stream?timeout=0.01",
            headers=headers,
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            buffered=False,
        )
        assert response.status_code == 200

        try:
            first = _next_chunk(response)
            assert first.startswith("event: init")

            store.write(title="Scoped", body="Only for james", user_id="james")
            event_chunk = _next_chunk(response)
            assert event_chunk.startswith("event: notification")
            payload_line = next(
                line for line in event_chunk.splitlines() if line.startswith("data: ")
            )
            payload = json.loads(payload_line[6:])
            assert payload["user_id"] == "james"
        finally:
            response.close()

    def test_stream_filters_other_user_events(self, app_client, auth_header):
        client, store, _ = app_client
        headers, _token = auth_header

        response = client.get(
            "/api/notifications/stream?timeout=0.01",
            headers=headers,
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            buffered=False,
        )
        assert response.status_code == 200
        try:
            _next_chunk(response)  # init

            store.write(title="Other", body="for alex", user_id="alex")
            # Next emitted chunk should be keepalive, not a cross-user notification.
            chunk = _next_chunk(response)
            assert chunk.startswith(": keep-alive")
        finally:
            response.close()


def test_notification_api_user_scope_enforced(app_client, auth_header):
    client, store, _ = app_client
    headers, _token = auth_header

    store.write(title="Mine", body="ok", user_id="james")
    store.write(title="Other", body="no", user_id="alex")

    response = client.get(
        "/api/notifications",
        headers=headers,
        environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
    )
    assert response.status_code == 200
    titles = [n["title"] for n in response.get_json()["notifications"]]
    assert titles == ["Mine"]

    forbidden = client.get(
        "/api/notifications?user_id=alex",
        headers=headers,
        environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
    )
    assert forbidden.status_code == 403
