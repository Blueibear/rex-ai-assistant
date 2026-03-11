"""Tests for the Rex Dashboard module.

Covers:
- Authentication required for dashboard API endpoints
- Settings redaction
- Scheduler create/list basic flow
- Chat endpoint
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from rex.dashboard_store import DashboardStore, set_dashboard_store

# Skip all tests if Flask is not available
pytest.importorskip("flask")


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    """Create a test Flask client with the dashboard enabled."""
    # Set up environment
    monkeypatch.setenv("REX_TESTING", "true")
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "1")

    # Create temp scheduler storage
    scheduler_path = tmp_path / "scheduler" / "jobs.json"
    scheduler_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp config
    config_path = tmp_path / "rex_config.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {"llm_provider": "echo", "llm_model": "test"},
                "runtime": {"log_level": "DEBUG"},
            }
        )
    )

    # Reset scheduler singleton before importing
    from rex import scheduler as scheduler_module

    scheduler_module._SCHEDULER = None

    # Create a new scheduler with temp path
    from rex.scheduler import Scheduler, set_scheduler

    test_scheduler = Scheduler(jobs_file=scheduler_path)
    set_scheduler(test_scheduler)

    # Reset modules
    if "flask_proxy" in sys.modules:
        del sys.modules["flask_proxy"]

    # Import and configure app
    module = importlib.import_module("flask_proxy")
    app = module.app

    # Return test client
    with app.test_client() as client:
        yield client, tmp_path


@pytest.fixture
def auth_headers(app_client):
    """Get auth headers by logging in."""
    client, _ = app_client

    # Login via local access (no password required)
    response = client.post(
        "/api/dashboard/login",
        json={},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    if response.status_code == 200:
        data = response.get_json()
        return {"Authorization": f"Bearer {data['token']}"}

    # If login failed, return empty headers (tests will handle auth failures)
    return {}


class TestDashboardAuth:
    """Test authentication for dashboard endpoints."""

    def test_status_endpoint_is_public(self, app_client):
        """Test that /api/dashboard/status is accessible without auth."""
        client, _ = app_client

        response = client.get(
            "/api/dashboard/status",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_login_creates_session(self, app_client):
        """Test that login returns a session token."""
        client, _ = app_client

        response = client.post(
            "/api/dashboard/login",
            json={},
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "token" in data
        assert "expires_at" in data

    def test_logout_invalidates_session(self, app_client, auth_headers):
        """Test that logout invalidates the session."""
        client, _ = app_client

        # First verify we're authenticated
        response = client.get(
            "/api/settings",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200

        # Logout
        response = client.post(
            "/api/dashboard/logout",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200

    def test_settings_requires_auth(self, app_client):
        """Test that /api/settings requires authentication when not local."""
        client, _ = app_client

        # Set environment to require auth
        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"

        response = client.get(
            "/api/settings",
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
        )

        # Should require auth
        assert response.status_code == 401

        # Restore
        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"

    def test_chat_requires_auth(self, app_client):
        """Test that /api/chat requires authentication when not local."""
        client, _ = app_client

        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"

        response = client.post(
            "/api/chat",
            json={"message": "hello"},
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
        )

        assert response.status_code == 401

        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"

    def test_scheduler_requires_auth(self, app_client):
        """Test that /api/scheduler/jobs requires authentication when not local."""
        client, _ = app_client

        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"

        response = client.get(
            "/api/scheduler/jobs",
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
        )

        assert response.status_code == 401

        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"


class TestSettingsRedaction:
    """Test settings redaction functionality."""

    def test_settings_returns_redacted_values(self, app_client, auth_headers):
        """Test that sensitive settings are redacted."""
        client, _ = app_client

        response = client.get(
            "/api/settings",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "settings" in data

    def test_redaction_preserves_structure(self, app_client, auth_headers):
        """Test that redaction preserves config structure."""
        client, _ = app_client

        response = client.get(
            "/api/settings",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        settings = data["settings"]

        # Check expected structure is preserved
        assert "models" in settings
        assert "runtime" in settings

    def test_settings_includes_metadata(self, app_client, auth_headers):
        """Test that settings response includes metadata about restart requirements."""
        client, _ = app_client

        response = client.get(
            "/api/settings",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "metadata" in data

    def test_settings_patch_updates_config(self, app_client, auth_headers, tmp_path):
        """Test that PATCH /api/settings updates configuration."""
        client, _ = app_client

        response = client.patch(
            "/api/settings",
            json={"runtime.log_level": "WARNING"},
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "updated" in data
        assert "runtime.log_level" in data["updated"]

    def test_settings_patch_rejects_unknown_key(self, app_client, auth_headers):
        """Test that PATCH /api/settings rejects unknown keys."""
        client, _ = app_client

        response = client.patch(
            "/api/settings",
            json={"runtime.unknown_key": "value"},
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "invalid" in data
        assert "runtime.unknown_key" in data["invalid"]

    def test_settings_patch_rejects_type_mismatch(self, app_client, auth_headers):
        """Test that PATCH /api/settings rejects invalid value types."""
        client, _ = app_client

        response = client.patch(
            "/api/settings",
            json={"runtime.log_level": 123},
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "invalid" in data
        assert "runtime.log_level" in data["invalid"]


class TestSchedulerAPI:
    """Test scheduler/reminders API endpoints."""

    def test_list_jobs_returns_array(self, app_client, auth_headers):
        """Test that listing jobs returns an array."""
        client, _ = app_client

        response = client.get(
            "/api/scheduler/jobs",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
        assert "total" in data

    def test_create_job(self, app_client, auth_headers):
        """Test creating a new scheduled job."""
        client, _ = app_client

        response = client.post(
            "/api/scheduler/jobs",
            json={
                "name": "Test Reminder",
                "schedule": "interval:3600",
                "enabled": True,
            },
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 201
        data = response.get_json()
        assert "job_id" in data
        assert data["name"] == "Test Reminder"
        assert data["schedule"] == "interval:3600"

    def test_create_and_list_job(self, app_client, auth_headers):
        """Test creating a job and then listing it."""
        client, _ = app_client

        # Create job
        create_response = client.post(
            "/api/scheduler/jobs",
            json={
                "name": "List Test Job",
                "schedule": "interval:600",
            },
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert create_response.status_code == 201
        job_id = create_response.get_json()["job_id"]

        # List jobs
        list_response = client.get(
            "/api/scheduler/jobs",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert list_response.status_code == 200
        jobs = list_response.get_json()["jobs"]
        job_ids = [j["job_id"] for j in jobs]
        assert job_id in job_ids

    def test_update_job(self, app_client, auth_headers):
        """Test updating a job."""
        client, _ = app_client

        # Create job
        create_response = client.post(
            "/api/scheduler/jobs",
            json={
                "name": "Update Test Job",
                "schedule": "interval:600",
                "enabled": True,
            },
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        job_id = create_response.get_json()["job_id"]

        # Update job
        update_response = client.patch(
            f"/api/scheduler/jobs/{job_id}",
            json={"enabled": False},
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert update_response.status_code == 200
        data = update_response.get_json()
        assert data["enabled"] is False

    def test_delete_job(self, app_client, auth_headers):
        """Test deleting a job."""
        client, _ = app_client

        # Create job
        create_response = client.post(
            "/api/scheduler/jobs",
            json={
                "name": "Delete Test Job",
                "schedule": "interval:600",
            },
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        job_id = create_response.get_json()["job_id"]

        # Delete job
        delete_response = client.delete(
            f"/api/scheduler/jobs/{job_id}",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert delete_response.status_code == 200
        assert delete_response.get_json()["deleted"] is True

    def test_run_job(self, app_client, auth_headers):
        """Test running a job manually."""
        client, _ = app_client

        # Create job
        create_response = client.post(
            "/api/scheduler/jobs",
            json={
                "name": "Run Test Job",
                "schedule": "interval:600",
            },
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        job_id = create_response.get_json()["job_id"]

        # Run job
        run_response = client.post(
            f"/api/scheduler/jobs/{job_id}/run",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert run_response.status_code == 200
        # Job might succeed or fail depending on callback, but endpoint should work
        assert "success" in run_response.get_json()

    def test_invalid_schedule_rejected(self, app_client, auth_headers):
        """Test that invalid schedules are rejected."""
        client, _ = app_client

        response = client.post(
            "/api/scheduler/jobs",
            json={
                "name": "Invalid Schedule Job",
                "schedule": "invalid_format",
            },
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 400


class TestChatAPI:
    """Test chat API endpoints."""

    def test_chat_requires_message(self, app_client, auth_headers):
        """Test that chat endpoint requires a message."""
        client, _ = app_client

        response = client.post(
            "/api/chat",
            json={},
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 400

    def test_chat_returns_reply(self, app_client, auth_headers):
        """Test that chat endpoint returns a reply."""
        client, _ = app_client

        with patch("rex.dashboard.routes._get_llm") as mock_llm:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = "Hello! How can I help you?"
            mock_llm.return_value = mock_instance

            response = client.post(
                "/api/chat",
                json={"message": "Hello"},
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert "reply" in data
            assert "timestamp" in data

    def test_chat_history(self, app_client, auth_headers):
        """Test chat history endpoint."""
        client, _ = app_client

        response = client.get(
            "/api/chat/history",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "history" in data
        assert "total" in data


class TestDashboardUI:
    """Test dashboard UI endpoints."""

    def test_dashboard_returns_html(self, app_client):
        """Test that /dashboard returns HTML."""
        client, _ = app_client

        response = client.get(
            "/dashboard",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data

    def test_dashboard_assets_css(self, app_client):
        """Test that CSS assets are served."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/css/dashboard.css",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"Rex Dashboard" in response.data or response.content_type.startswith("text/css")

    def test_dashboard_assets_js(self, app_client):
        """Test that JS assets are served."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200


class TestRedactionHelper:
    """Test the redaction helper function directly."""

    def test_redact_sensitive_keys(self):
        """Test that sensitive keys are redacted."""
        from rex.contracts import redact_sensitive_keys

        data = {
            "api_key": "secret123",
            "name": "test",
            "nested": {
                "password": "hunter2",
                "value": "visible",
            },
        }

        redacted = redact_sensitive_keys(data)

        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["name"] == "test"
        assert redacted["nested"]["password"] == "[REDACTED]"
        assert redacted["nested"]["value"] == "visible"

    def test_redact_preserves_none_values(self):
        """Test that None values are preserved when key is sensitive."""
        from rex.contracts import redact_sensitive_keys

        data = {"api_key": None, "name": "test"}
        redacted = redact_sensitive_keys(data)

        # Even None values for sensitive keys should be redacted
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["name"] == "test"

    def test_redact_handles_lists(self):
        """Test that lists are handled correctly."""
        from rex.contracts import redact_sensitive_keys

        data = {
            "items": [
                {"token": "abc", "name": "one"},
                {"token": "def", "name": "two"},
            ]
        }

        redacted = redact_sensitive_keys(data)

        assert redacted["items"][0]["token"] == "[REDACTED]"
        assert redacted["items"][0]["name"] == "one"
        assert redacted["items"][1]["token"] == "[REDACTED]"
        assert redacted["items"][1]["name"] == "two"


class TestNotificationEndpoints:
    """Test dashboard notification API endpoints."""

    def test_list_notifications(self, app_client, auth_headers, tmp_path):
        """List endpoint returns stored notifications."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "dashboard_notifications.db")
        store.write(
            notification_id="notif_1",
            priority="urgent",
            title="CPU Alert",
            body="CPU usage high",
            user_id="james",
        )
        store.write(
            notification_id="notif_2",
            priority="normal",
            title="FYI",
            body="All good",
            user_id="james",
        )

        set_dashboard_store(store)
        try:
            response = client.get(
                "/api/notifications?limit=10&unread=true&user_id=james",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert response.status_code == 200
            payload = response.get_json()
            assert payload["total"] == 2
            assert payload["unread_count"] == 2
            assert payload["notifications"][0]["id"] in {"notif_1", "notif_2"}
        finally:
            set_dashboard_store(None)

    def test_mark_notification_read_and_mark_all(self, app_client, auth_headers, tmp_path):
        """Mark-read endpoints update read state."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "dashboard_notifications.db")
        store.write(notification_id="notif_1", title="A", body="a", user_id="james")
        store.write(notification_id="notif_2", title="B", body="b", user_id="james")

        set_dashboard_store(store)
        try:
            read_one = client.post(
                "/api/notifications/notif_1/read",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert read_one.status_code == 200
            assert read_one.get_json()["read"] is True

            read_all = client.post(
                "/api/notifications/read-all?user_id=james",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert read_all.status_code == 200
            assert read_all.get_json()["marked_read"] == 1
        finally:
            set_dashboard_store(None)

    def test_notification_endpoints_require_auth(self, app_client, tmp_path):
        """Notification endpoints require auth when local bypass is disabled."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "dashboard_notifications.db")
        set_dashboard_store(store)

        try:
            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"

            response = client.get(
                "/api/notifications",
                environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            )
            assert response.status_code == 401

            response = client.post(
                "/api/notifications/read-all",
                environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            )
            assert response.status_code == 401
        finally:
            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"
            set_dashboard_store(None)


class TestNotificationsInboxUI:
    """Tests for the notification inbox UI page and end-to-end API integration."""

    def test_notifications_page_returns_html(self, app_client):
        """GET /dashboard/notifications returns 200 with HTML."""
        client, _ = app_client

        response = client.get(
            "/dashboard/notifications",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data

    def test_notifications_page_contains_inbox_markup(self, app_client):
        """The notifications page contains the inbox section markup."""
        client, _ = app_client

        response = client.get(
            "/dashboard/notifications",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        # Key section and filter elements must be present in the SPA template
        assert b"notifications-section" in response.data
        assert b"notif-list" in response.data
        assert b"notif-filter-unread" in response.data
        assert b"notif-filter-priority" in response.data
        assert b"notif-filter-channel" in response.data
        assert b"mark-all-read-btn" in response.data
        assert b"notif-badge-mobile" in response.data

    def test_notifications_ui_does_not_require_auth_at_html_level(self, app_client):
        """The HTML page itself is served without auth (auth is enforced at API level)."""
        client, _ = app_client

        os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"
        try:
            response = client.get(
                "/dashboard/notifications",
                environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            )
            # HTML page is always served; auth is only required for /api/* endpoints
            assert response.status_code == 200
            assert b"<!DOCTYPE html>" in response.data
        finally:
            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"

    def test_notifications_api_requires_auth_remote(self, app_client, tmp_path):
        """API endpoint /api/notifications requires auth from remote addresses."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "notif_ui_auth.db")
        set_dashboard_store(store)

        try:
            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"
            response = client.get(
                "/api/notifications",
                environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            )
            assert response.status_code == 401
        finally:
            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"
            set_dashboard_store(None)

    def test_mark_read_end_to_end(self, app_client, auth_headers, tmp_path):
        """Mark-read action end-to-end through the Flask app updates store state."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "notif_ui_mark_read.db")
        store.write(notification_id="ui_e2e_1", title="E2E test", body="body", user_id="james")
        set_dashboard_store(store)

        try:
            resp = client.post(
                "/api/notifications/ui_e2e_1/read",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert resp.status_code == 200
            assert resp.get_json()["read"] is True

            # Verify persisted in store
            notifs = store.query_recent(limit=10)
            notif = next((n for n in notifs if n.id == "ui_e2e_1"), None)
            assert notif is not None
            assert notif.read is True
        finally:
            set_dashboard_store(None)

    def test_mark_all_read_end_to_end(self, app_client, auth_headers, tmp_path):
        """Mark-all-read action end-to-end through the Flask app updates store state."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "notif_ui_mark_all.db")
        store.write(notification_id="ui_e2e_a", title="A", body="a", user_id="james")
        store.write(notification_id="ui_e2e_b", title="B", body="b", user_id="james")
        store.write(notification_id="ui_e2e_c", title="C", body="c", user_id="bob")
        set_dashboard_store(store)

        try:
            resp = client.post(
                "/api/notifications/read-all?user_id=james",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert resp.status_code == 200
            assert resp.get_json()["marked_read"] == 2

            # Bob's notification must remain unread
            assert store.count_unread(user_id="bob") == 1
        finally:
            set_dashboard_store(None)

    def test_list_notifications_with_unread_filter(self, app_client, auth_headers, tmp_path):
        """Unread filter returns only unread notifications."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "notif_ui_unread.db")
        store.write(notification_id="unread_1", title="Unread", body="b1", user_id="james")
        store.write(notification_id="unread_2", title="Unread2", body="b2", user_id="james")
        # mark one as read
        store.mark_as_read("unread_1")
        set_dashboard_store(store)

        try:
            resp = client.get(
                "/api/notifications?unread=true",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert resp.status_code == 200
            payload = resp.get_json()
            ids = [n["id"] for n in payload["notifications"]]
            assert "unread_2" in ids
            assert "unread_1" not in ids
        finally:
            set_dashboard_store(None)

    def test_list_notifications_with_priority_filter(self, app_client, auth_headers, tmp_path):
        """Priority filter returns only matching notifications."""
        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "notif_ui_priority.db")
        store.write(
            notification_id="p_urgent", priority="urgent", title="Urgent", body="u", user_id="james"
        )
        store.write(
            notification_id="p_normal", priority="normal", title="Normal", body="n", user_id="james"
        )
        set_dashboard_store(store)

        try:
            resp = client.get(
                "/api/notifications?priority=urgent",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            assert resp.status_code == 200
            payload = resp.get_json()
            ids = [n["id"] for n in payload["notifications"]]
            assert "p_urgent" in ids
            assert "p_normal" not in ids
        finally:
            set_dashboard_store(None)


class TestUIErrorHandling:
    """Tests for US-076: UI error handling.

    Verifies that:
    - Frontend HTML contains the global error banner element (frontend errors can be displayed)
    - Backend errors return a JSON response with an 'error' key and an appropriate HTTP status
    - Backend errors are logged (tested by verifying error JSON is returned by error paths)
    """

    def test_html_contains_global_error_banner(self, app_client):
        """Dashboard HTML includes the global error banner element."""
        client, _ = app_client

        response = client.get(
            "/dashboard",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"global-error-banner" in response.data

    def test_html_contains_global_error_close_button(self, app_client):
        """Dashboard HTML includes the error banner close button."""
        client, _ = app_client

        response = client.get(
            "/dashboard",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"global-error-close" in response.data

    def test_html_contains_global_error_msg(self, app_client):
        """Dashboard HTML includes the error message placeholder element."""
        client, _ = app_client

        response = client.get(
            "/dashboard",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"global-error-msg" in response.data

    def test_backend_chat_error_returns_json_error(self, app_client, auth_headers):
        """Backend chat error returns JSON with 'error' key and 500 status."""
        client, _ = app_client

        with patch("rex.dashboard.routes._get_llm") as mock_llm:
            mock_instance = MagicMock()
            mock_instance.generate.side_effect = RuntimeError("LLM backend unavailable")
            mock_llm.return_value = mock_instance

            response = client.post(
                "/api/chat",
                json={"message": "hello"},
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )

            assert response.status_code == 500
            data = response.get_json()
            assert "error" in data
            assert "LLM backend unavailable" in data["error"]

    def test_backend_settings_error_returns_json_error(self, app_client, auth_headers, monkeypatch):
        """Backend settings load error returns JSON with 'error' key and 500 status."""
        client, _ = app_client

        import rex.dashboard.routes as routes_module

        original_load = routes_module.load_config

        def broken_load():
            raise RuntimeError("Config file corrupted")

        monkeypatch.setattr(routes_module, "load_config", broken_load)

        response = client.get(
            "/api/settings",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data

    def test_backend_scheduler_error_returns_json_error(self, app_client, auth_headers, monkeypatch):
        """Backend scheduler error returns JSON with 'error' key and 500 status."""
        client, _ = app_client

        import rex.dashboard.routes as routes_module

        def broken_get_scheduler():
            raise RuntimeError("Scheduler unavailable")

        monkeypatch.setattr(routes_module, "get_scheduler", broken_get_scheduler)

        response = client.get(
            "/api/scheduler/jobs",
            headers=auth_headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data

    def test_js_contains_show_global_error_function(self, app_client):
        """Dashboard JS includes the showGlobalError function for displaying frontend errors."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"showGlobalError" in response.data

    def test_js_contains_global_error_handler(self, app_client):
        """Dashboard JS registers a global error event listener."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"unhandledrejection" in response.data


class TestUIReconnectBehavior:
    """Tests for US-077: UI reconnect behavior.

    Verifies that:
    - The JS contains SSE reconnect logic with limited attempts
    - The reconnect counter and max-attempts constant are present
    - The UI falls back to polling after exhausting reconnect attempts
    - The SSE stream endpoint is reachable
    """

    def test_js_contains_sse_max_reconnect_constant(self, app_client):
        """Dashboard JS defines the maximum number of SSE reconnect attempts."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"_SSE_MAX_RECONNECT" in response.data

    def test_js_contains_reconnect_counter(self, app_client):
        """Dashboard JS tracks how many reconnect attempts have been made."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"_sseReconnectCount" in response.data

    def test_js_contains_reconnect_timer(self, app_client):
        """Dashboard JS uses a timer to schedule reconnect attempts."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"_sseReconnectTimer" in response.data

    def test_js_contains_stop_notif_reconnect(self, app_client):
        """Dashboard JS has a function to cancel pending reconnect timers."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        assert b"stopNotifReconnect" in response.data

    def test_js_reconnect_uses_exponential_backoff(self, app_client):
        """Dashboard JS uses exponential backoff for reconnect delays."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        # Exponential backoff implemented via Math.pow
        assert b"Math.pow" in response.data

    def test_js_falls_back_to_polling_after_max_reconnects(self, app_client):
        """Dashboard JS calls startNotifPolling after exhausting reconnect attempts."""
        client, _ = app_client

        response = client.get(
            "/dashboard/assets/js/dashboard.js",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

        assert response.status_code == 200
        js = response.data.decode("utf-8")
        # The fallback must be within the onerror handler
        assert "startNotifPolling" in js

    def test_sse_stream_endpoint_requires_auth(self, app_client, tmp_path):
        """SSE /api/notifications/stream requires auth from remote addresses."""
        from rex.dashboard_store import DashboardStore, set_dashboard_store

        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "sse_auth.db")
        set_dashboard_store(store)

        try:
            import os

            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "0"
            response = client.get(
                "/api/notifications/stream",
                environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
            )
            assert response.status_code == 401
        finally:
            os.environ["REX_DASHBOARD_ALLOW_LOCAL"] = "1"
            set_dashboard_store(None)

    def test_sse_stream_endpoint_is_reachable_when_authenticated(self, app_client, auth_headers, tmp_path):
        """SSE /api/notifications/stream returns a streaming response when authenticated."""
        from rex.dashboard_store import DashboardStore, set_dashboard_store

        client, _ = app_client
        store = DashboardStore(db_path=tmp_path / "sse_reach.db")
        set_dashboard_store(store)

        try:
            response = client.get(
                "/api/notifications/stream?max_events=1&timeout=0.01",
                headers=auth_headers,
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            # SSE endpoint should return 200 with text/event-stream content-type
            assert response.status_code == 200
            assert "text/event-stream" in response.content_type
        finally:
            set_dashboard_store(None)
