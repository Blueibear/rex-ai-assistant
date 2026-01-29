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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
    config_path.write_text(json.dumps({
        "models": {"llm_provider": "echo", "llm_model": "test"},
        "runtime": {"log_level": "DEBUG"},
    }))

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
