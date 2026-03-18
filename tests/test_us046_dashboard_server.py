"""Tests for US-046: Dashboard server.

Acceptance criteria:
- server starts
- API reachable
- health endpoint works
- Typecheck passes
"""

from __future__ import annotations

import pytest
from flask import Flask

from rex.dashboard import dashboard_bp


@pytest.fixture()
def app():
    """Create a minimal Flask app with the dashboard blueprint registered."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# --- server starts ---


def test_blueprint_registers_without_error():
    """Dashboard blueprint can be registered on a fresh Flask app."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(dashboard_bp)
    assert flask_app is not None


def test_app_has_dashboard_routes(app):
    """Dashboard routes are available after blueprint registration."""
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/api/dashboard/status" in rules
    assert "/api/dashboard/login" in rules


# --- API reachable ---


def test_status_endpoint_reachable(client):
    """GET /api/dashboard/status returns 200."""
    response = client.get("/api/dashboard/status")
    assert response.status_code == 200


def test_status_endpoint_returns_json(client):
    """GET /api/dashboard/status returns valid JSON."""
    response = client.get("/api/dashboard/status")
    data = response.get_json()
    assert data is not None
    assert isinstance(data, dict)


def test_login_endpoint_reachable(client, monkeypatch):
    """POST /api/dashboard/login is reachable (returns a response, not 404/405)."""
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "wrong"},
        content_type="application/json",
    )
    # Any response other than 404/405 means the endpoint is registered
    assert response.status_code not in (404, 405)


# --- health endpoint works ---


def test_health_status_ok(client):
    """Health endpoint returns status=ok."""
    response = client.get("/api/dashboard/status")
    data = response.get_json()
    assert data["status"] == "ok"


def test_health_endpoint_has_version(client):
    """Health endpoint includes a version field."""
    response = client.get("/api/dashboard/status")
    data = response.get_json()
    assert "version" in data
    assert isinstance(data["version"], str)


def test_health_endpoint_has_uptime(client):
    """Health endpoint includes non-negative uptime_seconds."""
    response = client.get("/api/dashboard/status")
    data = response.get_json()
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


def test_health_endpoint_has_server_time(client):
    """Health endpoint includes a server_time field."""
    response = client.get("/api/dashboard/status")
    data = response.get_json()
    assert "server_time" in data
    assert isinstance(data["server_time"], str)


def test_health_endpoint_has_auth_enabled(client):
    """Health endpoint includes auth_enabled boolean."""
    response = client.get("/api/dashboard/status")
    data = response.get_json()
    assert "auth_enabled" in data
    assert isinstance(data["auth_enabled"], bool)


def test_health_endpoint_no_auth_required(client):
    """Health endpoint is publicly accessible (no authentication required)."""
    # No auth headers - should still return 200
    response = client.get("/api/dashboard/status")
    assert response.status_code == 200


def test_health_endpoint_content_type_json(client):
    """Health endpoint returns application/json content type."""
    response = client.get("/api/dashboard/status")
    assert "application/json" in response.content_type


def test_multiple_status_requests(client):
    """Health endpoint handles multiple consecutive requests."""
    for _ in range(3):
        response = client.get("/api/dashboard/status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
