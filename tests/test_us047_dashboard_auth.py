"""Tests for US-047: Dashboard authentication.

Acceptance criteria:
- login works
- sessions created
- invalid logins rejected
- Typecheck passes
"""

from __future__ import annotations

import pytest
from flask import Flask

from rex.dashboard import dashboard_bp
from rex.dashboard.auth import SessionManager, get_session_manager


@pytest.fixture(autouse=True)
def isolate_session_manager(monkeypatch):
    """Give each test a fresh SessionManager so state does not bleed between tests."""
    import rex.dashboard.auth as auth_module
    import rex.dashboard.routes as routes_module

    fresh = SessionManager(expiry_seconds=3600)
    monkeypatch.setattr(auth_module, "_session_manager", fresh)
    monkeypatch.setattr(routes_module, "get_session_manager", lambda: fresh)
    yield fresh


@pytest.fixture()
def app():
    """Flask test app with dashboard blueprint and TESTING=True."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def remote_client(app):
    """Test client that appears to come from a non-loopback address."""
    return app.test_client()


def _non_local_env():
    """environ_base simulating a remote (non-loopback) client."""
    return {"REMOTE_ADDR": "10.0.0.1"}


# --- login works ---


def test_login_with_correct_password_returns_200(client, monkeypatch):
    """POST /api/dashboard/login with correct password returns 200."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    assert response.status_code == 200


def test_login_response_includes_token(client, monkeypatch):
    """Successful login response body contains a non-empty token."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    data = response.get_json()
    assert "token" in data
    assert isinstance(data["token"], str)
    assert len(data["token"]) > 0


def test_login_response_includes_expires_at(client, monkeypatch):
    """Successful login response body contains expires_at timestamp."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    data = response.get_json()
    assert "expires_at" in data
    assert isinstance(data["expires_at"], str)


def test_login_sets_cookie(client, monkeypatch):
    """Successful login sets the rex_dashboard_token cookie."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    assert response.status_code == 200
    assert any("rex_dashboard_token" in c for c in response.headers.getlist("Set-Cookie"))


def test_local_login_without_password_succeeds(client, monkeypatch):
    """Local (loopback) login with no password configured returns 200."""
    monkeypatch.delenv("REX_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "1")
    response = client.post(
        "/api/dashboard/login",
        json={},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "token" in data


# --- sessions created ---


def test_login_creates_a_valid_session(client, monkeypatch, isolate_session_manager):
    """Token returned by login can be validated by the session manager."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    token = response.get_json()["token"]
    session = isolate_session_manager.validate_session(token)
    assert session is not None
    assert session.token == token


def test_session_count_increases_after_login(client, monkeypatch, isolate_session_manager):
    """Active session count increments after a successful login."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    before = isolate_session_manager.get_active_session_count()
    client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    after = isolate_session_manager.get_active_session_count()
    assert after == before + 1


def test_session_token_accepted_in_bearer_header(client, monkeypatch, isolate_session_manager):
    """Token returned by login is accepted as a Bearer token on protected endpoints."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    token = response.get_json()["token"]
    protected = client.get(
        "/api/chat/history",
        headers={"Authorization": f"Bearer {token}"},
        environ_base=_non_local_env(),
    )
    assert protected.status_code == 200


def test_session_token_accepted_in_x_dashboard_token_header(
    client, monkeypatch, isolate_session_manager
):
    """Token is accepted via X-Dashboard-Token header."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    token = response.get_json()["token"]
    protected = client.get(
        "/api/chat/history",
        headers={"X-Dashboard-Token": token},
        environ_base=_non_local_env(),
    )
    assert protected.status_code == 200


def test_logout_invalidates_session(client, monkeypatch, isolate_session_manager):
    """Logout invalidates the session so the token no longer works."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "s3cr3t")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "s3cr3t"},
        environ_base=_non_local_env(),
    )
    token = response.get_json()["token"]

    client.post(
        "/api/dashboard/logout",
        headers={"Authorization": f"Bearer {token}"},
        environ_base=_non_local_env(),
    )

    session = isolate_session_manager.validate_session(token)
    assert session is None


# --- invalid logins rejected ---


def test_wrong_password_returns_401(client, monkeypatch):
    """Wrong password returns HTTP 401."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "correct")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "wrong"},
        environ_base=_non_local_env(),
    )
    assert response.status_code == 401


def test_wrong_password_returns_error_message(client, monkeypatch):
    """Wrong password response body contains an error field."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "correct")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": "wrong"},
        environ_base=_non_local_env(),
    )
    data = response.get_json()
    assert "error" in data


def test_empty_password_rejected(client, monkeypatch):
    """Empty password string is rejected with 401."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "correct")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={"password": ""},
        environ_base=_non_local_env(),
    )
    assert response.status_code == 401


def test_missing_password_field_rejected(client, monkeypatch):
    """Request with no password field is rejected."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "correct")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.post(
        "/api/dashboard/login",
        json={},
        environ_base=_non_local_env(),
    )
    assert response.status_code == 401


def test_invalid_token_rejected_on_protected_endpoint(client, monkeypatch):
    """A made-up token is rejected with 401 on protected endpoints."""
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.get(
        "/api/chat/history",
        headers={"Authorization": "Bearer totally-fake-token"},
        environ_base=_non_local_env(),
    )
    assert response.status_code == 401


def test_no_token_rejected_on_protected_endpoint(client, monkeypatch):
    """Unauthenticated request to protected endpoint returns 401."""
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    response = client.get(
        "/api/chat/history",
        environ_base=_non_local_env(),
    )
    assert response.status_code == 401


def test_multiple_wrong_attempts_all_rejected(client, monkeypatch):
    """Multiple consecutive wrong password attempts are all rejected."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "correct")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    for _ in range(3):
        response = client.post(
            "/api/dashboard/login",
            json={"password": "bad"},
            environ_base=_non_local_env(),
        )
        assert response.status_code == 401
