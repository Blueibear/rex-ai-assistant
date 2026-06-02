"""Tests for US-REM-021: First-run setup flow.

Covers the end-to-end scenario from a clean install state:
- Clean state: setup endpoint completes with a valid setup token (creates user)
- After setup: user can authenticate via /api/auth/login and receive a JWT
- Second setup attempt: token is consumed, returns 403
- Invalid/missing setup token: returns 403

Note: REX_JWT_SECRET raises RuntimeError when unset (no hardcoded fallback).
All tests set REX_JWT_SECRET via monkeypatch and redirect the SQLite user DB
to a tmp_path so no real data/ directory is written.
"""

from __future__ import annotations

from pathlib import Path

import jwt
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the user DB and JWT secret to a safe test environment."""
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-first-run-secret-rem021-xxxxxxxxxx")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _setup_token(client: object) -> str:
    """Return the single-use setup token stored in app config."""
    return client.application.config.get("SETUP_TOKEN") or ""  # type: ignore[attr-defined]


_SETUP_PAYLOAD = {
    "username": "admin",
    "password": "strongpass99",
    "llm_provider": "local",
    "tts_provider": "none",
}


# ---------------------------------------------------------------------------
# First-run setup: clean state → setup completes and creates a user
# ---------------------------------------------------------------------------


class TestFirstRunSetup:
    def test_setup_with_valid_token_returns_201(self, flask_client) -> None:
        """Clean state: setup endpoint with valid token succeeds and creates a user."""
        resp = flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["ok"] is True
        assert "user_id" in data

    def test_setup_with_missing_token_returns_403(self, flask_client) -> None:
        """Clean state: setup without X-Setup-Token header is rejected."""
        resp = flask_client.post("/api/setup/complete", json=_SETUP_PAYLOAD)
        assert resp.status_code == 403

    def test_setup_with_invalid_token_returns_403(self, flask_client) -> None:
        """Clean state: setup with a wrong token value is rejected."""
        resp = flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": "not-the-real-token"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Post-setup: authentication flow
# ---------------------------------------------------------------------------


class TestPostSetupAuthentication:
    def test_user_can_login_after_setup(self, flask_client) -> None:
        """After setup completes, the created user can authenticate and receive a JWT."""
        # Complete setup to create the user.
        flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )

        # Authenticate with the credentials used during setup.
        resp = flask_client.post(
            "/api/auth/login",
            json={"username": _SETUP_PAYLOAD["username"], "password": _SETUP_PAYLOAD["password"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "token" in data
        assert data["token"]

    def test_jwt_token_is_valid_and_decodable(self, flask_client) -> None:
        """The JWT returned after login is a well-formed token signed with REX_JWT_SECRET."""
        flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )

        login_resp = flask_client.post(
            "/api/auth/login",
            json={"username": _SETUP_PAYLOAD["username"], "password": _SETUP_PAYLOAD["password"]},
        )
        token = login_resp.get_json()["token"]

        payload = jwt.decode(token, "test-first-run-secret-rem021-xxxxxxxxxx", algorithms=["HS256"])
        assert payload["username"] == _SETUP_PAYLOAD["username"]

    def test_wrong_password_returns_401(self, flask_client) -> None:
        """After setup, authenticating with wrong credentials is rejected."""
        flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )

        resp = flask_client.post(
            "/api/auth/login",
            json={"username": _SETUP_PAYLOAD["username"], "password": "wrongpassword"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Setup token consumed after first use
# ---------------------------------------------------------------------------


class TestSetupTokenConsumed:
    def test_second_setup_attempt_returns_403(self, flask_client) -> None:
        """After setup completes the single-use token is invalidated; re-running setup is rejected."""
        token = _setup_token(flask_client)

        # First setup call — must succeed.
        first = flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": token},
        )
        assert first.status_code == 201

        # Second setup call — token is consumed; must be rejected even with same string.
        second = flask_client.post(
            "/api/setup/complete",
            json={**_SETUP_PAYLOAD, "username": "admin2"},
            headers={"X-Setup-Token": token},
        )
        assert second.status_code == 403

    def test_setup_token_is_none_after_completion(self, flask_client) -> None:
        """App config SETUP_TOKEN is cleared to None after successful setup."""
        flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert flask_client.application.config.get("SETUP_TOKEN") is None  # type: ignore[attr-defined]
