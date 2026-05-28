"""Tests for US-RR-007: Protect /setup and /register routes.

Covers:
- POST /api/setup/complete without X-Setup-Token returns 403
- POST /api/setup/complete with wrong X-Setup-Token returns 403
- POST /api/setup/complete second call after token consumed returns 403
- POST /api/setup/complete with valid token succeeds (201)
- POST /api/auth/register without token when no users exist returns 403
- POST /api/auth/register with wrong token when no users exist returns 403
- POST /api/auth/register with valid token when no users exist succeeds (201)
- POST /api/auth/register after first user exists succeeds without token (no longer pre-setup)
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-rr007-secret")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _setup_token(client: object) -> str:
    return client.application.config.get("SETUP_TOKEN") or ""  # type: ignore[attr-defined]


_SETUP_PAYLOAD = {
    "username": "admin",
    "password": "securepass1",
    "llm_provider": "local",
    "tts_provider": "none",
}


# ---------------------------------------------------------------------------
# /api/setup/complete protection
# ---------------------------------------------------------------------------


class TestSetupCompleteProtection:
    def test_no_token_returns_403(self, flask_client) -> None:
        resp = flask_client.post("/api/setup/complete", json=_SETUP_PAYLOAD)
        assert resp.status_code == 403

    def test_wrong_token_returns_403(self, flask_client) -> None:
        resp = flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": "not-the-real-token"},
        )
        assert resp.status_code == 403

    def test_valid_token_returns_201(self, flask_client) -> None:
        resp = flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["ok"] is True

    def test_token_consumed_after_success_returns_403(self, flask_client) -> None:
        """After setup completes the single-use token is invalidated."""
        flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        # Second attempt — even with the same token string it must be rejected.
        resp = flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert resp.status_code == 403

    def test_setup_token_cleared_in_app_config_after_success(self, flask_client) -> None:
        flask_client.post(
            "/api/setup/complete",
            json=_SETUP_PAYLOAD,
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert flask_client.application.config.get("SETUP_TOKEN") is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# /api/auth/register protection (pre-setup window)
# ---------------------------------------------------------------------------


class TestRegisterProtection:
    def test_no_token_when_no_users_returns_403(self, flask_client) -> None:
        resp = flask_client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "pass1234"},
        )
        assert resp.status_code == 403

    def test_wrong_token_when_no_users_returns_403(self, flask_client) -> None:
        resp = flask_client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "pass1234"},
            headers={"X-Setup-Token": "invalid-token"},
        )
        assert resp.status_code == 403

    def test_valid_token_when_no_users_returns_201(self, flask_client) -> None:
        resp = flask_client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "pass1234"},
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["username"] == "alice"

    def test_no_token_required_after_first_user_exists(self, flask_client) -> None:
        """Once a user exists the pre-setup window is closed; register works normally."""
        # Create first user with the setup token.
        flask_client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "pass1234"},
            headers={"X-Setup-Token": _setup_token(flask_client)},
        )
        # Second user does not require a setup token.
        resp = flask_client.post(
            "/api/auth/register",
            json={"username": "bob", "password": "pass1234"},
        )
        assert resp.status_code == 201
        assert resp.get_json()["username"] == "bob"
