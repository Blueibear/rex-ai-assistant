"""Tests for US-059: Home Assistant setup screen API.

Covers:
- POST /api/ha/test: missing ha_base_url returns 400
- POST /api/ha/test: successful HA response returns ok=True
- POST /api/ha/test: failed HA connection returns ok=False with error
- POST /api/ha/test: unauthenticated request returns 401 (US-RR-009)
- POST /api/ha/save: requires authentication
- POST /api/ha/save: missing ha_base_url returns 400
- POST /api/ha/save: valid request writes ha_base_url to config
- POST /api/ha/save: ha_token written via _write_env_secrets
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-us059-secret")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _register_and_login(client, username: str = "admin", password: str = "pass1234") -> str:
    setup_token = client.application.config.get("SETUP_TOKEN") or ""
    client.post(
        "/api/auth/register",
        json={"username": username, "password": password},
        headers={"X-Setup-Token": setup_token},
    )
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.get_json()["token"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# POST /api/ha/test
# ---------------------------------------------------------------------------


class TestHaTest:
    def test_missing_base_url_returns_400(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        resp = flask_client.post(
            "/api/ha/test",
            json={"ha_token": "tok"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400

    def test_unauthenticated_returns_401(self, flask_client) -> None:  # type: ignore[override]
        """Route now requires auth — unauthenticated request must be rejected."""
        with patch("rex.routes.ha._request_home_assistant", side_effect=OSError("no host")):
            resp = flask_client.post("/api/ha/test", json={"ha_base_url": "http://ha.local:8123"})
        assert resp.status_code == 401

    def test_successful_connection_returns_ok_true(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("rex.routes.ha._request_home_assistant", return_value=mock_resp):
            resp = flask_client.post(
                "/api/ha/test",
                json={"ha_base_url": "http://ha.local:8123", "ha_token": "good-token"},
                headers={"Authorization": f"Bearer {token}"},
            )
        data = resp.get_json()
        assert data["ok"] is True

    def test_failed_connection_returns_ok_false(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        with patch(
            "rex.routes.ha._request_home_assistant",
            side_effect=OSError("connection refused"),
        ):
            resp = flask_client.post(
                "/api/ha/test",
                json={"ha_base_url": "http://ha.local:8123"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is False
        assert "error" in data

    def test_non_200_response_returns_ok_false(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        mock_resp = MagicMock()
        mock_resp.status = 401
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("rex.routes.ha._request_home_assistant", return_value=mock_resp):
            resp = flask_client.post(
                "/api/ha/test",
                json={"ha_base_url": "http://ha.local:8123"},
                headers={"Authorization": f"Bearer {token}"},
            )
        data = resp.get_json()
        assert data["ok"] is False


# ---------------------------------------------------------------------------
# POST /api/ha/save
# ---------------------------------------------------------------------------


class TestHaSave:
    def test_requires_auth(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.post(
            "/api/ha/save",
            json={"ha_base_url": "http://ha.local:8123"},
        )
        assert resp.status_code == 401

    def test_missing_base_url_returns_400(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        resp = flask_client.post(
            "/api/ha/save",
            json={"ha_token": "tok"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400

    def test_valid_save_returns_ok(self, flask_client, tmp_path: Path) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)

        with (
            patch("rex.config_manager.load_config", return_value={}),
            patch("rex.config_manager.save_config") as mock_save,
            patch("rex.gui_app._write_env_secrets"),
        ):
            resp = flask_client.post(
                "/api/ha/save",
                json={"ha_base_url": "http://ha.local:8123", "ha_token": "my-token"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        mock_save.assert_called_once()

    def test_saves_base_url_to_config(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        saved: dict[str, object] = {}

        def fake_save(cfg: dict, path: str = "") -> None:  # type: ignore[override]
            saved.update(cfg)

        with (
            patch("rex.config_manager.load_config", return_value={}),
            patch("rex.config_manager.save_config", side_effect=fake_save),
            patch("rex.gui_app._write_env_secrets"),
        ):
            flask_client.post(
                "/api/ha/save",
                json={"ha_base_url": "http://my-ha.local:8123"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert saved.get("home_assistant", {}).get("base_url") == "http://my-ha.local:8123"  # type: ignore[union-attr]

    def test_token_written_via_write_env_secrets(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        captured: dict[str, str] = {}

        def fake_write_env(
            path: Path, *, llm_provider: str, llm_api_key: str, ha_token: str
        ) -> None:
            captured["ha_token"] = ha_token

        with (
            patch("rex.config_manager.load_config", return_value={}),
            patch("rex.config_manager.save_config"),
            patch("rex.gui_app._write_env_secrets", side_effect=fake_write_env),
            patch("rex.bridge_utils.repo_root", return_value=Path("/tmp")),
        ):
            flask_client.post(
                "/api/ha/save",
                json={"ha_base_url": "http://ha.local:8123", "ha_token": "secret-ha-token"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert captured.get("ha_token") == "secret-ha-token"
