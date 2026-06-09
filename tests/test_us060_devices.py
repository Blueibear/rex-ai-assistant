"""Tests for US-060: Device control panel API.

Covers:
- GET /api/devices returns devices from device_aliases.json
- GET /api/devices returns empty list when file missing or malformed
- POST /api/devices/<entity_id>/command requires auth
- POST /api/devices/<entity_id>/command returns 400 for missing command
- POST /api/devices/<entity_id>/command returns 400 for unknown command
- POST /api/devices/<entity_id>/command forwards turn_on/turn_off to HA
- POST /api/devices/<entity_id>/command returns 503 when HA not configured
- POST /api/devices/<entity_id>/command handles set_brightness with value
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-us060-secret-long-enough")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _register_and_login(client, username: str = "admin", password: str = "pass1234x!") -> str:
    setup_token = client.application.config.get("SETUP_TOKEN") or ""
    client.post(
        "/api/auth/register",
        json={"username": username, "password": password},
        headers={"X-Setup-Token": setup_token},
    )
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.get_json()["token"]  # type: ignore[index]


def _write_device_aliases(tmp_path: Path, devices: list) -> None:
    aliases_file = tmp_path / "config" / "device_aliases.json"
    aliases_file.parent.mkdir(parents=True, exist_ok=True)
    aliases_file.write_text(json.dumps({"devices": devices}), encoding="utf-8")


# ---------------------------------------------------------------------------
# GET /api/devices
# ---------------------------------------------------------------------------


class TestListDevices:
    def test_returns_200_with_devices_key(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/devices")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "devices" in data

    def test_returns_devices_from_config(
        self, flask_client, tmp_data_dir: Path
    ) -> None:  # type: ignore[override]
        sample = [{"entity_id": "light.kitchen", "name": "Kitchen Light", "type": "light"}]
        with patch(
            "rex.bridge_utils.repo_root",
            return_value=tmp_data_dir,
        ):
            aliases_file = tmp_data_dir / "config" / "device_aliases.json"
            aliases_file.parent.mkdir(parents=True, exist_ok=True)
            aliases_file.write_text(json.dumps({"devices": sample}), encoding="utf-8")
            resp = flask_client.get("/api/devices")

        data = resp.get_json()
        assert len(data["devices"]) == 1
        assert data["devices"][0]["entity_id"] == "light.kitchen"

    def test_returns_empty_list_on_missing_file(self, flask_client) -> None:  # type: ignore[override]
        with patch("rex.bridge_utils.repo_root", side_effect=RuntimeError("no root")):
            with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
                resp = flask_client.get("/api/devices")
        assert resp.status_code == 200
        assert resp.get_json()["devices"] == []

    def test_no_auth_required(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/devices")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/devices/<entity_id>/command
# ---------------------------------------------------------------------------


class TestDeviceCommand:
    def test_requires_auth(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.post(
            "/api/devices/light.living_room/command", json={"command": "turn_on"}
        )
        assert resp.status_code == 401

    def test_missing_command_returns_400(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        resp = flask_client.post(
            "/api/devices/light.living_room/command",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400

    def test_unknown_command_returns_400(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)
        resp = flask_client.post(
            "/api/devices/light.living_room/command",
            json={"command": "explode"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400

    def test_ha_not_configured_returns_503(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)

        mock_cfg = MagicMock()
        mock_cfg.ha_base_url = ""
        mock_cfg.ha_token = ""

        with patch("rex.config.load_config", return_value=mock_cfg):
            resp = flask_client.post(
                "/api/devices/light.living_room/command",
                json={"command": "turn_on"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 503

    def test_turn_on_sends_request_to_ha(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)

        mock_cfg = MagicMock()
        mock_cfg.ha_base_url = "http://ha.local:8123"
        mock_cfg.ha_token = "my-token"

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("rex.config.load_config", return_value=mock_cfg),
            patch("rex.routes.ha._request_home_assistant", return_value=mock_resp) as mock_request,
        ):
            resp = flask_client.post(
                "/api/devices/light.living_room/command",
                json={"command": "turn_on"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        mock_request.assert_called_once()

    def test_set_brightness_includes_value(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)

        mock_cfg = MagicMock()
        mock_cfg.ha_base_url = "http://ha.local:8123"
        mock_cfg.ha_token = "my-token"

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        sent_body: list[bytes] = []

        def capture_request(
            url: str,
            *,
            method: str = "GET",
            headers: dict[str, str] | None = None,
            body: bytes | None = None,
            timeout: float = 5,
            ssl_context=None,
        ):  # type: ignore[no-untyped-def]
            sent_body.append(body or b"")
            return mock_resp

        with (
            patch("rex.config.load_config", return_value=mock_cfg),
            patch("rex.routes.ha._request_home_assistant", side_effect=capture_request),
        ):
            flask_client.post(
                "/api/devices/light.living_room/command",
                json={"command": "set_brightness", "value": 128},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert sent_body
        body = json.loads(sent_body[0])
        assert body["brightness"] == 128

    def test_failed_ha_call_returns_ok_false(self, flask_client) -> None:  # type: ignore[override]
        token = _register_and_login(flask_client)

        mock_cfg = MagicMock()
        mock_cfg.ha_base_url = "http://ha.local:8123"
        mock_cfg.ha_token = ""

        with (
            patch("rex.config.load_config", return_value=mock_cfg),
            patch(
                "rex.routes.ha._request_home_assistant",
                side_effect=OSError("connection refused"),
            ),
        ):
            resp = flask_client.post(
                "/api/devices/switch.fan/command",
                json={"command": "turn_off"},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is False
        assert "error" in data
