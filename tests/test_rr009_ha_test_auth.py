"""Security tests for US-RR-009: Protect HA connection-test route.

Covers:
- Unauthenticated requests to /api/ha/test are rejected with 401
- Invalid URL schemes (file://, ftp://, data://) are rejected with 400
- Valid authenticated request with http:// scheme succeeds
- Valid authenticated request with https:// scheme succeeds
- Raw exception text is not returned to the client
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-rr009-secret")
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


class TestHaTestAuthRequired:
    def test_unauthenticated_no_header_returns_401(self, flask_client) -> None:
        """No Authorization header → 401."""
        resp = flask_client.post("/api/ha/test", json={"ha_base_url": "http://ha.local:8123"})
        assert resp.status_code == 401

    def test_unauthenticated_invalid_token_returns_401(self, flask_client) -> None:
        """Bogus Bearer token → 401."""
        resp = flask_client.post(
            "/api/ha/test",
            json={"ha_base_url": "http://ha.local:8123"},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401


class TestHaTestSchemeValidation:
    def test_file_scheme_rejected_with_400(self, flask_client) -> None:
        """file:// scheme must be rejected to prevent local filesystem reads."""
        token = _register_and_login(flask_client)
        resp = flask_client.post(
            "/api/ha/test",
            json={"ha_base_url": "file:///etc/passwd"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False

    def test_ftp_scheme_rejected_with_400(self, flask_client) -> None:
        """ftp:// scheme must be rejected."""
        token = _register_and_login(flask_client)
        resp = flask_client.post(
            "/api/ha/test",
            json={"ha_base_url": "ftp://attacker.example.com/evil"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False

    def test_data_scheme_rejected_with_400(self, flask_client) -> None:
        """data: URI scheme must be rejected."""
        token = _register_and_login(flask_client)
        resp = flask_client.post(
            "/api/ha/test",
            json={"ha_base_url": "data:text/plain,payload"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400

    def test_http_scheme_accepted(self, flask_client) -> None:
        """http:// is a valid scheme."""
        token = _register_and_login(flask_client)
        with patch("rex.routes.ha._request_home_assistant", side_effect=OSError("no host")):
            resp = flask_client.post(
                "/api/ha/test",
                json={"ha_base_url": "http://ha.local:8123"},
                headers={"Authorization": f"Bearer {token}"},
            )
        # Connection failure → ok=False, but not a 400 scheme error
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is False

    def test_https_scheme_accepted(self, flask_client) -> None:
        """https:// is a valid scheme."""
        token = _register_and_login(flask_client)
        with patch("rex.routes.ha._request_home_assistant", side_effect=OSError("no host")):
            resp = flask_client.post(
                "/api/ha/test",
                json={"ha_base_url": "https://ha.example.com:8123"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is False


class TestHaTestErrorRedaction:
    def test_raw_exception_text_not_returned(self, flask_client) -> None:
        """Connection errors must not leak internal exception details."""
        token = _register_and_login(flask_client)
        internal_message = "INTERNAL_STACK_TRACE_DETAIL_xyz123"
        with patch(
            "rex.routes.ha._request_home_assistant",
            side_effect=OSError(internal_message),
        ):
            resp = flask_client.post(
                "/api/ha/test",
                json={"ha_base_url": "http://ha.local:8123"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert internal_message not in body
        data = resp.get_json()
        assert data["ok"] is False
        assert "error" in data
