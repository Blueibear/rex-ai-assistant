"""Tests for US-RR-008: Protect log streaming/download routes with authentication.

Covers:
- GET /api/logs/stream without token returns 401
- GET /api/logs/download without token returns 401
- GET /api/logs/stream with valid token returns 200 text/event-stream
- GET /api/logs/download with valid token and missing file returns 404
- GET /api/logs/download with valid token and existing file returns 200 with content
- Home-directory paths are redacted from download response
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
    monkeypatch.setenv("REX_JWT_SECRET", "test-rr008-secret")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Flask test client with a patched log file path."""
    from rex.gui_app import _create_flask_app

    log_file = tmp_data_dir / "rex.log"

    original_truediv = Path.__truediv__

    def patched_truediv(self: Path, key: object) -> Path:
        result = original_truediv(self, key)
        if result.name == "rex.log" and result.parent.name == "logs":
            return log_file
        return result

    monkeypatch.setattr(Path, "__truediv__", patched_truediv)

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, log_file


@pytest.fixture()
def auth_header(flask_client):
    """Register + login a test user and return the Authorization header."""
    client, _ = flask_client
    setup_token = client.application.config.get("SETUP_TOKEN") or ""
    client.post(
        "/api/auth/register",
        json={"username": "logadmin", "password": "securepass1"},
        headers={"X-Setup-Token": setup_token},
    )
    resp = client.post(
        "/api/auth/login",
        json={"username": "logadmin", "password": "securepass1"},
    )
    token = resp.get_json()["token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Unauthenticated access must be rejected
# ---------------------------------------------------------------------------


class TestLogRoutesRequireAuth:
    def test_stream_without_token_returns_401(self, flask_client) -> None:
        client, _ = flask_client
        resp = client.get("/api/logs/stream")
        assert resp.status_code == 401

    def test_download_without_token_returns_401(self, flask_client) -> None:
        client, _ = flask_client
        resp = client.get("/api/logs/download")
        assert resp.status_code == 401

    def test_stream_with_wrong_token_returns_401(self, flask_client) -> None:
        client, _ = flask_client
        resp = client.get(
            "/api/logs/stream",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401

    def test_download_with_wrong_token_returns_401(self, flask_client) -> None:
        client, _ = flask_client
        resp = client.get(
            "/api/logs/download",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Authenticated access succeeds
# ---------------------------------------------------------------------------


class TestLogRoutesAuthenticatedAccess:
    def test_stream_with_valid_token_returns_event_stream(self, flask_client, auth_header) -> None:
        client, _ = flask_client
        resp = client.get("/api/logs/stream", headers=auth_header)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

    def test_download_with_valid_token_file_missing_returns_404(
        self, flask_client, auth_header
    ) -> None:
        client, _ = flask_client
        resp = client.get("/api/logs/download", headers=auth_header)
        assert resp.status_code == 404
        # Error response must NOT disclose filesystem path
        body = resp.get_json() or {}
        assert "active_log_path" not in body
        assert "legacy_log_path" not in body

    def test_download_with_valid_token_returns_file_content(
        self, flask_client, auth_header
    ) -> None:
        client, log_file = flask_client
        log_file.write_text('{"level":"INFO","message":"startup"}\n', encoding="utf-8")
        resp = client.get("/api/logs/download", headers=auth_header)
        assert resp.status_code == 200
        assert b"startup" in resp.data

    def test_download_redacts_home_directory_path(
        self, flask_client, auth_header, tmp_data_dir: Path
    ) -> None:
        """Home-directory paths in log lines must be replaced with ~ in download."""
        client, log_file = flask_client
        home = str(Path.home())
        log_file.write_text(
            f'{{"level":"INFO","message":"config at {home}/rex/config.json"}}\n',
            encoding="utf-8",
        )
        resp = client.get("/api/logs/download", headers=auth_header)
        assert resp.status_code == 200
        assert home.encode() not in resp.data
        assert b"~/rex/config.json" in resp.data
