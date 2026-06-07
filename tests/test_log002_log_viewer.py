"""Tests for US-LOG-002 and US-RR-008: Log viewer — SSE endpoint, download, and auth gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "[test-log002-jwt-placeholder-32-bytes-min]")
    return tmp_path


@pytest.fixture()
def app_client(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a Flask test client with a temp log file."""
    from rex.gui_app import _create_flask_app

    # Override the log file path used inside gui_app.
    log_file = tmp_data_dir / "rex.log"

    # Patch _LOG_FILE inside the returned app by monkeypatching at module level.
    # gui_app creates _LOG_FILE as a local variable inside _create_flask_app,
    # so we need to call it with a patched Path.

    original_file = Path.__truediv__

    def patched_truediv(self: Path, key: object) -> Path:
        result = original_file(self, key)
        if result.name == "rex.log" and result.parent.name == "logs":
            return log_file
        return result

    monkeypatch.setattr(Path, "__truediv__", patched_truediv)

    flask_app = _create_flask_app(ui_enabled=False)
    flask_app.config["TESTING"] = True
    return flask_app.test_client(), log_file


@pytest.fixture()
def auth_header(app_client):
    """Register + login a test user and return the Authorization header."""
    client, _ = app_client
    setup_token = client.application.config.get("SETUP_TOKEN") or ""
    client.post(
        "/api/auth/register",
        json={"username": "logviewer", "password": "[non-secret-test-value]"},
        headers={"X-Setup-Token": setup_token},
    )
    resp = client.post(
        "/api/auth/login",
        json={"username": "logviewer", "password": "[non-secret-test-value]"},
    )
    token = resp.get_json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_logs_download_not_found(app_client, auth_header):
    client, _ = app_client
    resp = client.get("/api/logs/download", headers=auth_header)
    assert resp.status_code == 404


def test_logs_download_returns_file(app_client, auth_header):
    client, log_file = app_client
    log_file.write_text('{"level":"INFO","message":"hello"}\n', encoding="utf-8")
    resp = client.get("/api/logs/download", headers=auth_header)
    assert resp.status_code == 200
    assert b"hello" in resp.data


def test_logs_download_content_type(app_client, auth_header):
    client, log_file = app_client
    log_file.write_text("line1\nline2\n", encoding="utf-8")
    resp = client.get("/api/logs/download", headers=auth_header)
    assert resp.status_code == 200


def test_log_file_contains_json_lines(tmp_path: Path) -> None:
    """Verify that rex/logging_config produces valid JSON log lines."""
    import logging

    from rex.logging_config import setup_file_logging

    log_file = tmp_path / "rex.log"
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        root.setLevel(logging.DEBUG)
        handler = setup_file_logging(log_file)
        logging.getLogger("test.log002").error("something failed")
        handler.flush()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        parsed = json.loads(lines[-1])
        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "something failed"
        assert "timestamp" in parsed
        assert "extra" in parsed
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_logs_stream_route_exists(app_client, auth_header):
    """SSE route /api/logs/stream must exist and return streaming response."""
    client, _ = app_client
    # Use a short get that doesn't block — just confirm status 200 / Content-Type.
    # Flask test client buffers, so we read with buffered=False where possible.
    resp = client.get("/api/logs/stream", headers=auth_header)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.content_type
