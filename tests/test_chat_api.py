"""Tests for the Rex GUI chat API endpoints (US-UI-002)."""

from __future__ import annotations

import json


def _make_app() -> object:
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    return app


def test_chat_history_empty() -> None:
    """GET /api/chat/history returns an empty list initially."""
    from rex.gui_app import _store_clear_history

    _store_clear_history()
    with _make_app().test_client() as client:
        resp = client.get("/api/chat/history")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert isinstance(data, list)


def test_chat_send_requires_message() -> None:
    """POST /api/chat/send with empty message returns 400."""
    from rex.gui_app import _store_clear_history

    _store_clear_history()
    with _make_app().test_client() as client:
        resp = client.post(
            "/api/chat/send",
            data=json.dumps({"message": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400


def test_chat_send_stores_user_message() -> None:
    """POST /api/chat/send adds the user message to history."""
    from rex.gui_app import _store_clear_history, _store_get_history

    _store_clear_history()
    with _make_app().test_client() as client:
        # Consume the SSE response fully
        resp = client.post(
            "/api/chat/send",
            data=json.dumps({"message": "hello rex"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        # Read the SSE stream
        body = b"".join(resp.response).decode()
        assert "data:" in body

    history = _store_get_history()
    roles = [m["role"] for m in history]
    assert "user" in roles
    assert "assistant" in roles


def test_chat_send_sse_content_type() -> None:
    """POST /api/chat/send returns text/event-stream content type."""
    from rex.gui_app import _store_clear_history

    _store_clear_history()
    with _make_app().test_client() as client:
        resp = client.post(
            "/api/chat/send",
            data=json.dumps({"message": "ping"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type


def test_chat_clear() -> None:
    """POST /api/chat/clear empties the history."""
    from rex.gui_app import _store_add_message, _store_clear_history, _store_get_history

    _store_clear_history()
    _store_add_message("user", "test")
    with _make_app().test_client() as client:
        resp = client.post("/api/chat/clear")
        assert resp.status_code == 200

    assert _store_get_history() == []


def test_chat_history_persists_messages() -> None:
    """GET /api/chat/history returns messages added via the store directly."""
    from rex.gui_app import _store_add_message, _store_clear_history

    _store_clear_history()
    _store_add_message("user", "hello")
    _store_add_message("assistant", "hi there")
    with _make_app().test_client() as client:
        resp = client.get("/api/chat/history")
        data = json.loads(resp.data)
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"


def test_chat_store_add_and_clear() -> None:
    """Unit test: inline store add_message / clear_history."""
    from rex.gui_app import _store_add_message, _store_clear_history, _store_get_history

    _store_clear_history()
    msg = _store_add_message("user", "hello", attachment_name="file.txt")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.attachment_name == "file.txt"

    history = _store_get_history()
    assert len(history) == 1
    assert history[0]["id"] == msg.id

    _store_clear_history()
    assert _store_get_history() == []
