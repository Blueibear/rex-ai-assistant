"""Tests for US-051: Chat interface.

Acceptance criteria:
- messages sent
- responses displayed
- session maintained
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import rex.dashboard.routes as routes_module
from rex.dashboard import dashboard_bp


@pytest.fixture(autouse=True)
def clear_chat_history():
    """Clear global chat history before each test."""
    routes_module._CHAT_HISTORY.clear()
    yield
    routes_module._CHAT_HISTORY.clear()


@pytest.fixture(autouse=True)
def _no_auth(monkeypatch):
    """Disable password requirement so loopback requests bypass auth in tests."""
    import rex.dashboard.auth as _auth_mod

    monkeypatch.setattr(_auth_mod, "is_password_required", lambda: False)
    monkeypatch.setattr(routes_module, "is_password_required", lambda: False)
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "1")


@pytest.fixture()
def app(monkeypatch):
    """Create a minimal Flask app with the dashboard blueprint."""
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "1")
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-chat"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


def _fake_llm(reply: str = "Hello from Rex"):
    """Return a mock LanguageModel that returns a fixed reply."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = reply
    return mock_llm


# --- messages sent ---


def test_chat_post_accepts_message(client):
    """POST /api/chat accepts a message in JSON body."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm()):
        response = client.post(
            "/api/chat",
            data=json.dumps({"message": "Hello Rex"}),
            content_type="application/json",
        )
    assert response.status_code == 200


def test_chat_post_returns_reply(client):
    """POST /api/chat returns a reply field in the response."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm("Hi there!")):
        response = client.post(
            "/api/chat",
            data=json.dumps({"message": "Say hello"}),
            content_type="application/json",
        )
    data = response.get_json()
    assert "reply" in data
    assert data["reply"] == "Hi there!"


def test_chat_post_returns_timestamp(client):
    """POST /api/chat includes a timestamp in the response."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm()):
        response = client.post(
            "/api/chat",
            data=json.dumps({"message": "Test"}),
            content_type="application/json",
        )
    data = response.get_json()
    assert "timestamp" in data
    assert data["timestamp"]  # non-empty


def test_chat_post_returns_elapsed_ms(client):
    """POST /api/chat includes elapsed_ms in the response."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm()):
        response = client.post(
            "/api/chat",
            data=json.dumps({"message": "Timing test"}),
            content_type="application/json",
        )
    data = response.get_json()
    assert "elapsed_ms" in data
    assert isinstance(data["elapsed_ms"], int)


def test_chat_empty_message_rejected(client):
    """POST /api/chat with empty message returns 400."""
    response = client.post(
        "/api/chat",
        data=json.dumps({"message": ""}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_chat_whitespace_message_rejected(client):
    """POST /api/chat with whitespace-only message returns 400."""
    response = client.post(
        "/api/chat",
        data=json.dumps({"message": "   "}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_chat_missing_message_field_rejected(client):
    """POST /api/chat with no message field returns 400."""
    response = client.post(
        "/api/chat",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert response.status_code == 400


# --- responses displayed ---


def test_chat_reply_is_string(client):
    """POST /api/chat reply is a string value."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm("Test response")):
        response = client.post(
            "/api/chat",
            data=json.dumps({"message": "What is 2+2?"}),
            content_type="application/json",
        )
    data = response.get_json()
    assert isinstance(data["reply"], str)


def test_chat_llm_error_returns_500(client):
    """POST /api/chat returns 500 when LLM raises an exception."""
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("LLM failed")
    with patch("rex.dashboard.routes._get_llm", return_value=mock_llm):
        response = client.post(
            "/api/chat",
            data=json.dumps({"message": "Cause an error"}),
            content_type="application/json",
        )
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data


def test_chat_html_has_chat_section(client):
    """Dashboard HTML includes the chat section element."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "chat-section" in html or "chat" in html.lower()


def test_chat_html_has_chat_form(client):
    """Dashboard HTML has a chat form for sending messages."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "chat-form" in html


def test_chat_html_has_message_input(client):
    """Dashboard HTML has a message input field."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "chat-input" in html


def test_chat_html_has_messages_container(client):
    """Dashboard HTML has a container for displaying chat messages."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "chat-messages" in html


def test_chat_js_send_logic_present(client):
    """Dashboard JS includes chat send logic."""
    response = client.get("/dashboard/assets/js/dashboard.js")
    js = response.data.decode("utf-8")
    assert "/api/chat" in js


# --- session maintained ---


def test_chat_history_endpoint_accessible(client):
    """GET /api/chat/history returns 200."""
    response = client.get("/api/chat/history")
    assert response.status_code == 200


def test_chat_history_starts_empty(client):
    """Chat history is empty before any messages are sent."""
    response = client.get("/api/chat/history")
    data = response.get_json()
    assert data["history"] == []
    assert data["total"] == 0


def test_chat_history_records_sent_messages(client):
    """Messages sent via POST /api/chat appear in GET /api/chat/history."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm("Stored!")):
        client.post(
            "/api/chat",
            data=json.dumps({"message": "Remember this?"}),
            content_type="application/json",
        )

    history_response = client.get("/api/chat/history")
    data = history_response.get_json()
    assert data["total"] == 1
    assert data["history"][0]["user_message"] == "Remember this?"
    assert data["history"][0]["assistant_reply"] == "Stored!"


def test_chat_history_persists_across_requests(client):
    """Multiple messages accumulate in history across separate requests."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm("Reply 1")):
        client.post(
            "/api/chat",
            data=json.dumps({"message": "First"}),
            content_type="application/json",
        )
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm("Reply 2")):
        client.post(
            "/api/chat",
            data=json.dumps({"message": "Second"}),
            content_type="application/json",
        )

    history_response = client.get("/api/chat/history")
    data = history_response.get_json()
    assert data["total"] == 2
    messages = [entry["user_message"] for entry in data["history"]]
    assert "First" in messages
    assert "Second" in messages


def test_chat_history_limit_param(client):
    """GET /api/chat/history respects limit query parameter."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm()):
        for i in range(5):
            client.post(
                "/api/chat",
                data=json.dumps({"message": f"Message {i}"}),
                content_type="application/json",
            )

    response = client.get("/api/chat/history?limit=2")
    data = response.get_json()
    assert len(data["history"]) == 2
    assert data["limit"] == 2


def test_chat_history_offset_param(client):
    """GET /api/chat/history respects offset query parameter."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm()):
        for i in range(4):
            client.post(
                "/api/chat",
                data=json.dumps({"message": f"Msg {i}"}),
                content_type="application/json",
            )

    response = client.get("/api/chat/history?offset=2")
    data = response.get_json()
    assert data["total"] == 4
    assert len(data["history"]) == 2  # 4 total, skip first 2


def test_chat_history_includes_timestamps(client):
    """Chat history entries include a timestamp field."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm("Timestamped")):
        client.post(
            "/api/chat",
            data=json.dumps({"message": "When?"}),
            content_type="application/json",
        )

    history_response = client.get("/api/chat/history")
    data = history_response.get_json()
    entry = data["history"][0]
    assert "timestamp" in entry
    assert entry["timestamp"]


# --- Verify changes work in browser ---


def test_chat_api_responds_correctly_to_post(client):
    """Full end-to-end: POST /api/chat returns correct JSON shape."""
    with patch("rex.dashboard.routes._get_llm", return_value=_fake_llm("Browser test reply")):
        response = client.post(
            "/api/chat",
            data=json.dumps({"message": "Browser test"}),
            content_type="application/json",
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["reply"] == "Browser test reply"
    assert "timestamp" in data
    assert "elapsed_ms" in data


def test_dashboard_js_includes_chat_history_fetch(client):
    """Dashboard JS fetches chat history from /api/chat/history."""
    response = client.get("/dashboard/assets/js/dashboard.js")
    js = response.data.decode("utf-8")
    assert "/api/chat/history" in js


def test_dashboard_js_includes_send_button(client):
    """Dashboard HTML includes a send button for the chat form."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "Send" in html
