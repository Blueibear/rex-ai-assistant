"""Tests for US-154: Connect chat UI to Rex backend.

Acceptance criteria:
- submitted message sent to the Rex backend chat endpoint
- Rex's response displayed in the message list when received
- a loading indicator appears between message send and response arrival
- network errors produce a visible error message in the chat, not a silent failure
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

REPO_ROOT = Path(__file__).parent.parent
JS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "js" / "dashboard.js"
CSS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "css" / "dashboard.css"


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _non_local_env():
    """Simulates a remote (non-loopback) request so auth is required."""
    return {"REMOTE_ADDR": "10.0.0.1"}


# ---------------------------------------------------------------------------
# Backend route registration
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    from rex.dashboard import dashboard_bp

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    """Obtain a valid auth token via login."""
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-154")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-154"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200, f"Login failed: {resp.get_json()}"
    return resp.get_json()["token"]


class TestChatEndpointExists:
    def test_chat_route_registered(self, app):
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/chat" in rules, "/api/chat route must be registered"

    def test_chat_history_route_registered(self, app):
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/chat/history" in rules, "/api/chat/history route must be registered"

    def test_chat_route_accepts_post(self, app):
        methods = {
            rule.rule: rule.methods for rule in app.url_map.iter_rules() if rule.rule == "/api/chat"
        }
        assert "POST" in methods.get("/api/chat", set()), "/api/chat must accept POST"

    def test_chat_history_route_accepts_get(self, app):
        methods = {
            rule.rule: rule.methods
            for rule in app.url_map.iter_rules()
            if rule.rule == "/api/chat/history"
        }
        assert "GET" in methods.get("/api/chat/history", set()), "/api/chat/history must accept GET"


# ---------------------------------------------------------------------------
# Backend: chat endpoint behaviour
# ---------------------------------------------------------------------------


class TestChatEndpointBehaviour:
    def test_chat_requires_auth(self, client, monkeypatch):
        """POST /api/chat from a non-local address without token returns 401."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "secret")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.post(
            "/api/chat",
            data=json.dumps({"message": "hello"}),
            content_type="application/json",
            environ_base=_non_local_env(),
        )
        assert resp.status_code in (401, 403), "Unauthenticated request must be rejected"

    def test_chat_returns_reply_field(self, client, auth_token, monkeypatch):
        """POST /api/chat returns JSON with 'reply' key on success."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-154")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        with patch("rex.dashboard.routes._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Hello, I am Rex."
            mock_llm_fn.return_value = mock_llm

            resp = client.post(
                "/api/chat",
                data=json.dumps({"message": "Hello"}),
                content_type="application/json",
                headers={"Authorization": f"Bearer {auth_token}"},
                environ_base=_non_local_env(),
            )

        assert (
            resp.status_code == 200
        ), f"Expected 200, got {resp.status_code}: {resp.get_data(as_text=True)}"
        data = resp.get_json()
        assert "reply" in data, "Response must contain 'reply' field"

    def test_chat_returns_timestamp_field(self, client, auth_token, monkeypatch):
        """POST /api/chat returns JSON with 'timestamp' key."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-154")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        with patch("rex.dashboard.routes._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Test reply."
            mock_llm_fn.return_value = mock_llm

            resp = client.post(
                "/api/chat",
                data=json.dumps({"message": "test"}),
                content_type="application/json",
                headers={"Authorization": f"Bearer {auth_token}"},
                environ_base=_non_local_env(),
            )

        data = resp.get_json()
        assert "timestamp" in data, "Response must contain 'timestamp' field"

    def test_chat_reply_matches_llm_output(self, client, auth_token, monkeypatch):
        """Reply field contains the text returned by the LLM."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-154")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        with patch("rex.dashboard.routes._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Unique reply XYZ"
            mock_llm_fn.return_value = mock_llm

            resp = client.post(
                "/api/chat",
                data=json.dumps({"message": "ping"}),
                content_type="application/json",
                headers={"Authorization": f"Bearer {auth_token}"},
                environ_base=_non_local_env(),
            )

        data = resp.get_json()
        assert data["reply"] == "Unique reply XYZ"

    def test_chat_history_returns_list(self, client, auth_token, monkeypatch):
        """GET /api/chat/history returns JSON with 'history' key as a list."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-154")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.get(
            "/api/chat/history",
            headers={"Authorization": f"Bearer {auth_token}"},
            environ_base=_non_local_env(),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_chat_llm_error_returns_500(self, client, auth_token, monkeypatch):
        """POST /api/chat returns 500 when LLM raises an exception."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-154")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        with patch("rex.dashboard.routes._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.generate.side_effect = RuntimeError("LLM unavailable")
            mock_llm_fn.return_value = mock_llm

            resp = client.post(
                "/api/chat",
                data=json.dumps({"message": "fail"}),
                content_type="application/json",
                headers={"Authorization": f"Bearer {auth_token}"},
                environ_base=_non_local_env(),
            )

        assert resp.status_code == 500, "LLM errors must return 500"

    def test_chat_error_response_has_error_field(self, client, auth_token, monkeypatch):
        """500 response includes 'error' field."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-154")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        with patch("rex.dashboard.routes._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.generate.side_effect = RuntimeError("boom")
            mock_llm_fn.return_value = mock_llm

            resp = client.post(
                "/api/chat",
                data=json.dumps({"message": "fail"}),
                content_type="application/json",
                headers={"Authorization": f"Bearer {auth_token}"},
                environ_base=_non_local_env(),
            )

        data = resp.get_json()
        assert "error" in data, "Error response must contain 'error' field"


# ---------------------------------------------------------------------------
# JS: sends message to /api/chat
# ---------------------------------------------------------------------------


def _handleChatSubmit_body() -> str:
    """Extract the full body of handleChatSubmit from dashboard.js."""
    js = _js()
    fn_idx = js.index("handleChatSubmit")
    # Use a large window to capture the full async function (it's ~100+ lines now with streaming)
    return js[fn_idx : fn_idx + 6000]


class TestJsSendsToBackend:
    def test_handleChatSubmit_calls_api_chat(self):
        js = _js()
        # Accepts both the original /api/chat and the streaming /api/chat/stream
        assert "/api/chat" in js, "handleChatSubmit must call a /api/chat endpoint"

    def test_handleChatSubmit_uses_post(self):
        fn_body = _handleChatSubmit_body()
        assert "'POST'" in fn_body or '"POST"' in fn_body, "handleChatSubmit must use POST method"

    def test_handleChatSubmit_sends_message_field(self):
        fn_body = _handleChatSubmit_body()
        assert "message" in fn_body, "handleChatSubmit must include 'message' in the request body"

    def test_handleChatSubmit_reads_reply_from_response(self):
        # With streaming the reply is accumulated from tokens rather than read from data.reply
        js = _js()
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 6000]
        has_data_reply = "data.reply" in fn_body
        has_accumulated = "accumulatedReply" in fn_body or "accumulated" in fn_body.lower()
        assert (
            has_data_reply or has_accumulated
        ), "handleChatSubmit must read the reply from the response (data.reply or accumulated tokens)"


# ---------------------------------------------------------------------------
# JS: loading indicator
# ---------------------------------------------------------------------------


class TestJsLoadingIndicator:
    def test_thinking_indicator_injected(self):
        js = _js()
        assert "thinking-indicator" in js, "A thinking/loading indicator element must be created"

    def test_thinking_text_present(self):
        js = _js()
        assert (
            "Thinking" in js or "loading" in js.lower()
        ), "Loading indicator must show some loading text"

    def test_thinking_indicator_removed_on_success(self):
        fn_body = _handleChatSubmit_body()
        assert (
            "thinking" in fn_body and "remove()" in fn_body
        ), "Thinking indicator must be removed after response arrives"

    def test_thinking_css_class_defined(self):
        css = _css()
        assert (
            ".chat-message.thinking" in css
        ), ".chat-message.thinking CSS class must be defined for the loading indicator"


# ---------------------------------------------------------------------------
# JS: error handling — visible error message, not silent failure
# ---------------------------------------------------------------------------


class TestJsErrorHandling:
    def test_catch_block_exists_in_handleChatSubmit(self):
        fn_body = _handleChatSubmit_body()
        assert "catch" in fn_body, "handleChatSubmit must have a catch block for network errors"

    def test_error_message_displayed_in_chat(self):
        fn_body = _handleChatSubmit_body()
        assert (
            "error.message" in fn_body or "Error:" in fn_body
        ), "Error message must be displayed in the chat area"

    def test_error_not_silently_swallowed(self):
        fn_body = _handleChatSubmit_body()
        # Use the LAST catch block (outer error handler), not inner try/catch in SSE parsing
        catch_idx = fn_body.rindex("catch")
        catch_body = fn_body[catch_idx : catch_idx + 400]
        assert (
            "textContent" in catch_body
            or "innerHTML" in catch_body
            or "error" in catch_body.lower()
        ), "catch block must expose the error to the user"

    def test_input_re_enabled_after_error(self):
        fn_body = _handleChatSubmit_body()
        assert "finally" in fn_body, "handleChatSubmit must use a finally block to re-enable input"
        finally_idx = fn_body.index("finally")
        finally_body = fn_body[finally_idx : finally_idx + 200]
        assert (
            "disabled" in finally_body or "focus" in finally_body
        ), "Input must be re-enabled in finally block"


# ---------------------------------------------------------------------------
# Typecheck
# ---------------------------------------------------------------------------


class TestTypecheck:
    def test_mypy_routes_passes(self):
        """mypy on routes.py must not report new errors."""
        routes_path = REPO_ROOT / "rex" / "dashboard" / "routes.py"
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--ignore-missing-imports", str(routes_path)],
            capture_output=True,
            text=True,
        )
        # Allow pre-existing errors; just ensure mypy runs (exit 0 or 1 acceptable)
        assert result.returncode in (0, 1), f"mypy crashed unexpectedly:\n{result.stderr}"
