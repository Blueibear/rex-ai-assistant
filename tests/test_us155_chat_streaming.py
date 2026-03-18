"""Tests for US-155: Chat message streaming display.

Acceptance criteria:
- backend streams the response token-by-token or chunk-by-chunk via SSE or WebSocket
- chat UI appends tokens to the current message bubble as they arrive
- the loading indicator is replaced by the streaming message (not shown simultaneously)
- streaming works correctly for responses of at least 500 tokens
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


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _non_local_env():
    return {"REMOTE_ADDR": "10.0.0.1"}


@pytest.fixture()
def app():
    from rex.dashboard import dashboard_bp

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-155"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-155")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-155"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200, f"Login failed: {resp.get_json()}"
    return resp.get_json()["token"]


# ---------------------------------------------------------------------------
# Backend route registration
# ---------------------------------------------------------------------------


class TestStreamRouteRegistered:
    def test_stream_route_exists(self, app):
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/chat/stream" in rules, "/api/chat/stream must be registered"

    def test_stream_route_accepts_post(self, app):
        methods = {
            rule.rule: rule.methods
            for rule in app.url_map.iter_rules()
            if rule.rule == "/api/chat/stream"
        }
        assert "POST" in methods.get("/api/chat/stream", set()), "/api/chat/stream must accept POST"

    def test_stream_route_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "secret155")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.post(
            "/api/chat/stream",
            data=json.dumps({"message": "hi"}),
            content_type="application/json",
            environ_base=_non_local_env(),
        )
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Backend: SSE streaming behaviour
# ---------------------------------------------------------------------------


def _collect_sse(response_data: bytes) -> list[dict]:
    """Parse raw SSE bytes into list of {event, data} dicts."""
    events = []
    current_event = {}
    for line in response_data.decode("utf-8").splitlines():
        if line.startswith("event: "):
            current_event["event"] = line[7:].strip()
        elif line.startswith("data: "):
            current_event["data"] = json.loads(line[6:])
        elif line == "" and current_event:
            events.append(current_event)
            current_event = {}
    if current_event:
        events.append(current_event)
    return events


class TestStreamBackendBehaviour:
    def _stream_chat(self, client, auth_token, monkeypatch, message="Hello"):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-155")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        with patch("rex.dashboard.routes._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Hello from Rex."
            # No openai client on strategy — triggers fallback word-by-word
            mock_llm.strategy = None
            mock_llm.generation = MagicMock(temperature=0.7, max_new_tokens=512, top_p=0.9)
            mock_llm_fn.return_value = mock_llm

            resp = client.post(
                "/api/chat/stream",
                data=json.dumps({"message": message}),
                content_type="application/json",
                headers={"X-Auth-Token": auth_token},
                environ_base=_non_local_env(),
            )
            return resp, _collect_sse(resp.data)

    def test_stream_returns_event_stream_content_type(self, client, auth_token, monkeypatch):
        resp, _ = self._stream_chat(client, auth_token, monkeypatch)
        assert "text/event-stream" in resp.content_type

    def test_stream_emits_token_events(self, client, auth_token, monkeypatch):
        _, events = self._stream_chat(client, auth_token, monkeypatch)
        token_events = [e for e in events if e.get("event") == "token"]
        assert len(token_events) > 0, "At least one token event must be emitted"

    def test_stream_token_events_have_token_field(self, client, auth_token, monkeypatch):
        _, events = self._stream_chat(client, auth_token, monkeypatch)
        for e in events:
            if e.get("event") == "token":
                assert "token" in e["data"], "token event data must contain 'token' key"

    def test_stream_emits_done_event(self, client, auth_token, monkeypatch):
        _, events = self._stream_chat(client, auth_token, monkeypatch)
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1, "Exactly one done event must be emitted"

    def test_stream_done_event_has_timestamp(self, client, auth_token, monkeypatch):
        _, events = self._stream_chat(client, auth_token, monkeypatch)
        done = next(e for e in events if e.get("event") == "done")
        assert "timestamp" in done["data"]

    def test_stream_done_event_has_elapsed_ms(self, client, auth_token, monkeypatch):
        _, events = self._stream_chat(client, auth_token, monkeypatch)
        done = next(e for e in events if e.get("event") == "done")
        assert "elapsed_ms" in done["data"]
        assert isinstance(done["data"]["elapsed_ms"], int)

    def test_stream_tokens_concatenate_to_full_reply(self, client, auth_token, monkeypatch):
        _, events = self._stream_chat(client, auth_token, monkeypatch, "Hello")
        tokens = [e["data"]["token"] for e in events if e.get("event") == "token"]
        full = "".join(tokens)
        assert full == "Hello from Rex."

    def test_stream_done_is_last_event(self, client, auth_token, monkeypatch):
        _, events = self._stream_chat(client, auth_token, monkeypatch)
        assert events[-1]["event"] == "done", "done event must be the last event"

    def test_stream_no_cache_headers(self, client, auth_token, monkeypatch):
        resp, _ = self._stream_chat(client, auth_token, monkeypatch)
        assert resp.headers.get("Cache-Control") == "no-cache"

    def test_stream_rejects_empty_message(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-155")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.post(
            "/api/chat/stream",
            data=json.dumps({"message": "   "}),
            content_type="application/json",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        assert resp.status_code in (400, 422)

    def test_stream_500_tokens(self, client, auth_token, monkeypatch):
        """Streaming must work for at least 500 tokens (words)."""
        long_reply = " ".join(f"word{i}" for i in range(500))
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-155")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        with patch("rex.dashboard.routes._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = long_reply
            mock_llm.strategy = None
            mock_llm.generation = MagicMock(temperature=0.7, max_new_tokens=2048, top_p=0.9)
            mock_llm_fn.return_value = mock_llm

            resp = client.post(
                "/api/chat/stream",
                data=json.dumps({"message": "Tell me a long story"}),
                content_type="application/json",
                headers={"X-Auth-Token": auth_token},
                environ_base=_non_local_env(),
            )
            events = _collect_sse(resp.data)
            token_events = [e for e in events if e.get("event") == "token"]
            concatenated = "".join(e["data"]["token"] for e in token_events)
            assert len(token_events) >= 500, "Must emit at least 500 token events"
            assert concatenated == long_reply


# ---------------------------------------------------------------------------
# Frontend JS: streaming implementation
# ---------------------------------------------------------------------------


class TestFrontendStreamingJS:
    def test_fetch_stream_endpoint_called(self):
        js = _js()
        assert "/api/chat/stream" in js, "JS must call /api/chat/stream"

    def test_js_uses_fetch_not_api_helper_for_stream(self):
        js = _js()
        # The streaming path must use fetch() directly (EventSource or fetch with reader)
        assert (
            "fetch(" in js or "EventSource" in js
        ), "JS must use fetch() or EventSource for streaming"

    def test_js_reads_response_body(self):
        js = _js()
        assert (
            "getReader" in js or "EventSource" in js
        ), "JS must use ReadableStream reader or EventSource"

    def test_js_token_event_handling(self):
        js = _js()
        assert "'token'" in js or '"token"' in js, "JS must handle 'token' SSE events"

    def test_js_done_event_handling(self):
        js = _js()
        assert "'done'" in js or '"done"' in js, "JS must handle 'done' SSE events"

    def test_js_thinking_indicator_removed_on_first_token(self):
        js = _js()
        # The thinking indicator must be removed when streaming starts
        assert "thinking-indicator" in js, "thinking-indicator must be referenced in JS"
        # After first token, thinking indicator is replaced
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 4000]
        assert (
            "thinking" in fn_body and "remove()" in fn_body
        ), "thinking indicator must be removed when streaming starts"

    def test_js_streaming_bubble_appended(self):
        js = _js()
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 4000]
        # A new bubble is created and appended on first token
        assert (
            "createElement" in fn_body or "innerHTML +=" in fn_body
        ), "JS must create an assistant bubble for streaming"

    def test_js_tokens_accumulated(self):
        js = _js()
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 4000]
        assert (
            "accumulatedReply" in fn_body or "accumulated" in fn_body.lower()
        ), "JS must accumulate tokens into a reply string"

    def test_js_scrolls_on_token(self):
        js = _js()
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 4000]
        assert "scrollTop" in fn_body, "JS must scroll container as tokens arrive"

    def test_js_error_event_handling(self):
        js = _js()
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 4000]
        assert "'error'" in fn_body or '"error"' in fn_body, "JS must handle 'error' SSE events"

    def test_js_history_updated_after_stream(self):
        js = _js()
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 4000]
        assert (
            "chatHistory.push" in fn_body
        ), "JS must push completed reply to chatHistory after streaming"

    def test_js_input_re_enabled_in_finally(self):
        js = _js()
        fn_start = js.index("handleChatSubmit")
        fn_body = js[fn_start : fn_start + 6000]
        assert (
            "finally" in fn_body and "disabled = false" in fn_body
        ), "input must be re-enabled in finally block"


# ---------------------------------------------------------------------------
# Typecheck
# ---------------------------------------------------------------------------


class TestTypecheck:
    def test_mypy_routes(self):
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "rex/dashboard/routes.py", "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        errors = [ln for ln in result.stdout.splitlines() if "error:" in ln and "routes.py" in ln]
        assert errors == [], "mypy errors in routes.py:\n" + "\n".join(errors)
