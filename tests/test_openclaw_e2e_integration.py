"""End-to-end integration tests for the Rex ↔ OpenClaw HTTP integration — US-018.

These tests simulate the full Rex → OpenClaw flow using ``unittest.mock`` to
stand in for the HTTP gateway.  No real OpenClaw instance is required.

Test coverage:
- Test 1: Text mode e2e — RexAgent.respond() → mock OpenClaw → response
- Test 2: Tool call e2e — ToolBridge dispatches via /tools/invoke + RexAgent
          re-calls /v1/chat/completions with tool result → final response
- Test 3: Voice bridge e2e — VoiceBridge.generate_reply() → mock OpenClaw
- Test 4: Tool server e2e — Flask test client → POST /rex/tools/time_now
- Test 5: Fallback — mock OpenClaw returns 500 → Rex falls back to local LLM
- Test 6: Standalone — no gateway URL → all paths use local backends, zero HTTP

All tests run in CI without a live OpenClaw instance.
Tests are marked ``integration`` and also run under the standard ``pytest -q`` suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from rex.config import AppConfig
from rex.openclaw.agent import RexAgent
from rex.openclaw.errors import OpenClawAPIError
from rex.openclaw.policy_adapter import PolicyAdapter
from rex.openclaw.tool_bridge import ToolBridge
from rex.openclaw.tool_server import ToolServer, _add_health_routes
from rex.openclaw.voice_bridge import VoiceBridge

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GATEWAY = "http://127.0.0.1:18789"
_API_KEY = "e2e-test-key-xyz"


def _config(
    *,
    gateway_url: str = _GATEWAY,
    use_voice_backend: bool = True,
    use_openclaw_tools: bool = True,
) -> AppConfig:
    return AppConfig(
        openclaw_gateway_url=gateway_url,
        openclaw_gateway_token="test-token",
        use_openclaw_voice_backend=use_voice_backend,
        use_openclaw_tools=use_openclaw_tools,
        llm_model="openclaw:main",
    )


def _standalone_config() -> AppConfig:
    """Config with no gateway URL — forces all paths to local backends."""
    return AppConfig(
        openclaw_gateway_url="",
        use_openclaw_voice_backend=False,
        use_openclaw_tools=False,
        llm_model="gpt2",
    )


def _mock_llm(reply: str = "local reply") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = reply
    return llm


def _chat_response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _mock_client(post_return: dict | None = None, post_side_effect=None) -> MagicMock:
    client = MagicMock()
    if post_side_effect is not None:
        client.post.side_effect = post_side_effect
    else:
        client.post.return_value = post_return or _chat_response("OpenClaw says hello")
    return client


# ---------------------------------------------------------------------------
# Test 1: Text mode end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTextModeE2E:
    """RexAgent.respond() routes through mock OpenClaw and returns its response."""

    def test_respond_uses_openclaw_response(self):
        """User prompt → RexAgent.respond() → /v1/chat/completions → returned."""
        cfg = _config()
        agent = RexAgent(llm=_mock_llm(), config=cfg, system_prompt="You are Rex.")
        mock_client = _mock_client(_chat_response("The time is 3 pm."))

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            reply = agent.respond("What time is it?")

        assert reply == "The time is 3 pm."
        mock_client.post.assert_called_once()
        call = mock_client.post.call_args
        assert call.args[0] == "/v1/chat/completions"
        payload = call.kwargs["json"]
        assert payload["model"] == "openclaw:main"
        assert any(m["content"] == "What time is it?" for m in payload["messages"])

    def test_respond_includes_user_field(self):
        """The ``user`` field is present in every OpenClaw chat completion request."""
        cfg = _config()
        agent = RexAgent(llm=_mock_llm(), config=cfg, system_prompt="You are Rex.")
        mock_client = _mock_client()

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            agent.respond("Hello", user_key="alice")

        payload = mock_client.post.call_args.kwargs["json"]
        assert "user" in payload
        assert payload["user"] == "alice"

    def test_respond_includes_system_message(self):
        """System prompt is the first message in the payload."""
        cfg = _config()
        agent = RexAgent(llm=_mock_llm(), config=cfg, system_prompt="You are Rex.")
        mock_client = _mock_client()

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            agent.respond("Hi")

        messages = mock_client.post.call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert "Rex" in messages[0]["content"]


# ---------------------------------------------------------------------------
# Test 2: Tool call end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestToolCallE2E:
    """ToolBridge dispatches via /tools/invoke; RexAgent uses the result."""

    def test_tool_bridge_dispatches_via_openclaw_then_agent_uses_result(self):
        """Tool call is dispatched via HTTP; the result is then included in a chat reply."""
        cfg = _config()

        # ToolBridge: mock OpenClaw returns a successful tool result.
        tool_result = {"status": "success", "result": {"time": "15:00", "tz": "UTC"}}
        tool_client = _mock_client(post_return=tool_result)
        bridge = ToolBridge(config=cfg)

        with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=tool_client):
            result = bridge.execute_tool(
                {"tool": "time_now", "args": {"location": "London"}},
                {"session_key": "alice"},
            )

        assert result["status"] == "success"
        assert result["result"]["time"] == "15:00"
        tool_client.post.assert_called_once()
        payload = tool_client.post.call_args.kwargs["json"]
        assert payload["tool"] == "time_now"
        assert payload["sessionKey"] == "alice"

        # Now RexAgent incorporates the tool result in a follow-up chat completion.
        agent = RexAgent(llm=_mock_llm(), config=cfg, system_prompt="You are Rex.")
        follow_up_msg = "Based on the tool result, it is 15:00 UTC."
        chat_client = _mock_client(_chat_response(follow_up_msg))

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=chat_client):
            # Simulate re-calling OpenClaw with the tool result embedded in the prompt.
            tool_result_text = f"Tool returned: {result['result']}"
            reply = agent.respond(tool_result_text)

        assert reply == follow_up_msg

    def test_tool_bridge_404_falls_back_to_local(self):
        """404 from /tools/invoke triggers local execution fallback."""
        cfg = _config()
        api_error = OpenClawAPIError(404, "not found")
        tool_client = _mock_client(post_side_effect=api_error)
        bridge = ToolBridge(config=cfg)

        local_result = {"status": "ok", "result": "12:00 UTC"}
        with (
            patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=tool_client),
            patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=local_result
            ) as mock_local,
        ):
            result = bridge.execute_tool(
                {"tool": "time_now", "args": {}},
                {},
            )

        assert result == local_result
        mock_local.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3: Voice bridge end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVoiceBridgeE2E:
    """VoiceBridge.generate_reply() routes through RexAgent → mock OpenClaw."""

    def test_generate_reply_returns_openclaw_response(self):
        """Transcript → VoiceBridge → RexAgent → /v1/chat/completions → spoken string."""
        cfg = _config()
        agent = RexAgent(llm=_mock_llm(), config=cfg, system_prompt="You are Rex.")
        bridge = VoiceBridge(agent=agent, user_key="voice-user")
        mock_client = _mock_client(_chat_response("I am fine, thank you."))

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            reply = bridge.generate_reply("How are you?")

        assert reply == "I am fine, thank you."
        mock_client.post.assert_called_once()

    def test_generate_reply_empty_transcript_skips_http(self):
        """Empty transcript returns empty string with no HTTP call."""
        cfg = _config()
        agent = RexAgent(llm=_mock_llm(), config=cfg, system_prompt="You are Rex.")
        bridge = VoiceBridge(agent=agent)
        mock_client = _mock_client()

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            result = bridge.generate_reply("   ")

        assert result == ""
        mock_client.post.assert_not_called()

    def test_generate_reply_voice_mode_flag_accepted(self):
        """voice_mode=True is accepted without error."""
        cfg = _config()
        agent = RexAgent(llm=_mock_llm(), config=cfg, system_prompt="You are Rex.")
        bridge = VoiceBridge(agent=agent)
        mock_client = _mock_client(_chat_response("Roger that."))

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            reply = bridge.generate_reply("Turn off the lights.", voice_mode=True)

        assert reply == "Roger that."


# ---------------------------------------------------------------------------
# Test 4: Tool server end-to-end (Flask test client)
# ---------------------------------------------------------------------------


def _make_tool_server_app(monkeypatch) -> Flask:
    """Return a Flask test app with real time_now handler and disabled rate limiting."""
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")

    from rex.openclaw.tools.time_tool import time_now

    tools = {"time_now": time_now}
    policy = MagicMock(spec=PolicyAdapter)
    policy.guard.return_value = None

    app = Flask(__name__)
    app.config["TESTING"] = True
    server = ToolServer(policy=policy, tools=tools)
    server.register_all(app)
    _add_health_routes(app, server)
    return app


@pytest.mark.integration
class TestToolServerE2E:
    """POST /rex/tools/time_now via Flask test client — no real TCP socket needed."""

    def test_time_now_returns_200(self, monkeypatch):
        """POST /rex/tools/time_now → 200 with a dict result."""
        app = _make_tool_server_app(monkeypatch)
        headers = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}

        with app.test_client() as c:
            resp = c.post("/rex/tools/time_now", headers=headers, json={})

        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_time_now_requires_auth(self, monkeypatch):
        """POST /rex/tools/time_now without auth → 401."""
        app = _make_tool_server_app(monkeypatch)

        with app.test_client() as c:
            resp = c.post(
                "/rex/tools/time_now",
                headers={"Content-Type": "application/json"},
                json={},
            )

        assert resp.status_code == 401

    def test_health_live_endpoint(self, monkeypatch):
        """GET /health/live → 200 ok."""
        app = _make_tool_server_app(monkeypatch)

        with app.test_client() as c:
            resp = c.get("/health/live")

        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 5: Fallback — 500 from OpenClaw → local LLM
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFallbackE2E:
    """When OpenClaw returns 5xx, Rex falls back to local LLM silently."""

    def test_respond_falls_back_on_api_error(self):
        """HTTP 500 from /v1/chat/completions → local LLM used, response returned."""
        cfg = _config()
        local_llm = _mock_llm("local fallback answer")
        agent = RexAgent(llm=local_llm, config=cfg, system_prompt="You are Rex.")
        error_client = _mock_client(post_side_effect=OpenClawAPIError(500, "internal error"))

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=error_client):
            reply = agent.respond("What is the weather?")

        assert reply == "local fallback answer"
        local_llm.generate.assert_called_once()

    def test_respond_falls_back_on_connection_error(self):
        """Connection failure → local LLM used, response returned."""
        from rex.openclaw.errors import OpenClawConnectionError

        cfg = _config()
        local_llm = _mock_llm("fallback on connection error")
        agent = RexAgent(llm=local_llm, config=cfg, system_prompt="You are Rex.")
        error_client = _mock_client(
            post_side_effect=OpenClawConnectionError(_GATEWAY, ConnectionRefusedError())
        )

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=error_client):
            reply = agent.respond("Hello?")

        assert reply == "fallback on connection error"
        local_llm.generate.assert_called_once()

    def test_voice_bridge_returns_error_message_on_failure(self):
        """VoiceBridge catches RexAgent exceptions and returns a spoken error message."""
        # Make RexAgent.respond() raise an unexpected error.
        mock_agent = MagicMock(spec=RexAgent)
        mock_agent.respond.side_effect = RuntimeError("unexpected!")
        bridge = VoiceBridge(agent=mock_agent)

        reply = bridge.generate_reply("Play some music.")

        assert "trouble" in reply.lower()
        assert isinstance(reply, str)


# ---------------------------------------------------------------------------
# Test 6: Standalone mode — no gateway URL → zero HTTP calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStandaloneModeE2E:
    """All paths use local backends when no gateway URL is configured."""

    def test_respond_uses_local_llm_when_no_gateway(self):
        """No openclaw_gateway_url → get_openclaw_client() returns None → local LLM."""
        cfg = _standalone_config()
        local_llm = _mock_llm("standalone answer")
        agent = RexAgent(llm=local_llm, config=cfg, system_prompt="You are Rex.")

        # get_openclaw_client returns None when gateway_url is empty.
        with patch("rex.openclaw.agent.get_openclaw_client", return_value=None) as mock_get:
            reply = agent.respond("Hello?")

        assert reply == "standalone answer"
        local_llm.generate.assert_called_once()
        # OpenClaw client was checked but returned None — no HTTP calls possible.
        mock_get.assert_called_once()

    def test_tool_bridge_uses_local_when_no_gateway(self):
        """No gateway URL → ToolBridge.execute_tool() runs locally, no HTTP call."""
        cfg = _standalone_config()
        bridge = ToolBridge(config=cfg)
        local_result = {"status": "ok", "result": "12:00"}

        with (
            patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=None),
            patch(
                "rex.openclaw.tool_bridge._execute_tool", return_value=local_result
            ) as mock_local,
        ):
            result = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

        assert result == local_result
        mock_local.assert_called_once()

    def test_voice_bridge_uses_local_when_no_gateway(self):
        """VoiceBridge → RexAgent → local LLM when gateway URL is empty."""
        cfg = _standalone_config()
        local_llm = _mock_llm("local voice reply")
        agent = RexAgent(llm=local_llm, config=cfg, system_prompt="You are Rex.")
        bridge = VoiceBridge(agent=agent)

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=None):
            reply = bridge.generate_reply("Set a timer for five minutes.")

        assert reply == "local voice reply"
        local_llm.generate.assert_called_once()
