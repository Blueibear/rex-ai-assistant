"""Tests for US-006: VoiceBridge.generate_reply() HTTP delegation.

Acceptance criteria:
  - delegates to self.agent.respond() (which uses HTTP when gateway is configured)
  - empty/whitespace transcripts return "" without calling respond()
  - on exception from respond(): logs error, returns spoken error message
  - new test: HTTP call is made when gateway is configured (via RexAgent.respond)
  - new test: graceful error message on HTTP failure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rex.openclaw.voice_bridge import VoiceBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(
    respond_return="Hello!", respond_side_effect=None
) -> tuple[VoiceBridge, MagicMock]:
    """Return (bridge, mock_agent) with agent.respond() pre-configured."""
    mock_agent = MagicMock()
    if respond_side_effect is not None:
        mock_agent.respond.side_effect = respond_side_effect
    else:
        mock_agent.respond.return_value = respond_return
    bridge = VoiceBridge(agent=mock_agent, user_key="test-user")
    return bridge, mock_agent


# ---------------------------------------------------------------------------
# Empty / whitespace transcript tests
# ---------------------------------------------------------------------------


class TestEmptyTranscript:
    def test_empty_string_returns_empty(self):
        bridge, mock_agent = _make_bridge()
        result = bridge.generate_reply("")
        assert result == ""
        mock_agent.respond.assert_not_called()

    def test_whitespace_only_returns_empty(self):
        bridge, mock_agent = _make_bridge()
        result = bridge.generate_reply("   \n\t  ")
        assert result == ""
        mock_agent.respond.assert_not_called()


# ---------------------------------------------------------------------------
# Successful delegation to RexAgent.respond()
# ---------------------------------------------------------------------------


class TestSuccessfulDelegation:
    def test_delegates_to_agent_respond(self):
        bridge, mock_agent = _make_bridge(respond_return="The time is 3pm")
        result = bridge.generate_reply("what time is it")
        assert result == "The time is 3pm"
        mock_agent.respond.assert_called_once_with("what time is it", user_key="test-user")

    def test_voice_mode_flag_accepted_not_forwarded(self):
        """voice_mode kwarg does not change the call to respond()."""
        bridge, mock_agent = _make_bridge(respond_return="ok")
        result = bridge.generate_reply("hello", voice_mode=True)
        assert result == "ok"
        mock_agent.respond.assert_called_once_with("hello", user_key="test-user")

    def test_extra_kwargs_ignored(self):
        bridge, mock_agent = _make_bridge(respond_return="ok")
        result = bridge.generate_reply("hello", voice_mode=False, extra_kwarg="ignored")
        assert result == "ok"


# ---------------------------------------------------------------------------
# HTTP delegation: verify RexAgent.respond() goes through OpenClaw HTTP
# when gateway is configured and flag is set
# ---------------------------------------------------------------------------


class TestHTTPDelegation:
    def test_http_call_made_when_gateway_configured(self):
        """When the OpenClaw gateway is configured and flag is True,
        RexAgent.respond() POSTs to /v1/chat/completions.
        VoiceBridge delegates to respond() — so patching the HTTP client
        at the agent layer verifies the call chain reaches HTTP.
        """
        from rex.config import AppConfig
        from rex.openclaw.agent import RexAgent
        from rex.openclaw.http_client import OpenClawClient

        config = AppConfig(
            llm_provider="openai",
            openai_api_key="test-key",
            openai_model="openclaw:main",
            use_openclaw_voice_backend=True,
            openclaw_gateway_url="http://127.0.0.1:18789",
        )

        mock_client = MagicMock(spec=OpenClawClient)
        mock_client.post.return_value = {"choices": [{"message": {"content": "It is noon."}}]}

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            agent = RexAgent(config=config)
            bridge = VoiceBridge(agent=agent, user_key="james")
            result = bridge.generate_reply("what time is it")

        assert result == "It is noon."
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "/v1/chat/completions"
        payload = call_args.kwargs["json"]
        assert "messages" in payload
        assert payload.get("user") == "james"

    def test_no_http_call_when_gateway_not_configured(self):
        """When gateway is not configured, VoiceBridge falls back to local LLM."""
        from rex.config import AppConfig
        from rex.openclaw.agent import RexAgent

        config = AppConfig(
            llm_provider="openai",
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
            use_openclaw_voice_backend=False,
            openclaw_gateway_url="",
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "local response"

        with (
            patch("rex.openclaw.agent.get_openclaw_client", return_value=None),
            patch("rex.openclaw.agent.LanguageModel", return_value=mock_llm),
        ):
            agent = RexAgent(config=config)
            bridge = VoiceBridge(agent=agent, user_key="james")
            result = bridge.generate_reply("hello")

        assert result == "local response"
        mock_llm.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Exception handling: graceful error message
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    def test_connection_error_returns_graceful_message(self):
        """OpenClawConnectionError from respond() returns the spoken error string."""
        from rex.openclaw.errors import OpenClawConnectionError

        bridge, mock_agent = _make_bridge(
            respond_side_effect=OpenClawConnectionError("http://127.0.0.1:18789", "refused")
        )
        result = bridge.generate_reply("hello")
        assert "trouble" in result.lower() or "server" in result.lower()

    def test_api_error_returns_graceful_message(self):
        """OpenClawAPIError from respond() returns the spoken error string."""
        from rex.openclaw.errors import OpenClawAPIError

        bridge, mock_agent = _make_bridge(
            respond_side_effect=OpenClawAPIError(500, "internal error")
        )
        result = bridge.generate_reply("hello")
        assert result == "I had trouble reaching the server. Try again."

    def test_generic_exception_returns_graceful_message(self):
        """Any unexpected exception from respond() returns the spoken error string."""
        bridge, mock_agent = _make_bridge(respond_side_effect=RuntimeError("unexpected crash"))
        result = bridge.generate_reply("hello")
        assert result == "I had trouble reaching the server. Try again."

    def test_exception_does_not_propagate(self):
        """Exceptions from respond() must not propagate out of generate_reply."""
        bridge, mock_agent = _make_bridge(respond_side_effect=Exception("boom"))
        # Should not raise
        result = bridge.generate_reply("hello")
        assert isinstance(result, str)
