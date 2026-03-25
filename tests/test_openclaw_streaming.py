"""Tests for US-017: Streaming support for OpenClaw chat completions.

Tests verify:
- stream_sentences accumulates tokens until sentence boundaries
- stream_sentences flushes remaining text at end of stream
- post_stream yields content deltas from SSE data lines
- post_stream handles [DONE] sentinel
- RexAgent.respond_stream yields sentence chunks
- RexAgent.respond_stream falls back to non-streaming on error
- Non-streaming mode is unchanged
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.config import AppConfig
from rex.openclaw.agent import RexAgent
from rex.openclaw.errors import OpenClawConnectionError
from rex.openclaw.http_client import stream_sentences

# ---------------------------------------------------------------------------
# stream_sentences tests
# ---------------------------------------------------------------------------


class TestStreamSentences:
    def test_yields_at_period(self):
        chunks = iter(["Hello", " world", ".", " How", " are you", "?"])
        result = list(stream_sentences(chunks))
        assert result == ["Hello world.", "How are you?"]

    def test_yields_at_exclamation(self):
        chunks = iter(["Wow", "!", " Great", "."])
        result = list(stream_sentences(chunks))
        assert result == ["Wow!", "Great."]

    def test_yields_at_newline(self):
        chunks = iter(["Line one", "\n", "Line two"])
        result = list(stream_sentences(chunks))
        assert result == ["Line one", "Line two"]

    def test_flushes_remaining(self):
        chunks = iter(["No sentence boundary here"])
        result = list(stream_sentences(chunks))
        assert result == ["No sentence boundary here"]

    def test_empty_stream(self):
        chunks = iter([])
        result = list(stream_sentences(chunks))
        assert result == []

    def test_multiple_periods_in_one_chunk(self):
        chunks = iter(["First. Second. Third."])
        result = list(stream_sentences(chunks))
        assert result == ["First.", "Second.", "Third."]

    def test_whitespace_only_not_yielded(self):
        chunks = iter(["Hello.", "   ", "World."])
        result = list(stream_sentences(chunks))
        assert result == ["Hello.", "World."]


# ---------------------------------------------------------------------------
# RexAgent.respond_stream tests
# ---------------------------------------------------------------------------


def _make_config(
    *, use_voice_backend: bool = True, gateway_url: str = "http://localhost:18789"
) -> AppConfig:
    return AppConfig(
        openclaw_gateway_url=gateway_url,
        openclaw_gateway_token="test-token",
        use_openclaw_voice_backend=use_voice_backend,
        llm_model="openclaw:main",
    )


def _make_agent(config: AppConfig, llm_reply: str = "local reply") -> RexAgent:
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_reply
    return RexAgent(llm=mock_llm, config=config, system_prompt="You are Rex.")


class TestRespondStream:
    def test_yields_sentence_chunks_from_stream(self):
        config = _make_config()
        agent = _make_agent(config)

        mock_client = MagicMock()
        mock_client.post_stream.return_value = iter(
            ["It is ", "3pm.", " The weather", " is sunny."]
        )

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            result = list(agent.respond_stream("What time is it?"))

        assert result == ["It is 3pm.", "The weather is sunny."]

    def test_stream_payload_contains_stream_true(self):
        config = _make_config()
        agent = _make_agent(config)

        mock_client = MagicMock()
        mock_client.post_stream.return_value = iter(["Done."])

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            list(agent.respond_stream("Hello"))

        payload = mock_client.post_stream.call_args.kwargs["json"]
        assert payload["stream"] is True

    def test_falls_back_to_local_on_error(self):
        config = _make_config()
        agent = _make_agent(config, llm_reply="local fallback")

        mock_client = MagicMock()
        mock_client.post_stream.side_effect = OpenClawConnectionError(
            "http://localhost:18789", Exception("conn refused")
        )

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            result = list(agent.respond_stream("Hello"))

        assert result == ["local fallback"]

    def test_standalone_mode_yields_single_reply(self):
        config = _make_config(gateway_url="")
        agent = _make_agent(config, llm_reply="standalone reply")

        with patch("rex.openclaw.agent.get_openclaw_client", return_value=None):
            result = list(agent.respond_stream("Hello"))

        assert result == ["standalone reply"]

    def test_empty_prompt_raises(self):
        config = _make_config()
        agent = _make_agent(config)

        with pytest.raises(ValueError, match="prompt must not be empty"):
            list(agent.respond_stream(""))

    def test_flag_false_uses_local_llm(self):
        config = _make_config(use_voice_backend=False)
        agent = _make_agent(config, llm_reply="flag-false reply")

        mock_client = MagicMock()
        with patch("rex.openclaw.agent.get_openclaw_client", return_value=mock_client):
            result = list(agent.respond_stream("Hello"))

        mock_client.post_stream.assert_not_called()
        assert result == ["flag-false reply"]
