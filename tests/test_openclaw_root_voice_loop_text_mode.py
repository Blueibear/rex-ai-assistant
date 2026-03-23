"""Tests for US-P6-008: root voice_loop.py _process_conversation with VoiceBridge (text mode).

Acceptance criteria:
  - When _assistant is a VoiceBridge, _process_conversation calls generate_reply(transcript, voice_mode=True)
  - generate_reply response is passed to _speak_response
  - Empty/whitespace response from VoiceBridge skips TTS
  - generate_reply exception is handled gracefully (logged, no TTS)
  - No audio hardware is touched during tests
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.config import AppConfig


def _make_config(**kwargs) -> AppConfig:
    return AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2", **kwargs)


# ---------------------------------------------------------------------------
# Factory — build an AsyncRexAssistant with all heavy deps stubbed out.
# The returned instance has _assistant already set to the provided mock.
# ---------------------------------------------------------------------------

def _make_ara(config: AppConfig, mock_bridge: MagicMock):
    """Return an AsyncRexAssistant whose _assistant is mock_bridge, no heavy imports."""
    mock_numpy = MagicMock()
    mock_numpy.zeros.return_value = MagicMock(size=1024)

    with (
        patch("voice_loop.load_config", return_value=config),
        patch("voice_loop.LanguageModel", return_value=MagicMock()),
        patch("voice_loop.load_wakeword_model", return_value=(MagicMock(), "hey rex")),
        patch("voice_loop.ensure_wake_acknowledgment_sound", return_value=None),
        patch("voice_loop.load_plugins", return_value={}),
        patch("rex.assistant.Assistant", return_value=MagicMock()),
        patch("voice_loop.load_audio_config", return_value={}),
        patch("voice_loop.resolve_audio_device", return_value=(None, "no device")),
        patch("voice_loop.WakeWordListener", return_value=MagicMock()),
        patch("voice_loop.load_users_map", return_value={}),
        patch("voice_loop.load_all_profiles", return_value={}),
        patch("voice_loop.resolve_user_key", return_value="james"),
        patch("voice_loop.extract_voice_reference", return_value=None),
    ):
        import importlib

        import voice_loop as vl

        importlib.reload(vl)
        ara = vl.AsyncRexAssistant(config=config)
        # Override _assistant with the test bridge
        ara._assistant = mock_bridge
        return ara, vl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessConversationWithVoiceBridge:
    """_process_conversation routes the transcript through VoiceBridge.generate_reply."""

    @pytest.mark.asyncio
    async def test_generate_reply_called_with_transcript(self):
        """VoiceBridge.generate_reply is called with the STT transcript."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config()
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="Hello from OpenClaw")

        ara, vl = _make_ara(cfg, mock_bridge)

        fake_audio = MagicMock()
        fake_audio.size = 512

        with (
            patch.object(ara, "_record_audio", return_value=fake_audio),
            patch.object(ara, "transcribe", new=AsyncMock(return_value="what time is it")),
            patch.object(ara, "_speak_response", return_value=None),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await ara._process_conversation()

        mock_bridge.generate_reply.assert_awaited_once_with("what time is it", voice_mode=True)

    @pytest.mark.asyncio
    async def test_generate_reply_response_passed_to_tts(self):
        """Response from VoiceBridge.generate_reply is handed to _speak_response."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config()
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="It is 3pm")

        ara, vl = _make_ara(cfg, mock_bridge)

        fake_audio = MagicMock()
        fake_audio.size = 512

        speak_calls = []

        with (
            patch.object(ara, "_record_audio", return_value=fake_audio),
            patch.object(ara, "transcribe", new=AsyncMock(return_value="what time is it")),
            patch.object(ara, "_speak_response", side_effect=lambda r: speak_calls.append(r)),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await ara._process_conversation()

        assert speak_calls == ["It is 3pm"]

    @pytest.mark.asyncio
    async def test_empty_response_skips_tts(self):
        """Empty string from VoiceBridge skips _speak_response."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config()
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="")

        ara, vl = _make_ara(cfg, mock_bridge)

        fake_audio = MagicMock()
        fake_audio.size = 512

        speak_mock = MagicMock()

        with (
            patch.object(ara, "_record_audio", return_value=fake_audio),
            patch.object(ara, "transcribe", new=AsyncMock(return_value="hello")),
            patch.object(ara, "_speak_response", speak_mock),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await ara._process_conversation()

        speak_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_response_skips_tts(self):
        """Whitespace-only response from VoiceBridge skips _speak_response."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config()
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="   \n  ")

        ara, vl = _make_ara(cfg, mock_bridge)

        fake_audio = MagicMock()
        fake_audio.size = 512

        speak_mock = MagicMock()

        with (
            patch.object(ara, "_record_audio", return_value=fake_audio),
            patch.object(ara, "transcribe", new=AsyncMock(return_value="hello")),
            patch.object(ara, "_speak_response", speak_mock),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await ara._process_conversation()

        speak_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_reply_exception_does_not_propagate(self):
        """If VoiceBridge.generate_reply raises, _process_conversation handles it gracefully."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config()
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        ara, vl = _make_ara(cfg, mock_bridge)

        fake_audio = MagicMock()
        fake_audio.size = 512

        speak_mock = MagicMock()

        with (
            patch.object(ara, "_record_audio", return_value=fake_audio),
            patch.object(ara, "transcribe", new=AsyncMock(return_value="hello")),
            patch.object(ara, "_speak_response", speak_mock),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            # Should not raise
            await ara._process_conversation()

        speak_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_generate_reply(self):
        """Empty STT result skips VoiceBridge entirely."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config()
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="response")

        ara, vl = _make_ara(cfg, mock_bridge)

        fake_audio = MagicMock()
        fake_audio.size = 512

        with (
            patch.object(ara, "_record_audio", return_value=fake_audio),
            patch.object(ara, "transcribe", new=AsyncMock(return_value="")),
            patch.object(ara, "_speak_response", return_value=None),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await ara._process_conversation()

        mock_bridge.generate_reply.assert_not_awaited()
