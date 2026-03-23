"""Tests for US-P6-012: rex/voice_loop_optimized.py VoiceLoop routes through VoiceBridge (text mode).

Acceptance criteria:
  - When _assistant is a VoiceBridge, run() calls generate_reply(transcript)
  - Response is passed to _speak
  - Empty transcript skips generate_reply
  - generate_reply exception is handled gracefully (no TTS)
  - No audio hardware is touched during tests
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.openclaw.voice_bridge import VoiceBridge


async def _one_wake_event():
    """Async generator that yields a single wake event then stops."""
    yield MagicMock()


def _make_vl(mock_bridge: MagicMock, transcribe_result: str = "hello rex") -> tuple:
    """Build an optimized VoiceLoop with mock VoiceBridge and injectable transcribe result."""
    from rex.voice_loop_optimized import VoiceLoop

    mock_wake_listener = MagicMock()
    mock_wake_listener.listen.return_value = _one_wake_event()

    mock_speak = AsyncMock()
    mock_transcribe = AsyncMock(return_value=transcribe_result)
    mock_record = AsyncMock(return_value=MagicMock())

    with patch("rex.voice_loop_optimized.settings") as mock_settings:
        mock_settings.use_openclaw_voice_backend = False  # bridge set directly below

        vl = VoiceLoop(
            MagicMock(),  # placeholder — overwritten below
            wake_listener=mock_wake_listener,
            detection_source=AsyncMock(),
            record_phrase=mock_record,
            transcribe=mock_transcribe,
            speak=mock_speak,
        )

    vl._assistant = mock_bridge
    return vl, mock_speak, mock_transcribe


class TestVoiceLoopOptimizedRunWithVoiceBridge:
    """VoiceLoop.run() routes transcripts through VoiceBridge.generate_reply."""

    @pytest.mark.asyncio
    async def test_generate_reply_called_with_transcript(self):
        """generate_reply is called with the STT transcript (no voice_mode kwarg in optimized)."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="It is 3pm.")

        vl, _, _ = _make_vl(mock_bridge, transcribe_result="what time is it")
        await vl.run(max_interactions=1)

        mock_bridge.generate_reply.assert_awaited_once_with("what time is it")

    @pytest.mark.asyncio
    async def test_response_passed_to_speak(self):
        """Response from generate_reply is passed to the speak callable."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="It is 3pm")

        vl, mock_speak, _ = _make_vl(mock_bridge, transcribe_result="what time is it")
        await vl.run(max_interactions=1)

        mock_speak.assert_awaited_once()
        called_text = mock_speak.await_args[0][0]
        assert "It is 3pm" in called_text

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_generate_reply(self):
        """Empty STT result skips generate_reply."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="response")

        vl, mock_speak, _ = _make_vl(mock_bridge, transcribe_result="")
        await vl.run(max_interactions=1)

        mock_bridge.generate_reply.assert_not_awaited()
        mock_speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generate_reply_exception_does_not_propagate(self):
        """generate_reply exception is caught; run() continues without crashing."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        vl, mock_speak, _ = _make_vl(mock_bridge, transcribe_result="hello")
        await vl.run(max_interactions=1)

        mock_speak.assert_not_awaited()
