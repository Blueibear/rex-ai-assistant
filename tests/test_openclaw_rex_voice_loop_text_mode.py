"""Tests for US-P6-010: rex/voice_loop.py VoiceLoop routes through VoiceBridge (text mode).

Acceptance criteria:
  - When _assistant is a VoiceBridge, run() calls generate_reply(transcript, voice_mode=True)
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


def _make_vl(mock_bridge: MagicMock, transcribe_result: str = "hello rex") -> object:
    """Build a VoiceLoop with mock VoiceBridge and injectable transcribe result."""
    from rex.voice_loop import VoiceLoop

    mock_wake_listener = MagicMock()
    mock_wake_listener.listen.return_value = _one_wake_event()

    mock_speak = AsyncMock()
    mock_transcribe = AsyncMock(return_value=transcribe_result)
    mock_record = AsyncMock(return_value=MagicMock())

    with patch("rex.voice_loop.settings") as mock_settings:
        mock_settings.use_openclaw_voice_backend = False  # bridge already set directly

        vl = VoiceLoop(
            MagicMock(),  # placeholder assistant — will be overwritten below
            wake_listener=mock_wake_listener,
            detection_source=AsyncMock(),
            record_phrase=mock_record,
            transcribe=mock_transcribe,
            speak=mock_speak,
        )

    # Replace _assistant directly with the mock bridge (flag already tested in US-P6-009)
    vl._assistant = mock_bridge
    return vl, mock_speak, mock_transcribe


class TestVoiceLoopRunWithVoiceBridge:
    """VoiceLoop.run() routes transcripts through VoiceBridge.generate_reply."""

    @pytest.mark.asyncio
    async def test_generate_reply_called_with_transcript(self):
        """generate_reply is called with the STT transcript."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="It is 3pm.")

        vl, mock_speak, _ = _make_vl(mock_bridge, transcribe_result="what time is it")

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge.generate_reply.assert_awaited_once_with("what time is it", voice_mode=True)

    @pytest.mark.asyncio
    async def test_response_passed_to_speak(self):
        """Response from generate_reply is passed to the speak callable."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="It is 3pm")

        vl, mock_speak, _ = _make_vl(mock_bridge, transcribe_result="what time is it")

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        # Response gets a period appended if missing; verify speak was called
        mock_speak.assert_awaited_once()
        called_text = mock_speak.await_args[0][0]
        assert "It is 3pm" in called_text

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_generate_reply(self):
        """Empty STT result skips generate_reply."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="response")

        vl, mock_speak, _ = _make_vl(mock_bridge, transcribe_result="")

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge.generate_reply.assert_not_awaited()
        mock_speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generate_reply_exception_does_not_propagate(self):
        """generate_reply exception is caught; run() continues without crashing."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        vl, mock_speak, _ = _make_vl(mock_bridge, transcribe_result="hello")

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            # Should not raise
            await vl.run(max_interactions=1)

        mock_speak.assert_not_awaited()
