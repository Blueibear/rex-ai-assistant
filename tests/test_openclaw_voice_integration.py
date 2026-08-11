"""Canonical voice integration tests with the legacy OpenClaw flag enabled.

US-097 supersedes the old US-P6-013 brain-swap behavior: Rex's Assistant remains
the canonical TurnEngine-backed brain even when ``use_openclaw_voice_backend``
is true. OpenClaw remains an external capability/compatibility provider and must
not replace the Assistant inside the supported voice loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


async def _one_wake_event():
    """Async generator that yields exactly one wake event then stops."""
    yield MagicMock()


def _build_integrated_voice_loop(
    mock_assistant: MagicMock, transcribe_result: str
) -> tuple[object, AsyncMock, MagicMock]:
    """Build canonical VoiceLoop while the legacy OpenClaw flag is enabled."""
    from rex.voice_loop import VoiceLoop

    mock_wake_listener = MagicMock()
    mock_wake_listener.listen.return_value = _one_wake_event()

    mock_speak = AsyncMock()
    mock_transcribe = AsyncMock(return_value=transcribe_result)
    mock_record = AsyncMock(return_value=MagicMock())

    with (
        patch("rex.voice_loop.settings") as mock_settings,
        patch("rex.openclaw.voice_bridge.VoiceBridge") as mock_bridge_cls,
    ):
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = "http://localhost:8765"

        vl = VoiceLoop(
            mock_assistant,
            wake_listener=mock_wake_listener,
            detection_source=AsyncMock(),
            record_phrase=mock_record,
            transcribe=mock_transcribe,
            speak=mock_speak,
        )

    return vl, mock_speak, mock_bridge_cls


class TestVoicePipelineIntegration:
    """End-to-end wakeword -> STT -> canonical Assistant -> TTS integration."""

    @pytest.mark.asyncio
    async def test_full_pipeline_wake_to_tts(self):
        """Legacy flag does not replace Assistant; its reply reaches TTS."""
        mock_assistant = MagicMock()
        mock_assistant.generate_reply = AsyncMock(return_value="The time is 3pm.")

        vl, mock_speak, mock_bridge_cls = _build_integrated_voice_loop(
            mock_assistant, transcribe_result="what time is it"
        )
        assert vl._assistant is mock_assistant
        mock_bridge_cls.assert_not_called()

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_assistant.generate_reply.assert_awaited_once_with("what time is it", voice_mode=True)
        mock_speak.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_assistant_response_reaches_tts(self):
        """Canonical Assistant response text is what gets spoken."""
        mock_assistant = MagicMock()
        mock_assistant.generate_reply = AsyncMock(return_value="Sunny with a high of 22 degrees")

        vl, mock_speak, mock_bridge_cls = _build_integrated_voice_loop(
            mock_assistant, transcribe_result="what is the weather"
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge_cls.assert_not_called()
        spoken = mock_speak.await_args.args[0]
        assert "Sunny" in spoken
        assert "22 degrees" in spoken

    @pytest.mark.asyncio
    async def test_pipeline_empty_stt_skips_assistant_and_tts(self):
        """Empty STT result skips the Assistant and TTS."""
        mock_assistant = MagicMock()
        mock_assistant.generate_reply = AsyncMock(return_value="I heard nothing")

        vl, mock_speak, mock_bridge_cls = _build_integrated_voice_loop(
            mock_assistant, transcribe_result=""
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge_cls.assert_not_called()
        mock_assistant.generate_reply.assert_not_awaited()
        mock_speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pipeline_empty_assistant_response_skips_tts(self):
        """Empty canonical Assistant response produces no stale TTS output."""
        mock_assistant = MagicMock()
        mock_assistant.generate_reply = AsyncMock(return_value="")

        vl, mock_speak, mock_bridge_cls = _build_integrated_voice_loop(
            mock_assistant, transcribe_result="hello"
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge_cls.assert_not_called()
        mock_assistant.generate_reply.assert_awaited_once_with("hello", voice_mode=True)
        mock_speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pipeline_assistant_exception_does_not_crash_loop(self):
        """Assistant exception is contained and does not trigger VoiceBridge fallback."""
        mock_assistant = MagicMock()
        mock_assistant.generate_reply = AsyncMock(side_effect=RuntimeError("backend unavailable"))

        vl, mock_speak, mock_bridge_cls = _build_integrated_voice_loop(
            mock_assistant, transcribe_result="hello rex"
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge_cls.assert_not_called()
        mock_assistant.generate_reply.assert_awaited_once()
        mock_speak.assert_not_awaited()
