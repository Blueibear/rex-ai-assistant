"""Tests for US-P6-014: Voice identity works alongside OpenClaw VoiceBridge backend.

Acceptance criteria:
  - identify_speaker callback is called during pipeline run when VoiceBridge is active
  - Voice identity result does not block VoiceBridge.generate_reply
  - Voice identity failure (exception) is logged and pipeline continues to VoiceBridge
  - VoiceBridge still receives the transcript after voice identity runs
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.openclaw.voice_bridge import VoiceBridge


async def _one_wake_event():
    """Async generator that yields exactly one wake event then stops."""
    yield MagicMock()


def _build_loop_with_voice_id(
    mock_bridge: MagicMock,
    transcribe_result: str,
    identify_speaker_fn,
) -> tuple:
    """Build VoiceLoop with VoiceBridge _assistant and an identify_speaker callback."""
    from rex.voice_loop import VoiceLoop

    mock_wake_listener = MagicMock()
    mock_wake_listener.listen.return_value = _one_wake_event()

    mock_speak = AsyncMock()
    mock_transcribe = AsyncMock(return_value=transcribe_result)
    mock_record = AsyncMock(return_value=MagicMock())

    with patch("rex.voice_loop.settings") as mock_settings:
        mock_settings.use_openclaw_voice_backend = False  # bridge set directly

        vl = VoiceLoop(
            MagicMock(),
            wake_listener=mock_wake_listener,
            detection_source=AsyncMock(),
            record_phrase=mock_record,
            transcribe=mock_transcribe,
            speak=mock_speak,
            identify_speaker=identify_speaker_fn,
        )

    vl._assistant = mock_bridge
    return vl, mock_speak


class TestVoiceIdentityWithVoiceBridge:
    """Voice identity callback runs alongside VoiceBridge without interference."""

    @pytest.mark.asyncio
    async def test_identify_speaker_called_before_generate_reply(self):
        """identify_speaker is called during the pipeline run when VoiceBridge is active."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="Hello, James.")

        call_order = []
        identify_fn = MagicMock(side_effect=lambda *args: call_order.append("identify"))
        mock_bridge.generate_reply.side_effect = lambda t, **kw: (
            call_order.append("generate_reply") or "Hello, James."
        )

        vl, mock_speak = _build_loop_with_voice_id(
            mock_bridge, transcribe_result="hello", identify_speaker_fn=identify_fn
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        # Both should have been called
        assert "identify" in call_order
        assert "generate_reply" in call_order
        # identify_speaker runs before transcription/generate_reply
        assert call_order.index("identify") < call_order.index("generate_reply")

    @pytest.mark.asyncio
    async def test_voice_identity_failure_does_not_block_voice_bridge(self):
        """If identify_speaker raises, pipeline continues and VoiceBridge still responds."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="Sure, I can help.")

        failing_identify = MagicMock(side_effect=RuntimeError("embedding model unavailable"))

        vl, mock_speak = _build_loop_with_voice_id(
            mock_bridge, transcribe_result="help me", identify_speaker_fn=failing_identify
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        # Pipeline should have continued to VoiceBridge despite identify_speaker failure
        mock_bridge.generate_reply.assert_awaited_once_with("help me", voice_mode=True)
        mock_speak.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_voice_bridge_response_independent_of_voice_identity(self):
        """VoiceBridge response text is not affected by voice identity result."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="The weather is sunny.")

        # identify_speaker returns a user ID (normal operation)
        identify_fn = MagicMock(return_value="james")

        vl, mock_speak = _build_loop_with_voice_id(
            mock_bridge, transcribe_result="weather", identify_speaker_fn=identify_fn
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        # VoiceBridge response should be spoken regardless of identity result
        spoken = mock_speak.await_args[0][0]
        assert "sunny" in spoken

    @pytest.mark.asyncio
    async def test_no_identify_speaker_still_uses_voice_bridge(self):
        """Without identify_speaker callback, VoiceBridge still handles the transcript."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="Timer set.")

        vl, mock_speak = _build_loop_with_voice_id(
            mock_bridge,
            transcribe_result="set a timer for five minutes",
            identify_speaker_fn=None,
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge.generate_reply.assert_awaited_once_with(
            "set a timer for five minutes", voice_mode=True
        )
