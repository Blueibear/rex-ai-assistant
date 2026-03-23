"""Tests for US-P6-015: HA TTS delivery through voice loop + OpenClaw VoiceBridge.

Acceptance criteria:
  - VoiceBridge response is delivered to HaTtsClient.speak() when HA TTS is the speak backend
  - HA TTS speak is called with the exact response text from VoiceBridge
  - HA TTS failure (network error) does not crash the voice loop
  - Tests pass without a real Home Assistant instance
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.ha_tts.client import HaTtsClient, TtsResult
from rex.openclaw.voice_bridge import VoiceBridge


async def _one_wake_event():
    """Async generator that yields exactly one wake event then stops."""
    yield MagicMock()


def _build_ha_tts_speak(ha_client: HaTtsClient):
    """Wrap HaTtsClient.speak() as an async speak callable for VoiceLoop."""

    async def _async_speak(text: str) -> None:
        await asyncio.to_thread(ha_client.speak, text)

    return _async_speak


def _build_loop_with_ha_tts(
    mock_bridge: MagicMock,
    transcribe_result: str,
    ha_speak: object,
) -> object:
    """Build VoiceLoop with VoiceBridge _assistant and HA TTS speak callable."""
    from rex.voice_loop import VoiceLoop

    mock_wake_listener = MagicMock()
    mock_wake_listener.listen.return_value = _one_wake_event()

    mock_transcribe = AsyncMock(return_value=transcribe_result)
    mock_record = AsyncMock(return_value=MagicMock())

    with patch("rex.voice_loop.settings") as mock_settings:
        mock_settings.use_openclaw_voice_backend = False

        vl = VoiceLoop(
            MagicMock(),
            wake_listener=mock_wake_listener,
            detection_source=AsyncMock(),
            record_phrase=mock_record,
            transcribe=mock_transcribe,
            speak=ha_speak,
        )

    vl._assistant = mock_bridge
    return vl


class TestHaTtsWithVoiceBridge:
    """HA TTS is the delivery backend when VoiceBridge is the OpenClaw assistant."""

    @pytest.mark.asyncio
    async def test_voice_bridge_response_delivered_via_ha_tts(self):
        """VoiceBridge response text is passed to HaTtsClient.speak()."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="The lights are on.")

        mock_ha_client = MagicMock(spec=HaTtsClient)
        mock_ha_client.speak.return_value = TtsResult(ok=True)

        vl = _build_loop_with_ha_tts(
            mock_bridge,
            transcribe_result="turn on the lights",
            ha_speak=_build_ha_tts_speak(mock_ha_client),
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_ha_client.speak.assert_called_once()
        spoken_text = mock_ha_client.speak.call_args[0][0]
        assert "lights are on" in spoken_text

    @pytest.mark.asyncio
    async def test_ha_tts_receives_correct_response_not_transcript(self):
        """HA TTS gets the VoiceBridge response, not the original transcript."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="Playing your jazz playlist.")

        mock_ha_client = MagicMock(spec=HaTtsClient)
        mock_ha_client.speak.return_value = TtsResult(ok=True)

        vl = _build_loop_with_ha_tts(
            mock_bridge,
            transcribe_result="play jazz",
            ha_speak=_build_ha_tts_speak(mock_ha_client),
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        spoken_text = mock_ha_client.speak.call_args[0][0]
        assert "jazz playlist" in spoken_text
        assert "play jazz" not in spoken_text  # transcript, not response

    @pytest.mark.asyncio
    async def test_ha_tts_failure_does_not_crash_voice_loop(self):
        """HA TTS network error is caught; voice loop does not raise."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="Done.")

        mock_ha_client = MagicMock(spec=HaTtsClient)
        mock_ha_client.speak.side_effect = OSError("Connection refused")

        vl = _build_loop_with_ha_tts(
            mock_bridge,
            transcribe_result="hello",
            ha_speak=_build_ha_tts_speak(mock_ha_client),
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            # Should not raise even though HA TTS fails
            await vl.run(max_interactions=1)

        # speak was attempted
        mock_ha_client.speak.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_stt_skips_ha_tts(self):
        """Empty STT result means VoiceBridge and HA TTS are both skipped."""
        mock_bridge = MagicMock(spec=VoiceBridge)
        mock_bridge.generate_reply = AsyncMock(return_value="response")

        mock_ha_client = MagicMock(spec=HaTtsClient)
        mock_ha_client.speak.return_value = TtsResult(ok=True)

        vl = _build_loop_with_ha_tts(
            mock_bridge,
            transcribe_result="",
            ha_speak=_build_ha_tts_speak(mock_ha_client),
        )

        with patch("rex.voice_latency.VoiceLatencyTracker", MagicMock()):
            await vl.run(max_interactions=1)

        mock_bridge.generate_reply.assert_not_awaited()
        mock_ha_client.speak.assert_not_called()
