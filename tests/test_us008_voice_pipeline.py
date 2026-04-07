"""Tests for US-008: capture-to-STT stage logging and error handling."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

np = pytest.importorskip("numpy")

from rex.assistant_errors import SpeechToTextError  # noqa: E402
from rex.voice_loop import VoiceLoop  # noqa: E402


def _make_voice_loop(
    record_audio=None,
    transcribe_result="hello",
    transcribe_raises=None,
    sample_rate=16000,
):
    """Build a minimal VoiceLoop with mocked callables."""
    if record_audio is None:
        record_audio = np.ones(16000, dtype=np.float32)  # 1 second of audio

    async def _record_phrase():
        return record_audio

    async def _detection_source():
        return np.ones(4, dtype=np.float32)

    async def _transcribe(audio):
        if transcribe_raises is not None:
            raise transcribe_raises
        return transcribe_result

    async def _speak(text):
        pass

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value="response")

    # Wake listener that fires once then stops
    class _OnceListener:
        def __init__(self):
            self._fired = False

        async def listen(self, source):
            if not self._fired:
                self._fired = True
                yield await source()

        def stop(self):
            pass

    return VoiceLoop(
        assistant,
        wake_listener=_OnceListener(),
        detection_source=_detection_source,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
        sample_rate=sample_rate,
    )


def test_audio_capture_complete_log_emitted(caplog):
    """AC 1: audio_capture_complete event is logged after record_phrase completes."""
    loop = _make_voice_loop()

    with caplog.at_level(logging.DEBUG, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    events = [getattr(r, "event", None) for r in caplog.records]
    assert "audio_capture_complete" in events, "Expected audio_capture_complete log event"


def test_audio_capture_complete_log_includes_duration(caplog):
    """AC 1: audio_capture_complete log includes audio_duration_s field."""
    sample_rate = 16000
    audio = np.ones(sample_rate * 2, dtype=np.float32)  # 2 seconds
    loop = _make_voice_loop(record_audio=audio, sample_rate=sample_rate)

    with caplog.at_level(logging.DEBUG, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    records = [r for r in caplog.records if getattr(r, "event", None) == "audio_capture_complete"]
    assert records, "Expected audio_capture_complete log record"
    record = records[0]
    duration = getattr(record, "audio_duration_s", None)
    assert duration is not None, "audio_duration_s field must be present"
    assert abs(duration - 2.0) < 0.01, f"Expected ~2.0s duration, got {duration}"


def test_stt_handoff_log_emitted(caplog):
    """AC 2: stt_handoff event is logged before transcription begins."""
    loop = _make_voice_loop()

    with caplog.at_level(logging.DEBUG, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    events = [getattr(r, "event", None) for r in caplog.records]
    assert "stt_handoff" in events, "Expected stt_handoff log event"


def test_capture_before_stt_handoff_ordering(caplog):
    """AC 2: audio_capture_complete is logged before stt_handoff."""
    loop = _make_voice_loop()

    with caplog.at_level(logging.DEBUG, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    events = [getattr(r, "event", None) for r in caplog.records]
    assert "audio_capture_complete" in events
    assert "stt_handoff" in events
    cap_idx = events.index("audio_capture_complete")
    stt_idx = events.index("stt_handoff")
    assert cap_idx < stt_idx, "audio_capture_complete must be logged before stt_handoff"


def test_stt_error_logged_and_pipeline_resets(caplog):
    """AC 3: if STT fails, error is logged and the pipeline resets (loop continues, no hang)."""
    loop = _make_voice_loop(transcribe_raises=SpeechToTextError("whisper failed"))

    with caplog.at_level(logging.ERROR, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    error_records = [r for r in caplog.records if getattr(r, "event", None) == "stt_error"]
    assert error_records, "Expected stt_error log event on STT failure"
    assert "resetting pipeline" in error_records[0].getMessage()
