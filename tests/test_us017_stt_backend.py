"""Tests for US-017: Fix Whisper/STT runtime failure and error exposure."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from rex.doctor import Status, check_stt_backend

# ---------------------------------------------------------------------------
# AC #1 + #2: check_stt_backend reports installed backend(s)
# ---------------------------------------------------------------------------


def test_check_stt_backend_whisper_only():
    """Reports OK with openai-whisper name when only whisper is installed."""
    with (patch("rex.doctor.find_spec", side_effect=lambda m: m == "whisper" or None),):
        result = check_stt_backend()
    assert result.status == Status.OK
    assert "openai-whisper" in result.message


def test_check_stt_backend_faster_whisper_only():
    """Reports OK with faster-whisper name when only faster-whisper is installed."""
    with (patch("rex.doctor.find_spec", side_effect=lambda m: m == "faster_whisper" or None),):
        result = check_stt_backend()
    assert result.status == Status.OK
    assert "faster-whisper" in result.message


def test_check_stt_backend_both_installed():
    """Reports OK mentioning both backends when both are present."""

    def _both(m: str):
        return m in ("whisper", "faster_whisper") or None

    with patch("rex.doctor.find_spec", side_effect=_both):
        result = check_stt_backend()
    assert result.status == Status.OK
    assert "whisper" in result.message.lower()


def test_check_stt_backend_neither_installed():
    """Reports ERROR when neither whisper nor faster-whisper is installed."""
    with patch("rex.doctor.find_spec", return_value=None):
        result = check_stt_backend()
    assert result.status == Status.ERROR
    assert "No STT backend" in result.message
    assert "pip install" in result.details


# ---------------------------------------------------------------------------
# AC #3 + #4: STT runtime errors logged with traceback; loop resets
# ---------------------------------------------------------------------------


def _make_stt():
    """Return a minimal SpeechToText-like object usable in isolation."""
    from rex.voice_loop import SpeechToText

    stt = SpeechToText.__new__(SpeechToText)
    stt._model = None
    stt._language = None
    stt._load_error = None
    stt._loaded = True
    return stt


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("numpy") is None,
    reason="numpy required for SpeechToText.transcribe",
)
def test_stt_runtime_error_logged_with_traceback(caplog):
    """When whisper raises, the error is logged at ERROR level with traceback."""
    from rex.voice_loop import SpeechToText

    stt = SpeechToText.__new__(SpeechToText)
    stt._language = None
    stt._load_error = None
    stt._loaded = True

    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("GPU out of memory")
    stt._model = fake_model

    # Provide a minimal WAV buffer so format check passes
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 320)
    wav_bytes = buf.getvalue()

    with caplog.at_level(logging.ERROR, logger="rex.voice_loop"):
        with pytest.raises(Exception):  # noqa: B017
            asyncio.run(stt.transcribe(audio=wav_bytes, sample_rate=16000))

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records, "Expected at least one ERROR log record"
    # exc_info=True means the record has exc_info tuple (not all None)
    assert any(
        r.exc_info and r.exc_info[0] is not None for r in error_records
    ), "ERROR log must include traceback (exc_info)"


def test_stt_error_in_voice_loop_logs_traceback(caplog):
    """SpeechToTextError in voice loop is logged with full traceback."""
    pytest.importorskip("numpy")
    from rex.assistant_errors import SpeechToTextError
    from rex.voice_loop import VoiceLoop

    loop = VoiceLoop.__new__(VoiceLoop)
    loop._stt_timeout = 30.0
    loop._llm_timeout = 60.0
    loop._tts_timeout = 30.0
    loop._sample_rate = 16000
    loop._max_interactions = 1

    # transcribe raises SpeechToTextError immediately
    async def _bad_transcribe(_audio):
        raise SpeechToTextError("forced test failure")

    loop._transcribe = _bad_transcribe

    # record_phrase returns a dummy buffer
    async def _record(_dur):
        return b"\x00" * 32

    loop._record_phrase = _record

    # wakeword detection yields once then stops
    async def _wakeword_gen():
        yield None

    loop._detect_wakeword = _wakeword_gen

    with caplog.at_level(logging.ERROR, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    stt_errors = [
        r for r in caplog.records if r.levelno == logging.ERROR and "stt" in r.getMessage().lower()
    ]
    assert stt_errors, "Expected STT error to be logged"
    assert any(
        r.exc_info and r.exc_info[0] is not None for r in stt_errors
    ), "STT ERROR log must include traceback (exc_info)"
