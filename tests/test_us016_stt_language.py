"""US-016: STT language handling for 'auto' mode and edge cases."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import patch

_FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "  # minimal bytes that passes WAV check


def _make_fake_whisper(call_log: list[str | None]) -> types.ModuleType:
    fake = types.ModuleType("whisper")

    class FakeModel:
        def transcribe(self, audio, language="en", fp16=False):
            call_log.append(language)
            return {"text": "hello"}

    fake.load_model = lambda name, device="cpu": FakeModel()
    return fake


def _make_failing_then_fallback_whisper(call_log: list[str | None]) -> types.ModuleType:
    """First call (language=None) raises; second call (language='en') succeeds."""
    fake = types.ModuleType("whisper")

    class FakeModel:
        def transcribe(self, audio, language="en", fp16=False):
            call_log.append(language)
            if language is None:
                raise RuntimeError("auto-detect not supported")
            return {"text": "hello fallback"}

    fake.load_model = lambda name, device="cpu": FakeModel()
    return fake


def _run_transcribe(stt):
    """Run transcribe with audio pre-processing patched out (numpy not available)."""
    with (
        patch("rex.voice_loop._prepare_audio_for_stt", return_value=b"audio"),
        patch("rex.voice_loop._to_wav_buffer", return_value=_FAKE_WAV),
    ):
        return asyncio.run(stt.transcribe(audio=[], sample_rate=16000))


def _make_stt(monkeypatch, whisper_module, language_value):
    """Helper: create a SpeechToText with a given language config setting."""
    from rex.voice_loop import SpeechToText
    from rex.voice_loop import settings as voice_settings

    monkeypatch.setattr(voice_settings, "whisper_language", language_value)
    with patch("rex.voice_loop._lazy_import_whisper", return_value=whisper_module):
        return SpeechToText(model_name="base", device="cpu")


def test_stt_auto_language_calls_whisper_with_none(monkeypatch) -> None:
    """'auto' config value must call Whisper with language=None."""
    calls: list[str | None] = []
    stt = _make_stt(monkeypatch, _make_fake_whisper(calls), "auto")
    _run_transcribe(stt)
    assert calls == [None]


def test_stt_empty_string_language_calls_whisper_with_none(monkeypatch) -> None:
    """Empty string config value must call Whisper with language=None."""
    calls: list[str | None] = []
    stt = _make_stt(monkeypatch, _make_fake_whisper(calls), "")
    _run_transcribe(stt)
    assert calls == [None]


def test_stt_none_language_calls_whisper_with_none(monkeypatch) -> None:
    """Explicit None config value must call Whisper with language=None."""
    calls: list[str | None] = []
    stt = _make_stt(monkeypatch, _make_fake_whisper(calls), None)
    _run_transcribe(stt)
    assert calls == [None]


def test_stt_en_language_calls_whisper_with_en(monkeypatch) -> None:
    """'en' config value must call Whisper with language='en'."""
    calls: list[str | None] = []
    stt = _make_stt(monkeypatch, _make_fake_whisper(calls), "en")
    _run_transcribe(stt)
    assert calls == ["en"]


def test_stt_auto_fallback_to_en_when_auto_detect_unsupported(monkeypatch) -> None:
    """When auto-detect raises, fall back to 'en' without crashing."""
    calls: list[str | None] = []
    stt = _make_stt(monkeypatch, _make_failing_then_fallback_whisper(calls), "auto")
    result = _run_transcribe(stt)
    # First call with None (auto), second call with "en" (fallback)
    assert calls == [None, "en"]
    assert result == "hello fallback"
