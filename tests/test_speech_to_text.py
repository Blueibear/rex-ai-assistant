from __future__ import annotations

import asyncio
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rex.assistant_errors import AudioFormatError
from rex.config import AppConfig, build_app_config
from rex.doctor import CheckResult, Status, run_diagnostics


def _make_fake_whisper(call_log: list[str | None]) -> types.ModuleType:
    fake = types.ModuleType("whisper")

    class FakeModel:
        def transcribe(self, audio, language="en", fp16=False):
            call_log.append(language)
            return {"text": "bonjour rex"}

    def load_model(name, device="cpu"):
        return FakeModel()

    setattr(fake, "load_model", load_model)
    return fake


def test_app_config_whisper_language_defaults_to_english() -> None:
    config = AppConfig()
    assert config.whisper_language == "en"


def test_build_app_config_accepts_null_whisper_language() -> None:
    config = build_app_config({"models": {"stt_language": None}})
    assert config.whisper_language is None


def test_speech_to_text_uses_configured_whisper_language(monkeypatch) -> None:
    from rex.voice_loop import SpeechToText
    from rex.voice_loop import settings as voice_settings

    calls: list[str | None] = []
    monkeypatch.setattr(voice_settings, "whisper_language", "fr")

    with patch("rex.voice_loop._lazy_import_whisper", return_value=_make_fake_whisper(calls)):
        stt = SpeechToText(model_name="base", device="cpu")

    transcript = asyncio.run(stt.transcribe(audio=[], sample_rate=16000))

    assert transcript == "bonjour rex"
    assert calls == ["fr"]


def test_speech_to_text_passes_none_for_auto_detect(monkeypatch) -> None:
    from rex.voice_loop import SpeechToText
    from rex.voice_loop import settings as voice_settings

    calls: list[str | None] = []
    monkeypatch.setattr(voice_settings, "whisper_language", None)

    with patch("rex.voice_loop._lazy_import_whisper", return_value=_make_fake_whisper(calls)):
        stt = SpeechToText(model_name="base", device="cpu")

    asyncio.run(stt.transcribe(audio=[], sample_rate=16000))

    assert calls == [None]


def test_speech_to_text_rejects_non_wav_audio(monkeypatch) -> None:
    from rex.voice_loop import SpeechToText

    monkeypatch.setattr("rex.voice_loop._lazy_import_whisper", lambda: _make_fake_whisper([]))
    stt = SpeechToText(model_name="base", device="cpu")

    with pytest.raises(AudioFormatError, match="Expected WAV, got ID3"):
        asyncio.run(stt.transcribe(audio=b"ID3\x04fake-mp3-data", sample_rate=16000))


def test_doctor_output_includes_whisper_language(capsys, monkeypatch) -> None:
    ok = CheckResult(name="OK", status=Status.OK, message="ok")

    monkeypatch.setattr("rex.doctor._find_project_root", lambda: None)
    monkeypatch.setattr("rex.doctor.check_python_version", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_package_installation", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_config_file", lambda root: ok)
    monkeypatch.setattr("rex.doctor.check_env_file", lambda root: ok)
    monkeypatch.setattr("rex.doctor.check_environment_variables", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_config_permissions", lambda root: ok)
    monkeypatch.setattr("rex.doctor.check_core_dependencies", lambda: [ok])
    monkeypatch.setattr("rex.doctor.check_audio_input_device", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_audio_output_device", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_lm_studio_reachability", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_external_dependencies", lambda: [ok])
    monkeypatch.setattr("rex.doctor.check_gpu_availability", lambda: ok)
    monkeypatch.setattr("rex.config.load_config", lambda reload=True: SimpleNamespace(whisper_language="de"))

    exit_code = run_diagnostics(verbose=False)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Whisper language: de" in output
