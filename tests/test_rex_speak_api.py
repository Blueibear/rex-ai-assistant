"""Tests for the rex_speak_api Flask service."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("flask")


class DummyTTS:
    """Mock for Coqui TTS engine."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def tts_to_file(self, *, text, speaker_wav, language, file_path):
        Path(file_path).write_bytes(b"FAKEAUDIO")
        self.calls.append(
            {
                "text": text,
                "speaker_wav": speaker_wav,
                "language": language,
                "file_path": file_path,
            }
        )


def _mock_tts(monkeypatch):
    """Patch TTS.api.TTS with dummy class."""
    fake_api = ModuleType("TTS.api")
    fake_api.TTS = DummyTTS

    fake_pkg = ModuleType("TTS")
    fake_pkg.api = fake_api

    monkeypatch.setitem(sys.modules, "TTS", fake_pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)


def _load_app(monkeypatch, tmp_path) -> tuple:
    """Prepare environment and import rex_speak_api with mocks."""
    module_name = "rex_speak_api"

    _mock_tts(monkeypatch)

    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")

    # Clean config cache if needed
    if "config" in sys.modules:
        monkeypatch.setattr(sys.modules["config"], "_cached_config", None, raising=False)

    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.import_module(module_name)
    return module.app, module


def test_missing_api_key_prevents_start(monkeypatch):
    """Fail fast if REX_SPEAK_API_KEY is missing at app startup."""
    module_name = "rex_speak_api"

    monkeypatch.delenv("REX_SPEAK_API_KEY", raising=False)
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    _mock_tts(monkeypatch)

    if module_name in sys.modules:
        del sys.modules[module_name]

    with pytest.raises(RuntimeError):
        importlib.import_module(module_name)


def test_speak_requires_api_key(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    with app.test_client() as client:
        # No API key
        resp = client.post("/speak", json={"text": "Hello"})
        assert resp.status_code == 401

        # Valid API key
        resp = client.post("/speak", json={"text": "Hello"}, headers={"X-API-Key": "secret"})
        assert resp.status_code == 200
        assert resp.data.startswith(b"FAKEAUDIO")
        assert "audio/wav" in resp.content_type


def test_speak_requires_text_param(monkeypatch, tmp_path):
    app, _module = _load_app(monkeypatch, tmp_path)

    with app.test_client() as client:
        resp = client.post("/speak", json={}, headers={"X-API-Key": "secret"})

    assert resp.status_code == 400
    body = resp.get_json()
    assert "text" in body.get("error", "").lower()


def test_speak_generates_audio_with_missing_voice(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    # Simulate a profile with a missing speaker file
    module.USER_VOICES["james"] = str(tmp_path / "missing.wav")

    with app.test_client() as client:
        resp = client.post("/speak", json={"text": "Hi Rex"}, headers={"X-API-Key": "secret"})

    assert resp.status_code == 200
    assert module.xtts.calls
    call = module.xtts.calls[0]
    assert call["text"] == "Hi Rex"
    assert call["speaker_wav"] is None

