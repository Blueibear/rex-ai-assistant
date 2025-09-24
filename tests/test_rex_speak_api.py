from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("flask")


class DummyTTS:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def tts_to_file(self, *, text, speaker_wav, language, file_path):
        Path(file_path).write_bytes(b"audio")  # Simulate audio output
        self.calls.append((text, speaker_wav, language, file_path))


def _mock_tts(monkeypatch):
    fake_api = ModuleType("TTS.api")
    fake_api.TTS = DummyTTS
    fake_root = ModuleType("TTS")
    fake_root.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_root)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)


def _load_app(monkeypatch, tmp_path):
    _mock_tts(monkeypatch)

    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    monkeypatch.setenv("REX_RATE_LIMIT", "100/minute")
    monkeypatch.setenv("REX_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("REX_WAKEWORD", "rex")

    # Force config reload
    import config
    monkeypatch.setattr(config, "_cached_config", None, raising=False)

    if "rex_speak_api" in sys.modules:
        module = importlib.reload(sys.modules["rex_speak_api"])
    else:
        module = importlib.import_module("rex_speak_api")

    return module.app, module


def test_speak_requires_api_key(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    # Assume speaker_wav file exists
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        # Without API key
        response = client.post("/speak", json={"text": "Hello"})
        assert response.status_code == 401, "Missing API key should return 401"

        # With correct API key
        response = client.post("/speak", json={"text": "Hello"}, headers={"X-API-Key": "secret"})
        assert response.status_code == 200, "Valid request should succeed"
        assert response.data == b"audio", "Mocked TTS should return fake audio"
        assert response.headers["Content-Type"] in ("application/octet-stream", "audio/wav")


def test_speak_requires_text_param(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    # Pretend speaker_wav file exists
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        response = client.post("/speak", json={}, headers={"X-API-Key": "secret"})
        assert response.status_code == 400
        assert response.get_json() == {"error": "Missing text parameter"}

