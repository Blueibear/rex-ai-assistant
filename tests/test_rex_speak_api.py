from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("flask")


class DummyTTS:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def tts_to_file(self, *, text, speaker_wav, language, file_path):
        Path(file_path).write_bytes(b"audio")
        self.calls.append((text, speaker_wav, language, file_path))


def _mock_tts(monkeypatch):
    # Fake module structure: TTS.api.TTS
    fake_api = ModuleType("TTS.api")
    fake_api.TTS = DummyTTS

    fake_pkg = ModuleType("TTS")
    fake_pkg.api = fake_api

    monkeypatch.setitem(sys.modules, "TTS", fake_pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)


def _load_app(monkeypatch, tmp_path):
    _mock_tts(monkeypatch)

    # Set expected environment variables
    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    monkeypatch.setenv("REX_RATE_LIMIT", "100/minute")
    monkeypatch.setenv("REX_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("REX_WAKEWORD", "rex")

    # Reset config
    import config
    monkeypatch.setattr(config, "_cached_config", None, raising=False)

    # Reload or import the app module
    module_name = "rex_speak_api"
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)

    return module.app, module


def test_speak_requires_api_key(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    # Mock speaker_wav path always valid
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        # Missing API key
        resp = client.post("/speak", json={"text": "Hello"})
        assert resp.status_code == 401

        # With valid API key
        resp = client.post("/speak", json={"text": "Hello"}, headers={"X-API-Key": "secret"})
        assert resp.status_code == 200
        assert resp.data == b"audio"
        assert "audio" in resp.content_type


def test_speak_requires_text_param(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        response = client.post("/speak", json={}, headers={"X-API-Key": "secret"})

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"].lower().startswith("missing"), "Should report missing 'text'"
