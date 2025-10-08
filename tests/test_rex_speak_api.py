"""Tests for the rex_speak_api Flask service."""

from __future__ import annotations

import importlib
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
    monkeypatch.setenv("REX_SPEAK_RATE_LIMIT", "2")
    monkeypatch.setenv("REX_SPEAK_RATE_WINDOW", "60")

    if "config" in sys.modules:
        monkeypatch.setattr(sys.modules["config"], "_cached_config", None, raising=False)

    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.import_module(module_name)
    dummy_engine = DummyTTS()
    module._TTS_ENGINE = dummy_engine
    module._get_tts_engine = lambda: dummy_engine
    module._RATE_STATE.clear()

    app = module.app
    app.config.update(TESTING=True)
    return app, module


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
        # Missing key
        resp = client.post("/speak", json={"text": "Hello"})
        assert resp.status_code == 401

        # Valid key
        resp = client.post("/speak", json={"text": "Hello"}, headers={"X-API-Key": "secret"})
        assert resp.status_code == 200
        assert resp.data.startswith(b"FAKEAUDIO")
        assert "audio/wav" in resp.content_type


def test_speak_requires_text(monkeypatch, tmp_path):
    app, _ = _load_app(monkeypatch, tmp_path)

    with app.test_client() as client:
        resp = client.post("/speak", json={"api_key": "secret"}, headers={"X-API-Key": "secret"})

    assert resp.status_code == 400
    assert "text" in resp.get_json()["error"].lower()


def test_speak_rejects_long_text(monkeypatch, tmp_path):
    app, _ = _load_app(monkeypatch, tmp_path)

    payload = {"text": "x" * 2000}

    with app.test_client() as client:
        response = client.post(
            "/speak",
            json=payload,
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 400
    assert "maximum length" in response.get_json()["error"]


def test_speak_generates_audio_with_missing_voice(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    # Simulate profile with non-existent file
    module.USER_VOICES["james"] = str(tmp_path / "missing.wav")

    with app.test_client() as client:
        response = client.post(
            "/speak",
            json={"text": "Hello there"},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    assert module._TTS_ENGINE.calls
    call = module._TTS_ENGINE.calls[0]
    assert call["speaker_wav"] is None
    assert Path(call["file_path"]).suffix == ".wav"


def test_speak_rate_limit(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    with app.test_client() as client:
        for _ in range(2):
            resp = client.post(
                "/speak",
                json={"text": "hi"},
                headers={"X-API-Key": "secret"},
            )
            assert resp.status_code == 200

        blocked = client.post(
            "/speak",
            json={"text": "hi again"},
            headers={"X-API-Key": "secret"},
        )

    assert blocked.status_code == 429
    assert blocked.get_json()["error"] == "Too many requests"
