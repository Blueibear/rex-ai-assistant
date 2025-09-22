"""Tests for the rex_speak_api Flask service."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("flask")


class DummyTTS:
    """Lightweight stand-in for the Coqui XTTS engine."""

    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []

    def tts_to_file(self, *, text, speaker_wav, language, file_path):
        Path(file_path).write_bytes(b"RIFFdemoWAVE")
        self.calls.append(
            {
                "text": text,
                "speaker_wav": speaker_wav,
                "language": language,
                "file_path": file_path,
            }
        )


def _load_app(monkeypatch):
    """Import the Flask app with a stubbed TTS backend."""

    # Install a stub ``TTS`` package before importing the module under test.
    fake_api = types.ModuleType("TTS.api")
    fake_api.TTS = DummyTTS
    fake_pkg = types.ModuleType("TTS")
    fake_pkg.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)

    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")
    monkeypatch.setenv("REX_SPEAK_RATE_LIMIT", "2")
    monkeypatch.setenv("REX_SPEAK_RATE_WINDOW", "60")

    sys.modules.pop("rex_speak_api", None)
    module = importlib.import_module("rex_speak_api")

    dummy_engine = DummyTTS()
    module._TTS_ENGINE = dummy_engine
    module._get_tts_engine = lambda: dummy_engine
    module._RATE_STATE.clear()

    app = module.app
    app.config.update(TESTING=True)
    return app, module


def test_speak_requires_text(monkeypatch):
    app, _module = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.post(
            "/speak",
            json={"api_key": "secret"},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 400
    assert response.get_json()["error"]


def test_speak_generates_audio_when_voice_missing(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch)

    # Simulate a memory profile that references a non-existent speaker file.
    module.USER_VOICES["james"] = str(tmp_path / "missing.wav")

    with app.test_client() as client:
        response = client.post(
            "/speak",
            json={"text": "Hello there"},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    assert response.mimetype == "audio/wav"
    assert module._TTS_ENGINE.calls
    call = module._TTS_ENGINE.calls[0]
    assert call["speaker_wav"] is None
    assert Path(call["file_path"]).suffix == ".wav"


def test_speak_requires_api_key(monkeypatch):
    app, _module = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.post("/speak", json={"text": "hi"})

    assert response.status_code == 401


def test_speak_rate_limit(monkeypatch):
    app, module = _load_app(monkeypatch)

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


def test_speak_rejects_long_text(monkeypatch):
    app, _module = _load_app(monkeypatch)

    payload = {"text": "x" * 2000}

    with app.test_client() as client:
        response = client.post(
            "/speak",
            json=payload,
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 400
    assert "maximum length" in response.get_json()["error"]
