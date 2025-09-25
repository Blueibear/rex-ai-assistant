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
        Path(file_path).write_bytes(b"fake")
        self.calls.append((text, speaker_wav, language, file_path))


def _load_app(monkeypatch):
    module_name = "rex_speak_api"
    if module_name in sys.modules:
        del sys.modules[module_name]

    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")

    fake_api = ModuleType("TTS.api")
    fake_api.TTS = DummyTTS
    fake_root = ModuleType("TTS")
    fake_root.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_root)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)

    module = importlib.import_module(module_name)
    return module.app, module


def test_speak_requires_api_key(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        response = client.post("/speak", json={"text": "Hello"})
    assert response.status_code == 401

    with app.test_client() as client:
        response = client.post(
            "/speak",
            json={"text": "Hello"},
            headers={"X-API-Key": "secret"},
        )
    assert response.status_code == 200


def test_speak_requires_text_param(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        response = client.post(
            "/speak",
            json={},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 400
    body = response.get_json()
    assert body == {"error": "Missing text parameter"}
