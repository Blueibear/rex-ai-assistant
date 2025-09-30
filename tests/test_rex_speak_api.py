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


def _patch_tts(monkeypatch):
    fake_api = ModuleType("TTS.api")
    fake_api.TTS = DummyTTS

    fake_pkg = ModuleType("TTS")
    fake_pkg.api = fake_api

    monkeypatch.setitem(sys.modules, "TTS", fake_pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)


def _load_app(monkeypatch):
    module_name = "rex_speak_api"

    if module_name in sys.modules:
        del sys.modules[module_name]
    if "rex.config" in sys.modules:
        del sys.modules["rex.config"]

    _patch_tts(monkeypatch)

    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")

    module = importlib.import_module(module_name)
    return module.app, module


def test_missing_api_key_prevents_start(monkeypatch):
    module_name = "rex_speak_api"
    if module_name in sys.modules:
        del sys.modules[module_name]
    if "rex.config" in sys.modules:
        del sys.modules["rex.config"]

    monkeypatch.delenv("REX_SPEAK_API_KEY", raising=False)
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    _patch_tts(monkeypatch)

    with pytest.raises(RuntimeError):
        importlib.import_module(module_name)


def test_speak_requires_api_key(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        resp = client.post("/speak", json={"text": "Hi"})
    assert resp.status_code == 401

    with app.test_client() as client:
        resp = client.post(
            "/speak",
            json={"text": "Hi"},
            headers={"X-API-Key": "secret"},
        )
    assert resp.status_code == 200
    assert resp.data == b"audio"
    assert "audio/wav" in resp.content_type


def test_speak_requires_text_param(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch)
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        resp = client.post(
            "/speak",
            json={},
            headers={"X-API-Key": "secret"},
        )

    assert resp.status_code == 400
    body = resp.get_json()
    assert body == {"error": "Missing 'text' parameter"}
