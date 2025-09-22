import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")


def _mock_tts(monkeypatch, tmp_path):
    class DummyTTS:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def tts_to_file(self, text, speaker_wav, language, file_path):
            Path(file_path).write_bytes(b"audio")
            self.calls.append(text)

    fake_api = type(sys)("TTS.api")
    fake_api.TTS = DummyTTS
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)
    fake_pkg = type(sys)("TTS")
    fake_pkg.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_pkg)


def _load_app(monkeypatch, tmp_path):
    _mock_tts(monkeypatch, tmp_path)
    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")

    import rex
    import rex.config

    rex.config.get_settings.cache_clear()
    new_settings = rex.config.get_settings()
    monkeypatch.setattr("rex.config.settings", new_settings, raising=False)
    monkeypatch.setattr("rex.settings", new_settings, raising=False)

    if "rex_speak_api" in sys.modules:
        module = importlib.reload(sys.modules["rex_speak_api"])
    else:
        module = importlib.import_module("rex_speak_api")
    return module.app, module


def test_speak_requires_api_key(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

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
