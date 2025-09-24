from __future__ import annotations

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
            Path(file_path).write_bytes(b"audio")  # Simulate TTS output
            self.calls.append((text, file_path))

    # Fake TTS module structure: TTS.api.TTS
    fake_api = type(sys)("TTS.api")
    fake_api.TTS = DummyTTS
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)

    fake_pkg = type(sys)("TTS")
    fake_pkg.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_pkg)


def _load_app(monkeypatch, tmp_path):
    _mock_tts(monkeypatch, tmp_path)
    monkeypatch.setenv("REX_SPEAK_API_KEY", "secret")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    monkeypatch.setenv("REX_RATE_LIMIT", "100/minute")
    monkeypatch.setenv("REX_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("REX_WAKEWORD", "rex")

    import config
    monkeypatch.setattr(config, "_cached_config", None, raising=False)

    if "rex_speak_api" in sys.modules:
        module = importlib.reload(sys.modules["rex_speak_api"])
    else:
        module = importlib.import_module("rex_speak_api")

    return module.app, module


def test_speak_requires_api_key(monkeypatch, tmp_path):
    """Verify API key is required and TTS output is valid."""
    app, module = _load_app(monkeypatch, tmp_path)

    # Always simulate speaker_wav existing
    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    # Missing API key → 401 Unauthorized
    with app.test_client() as client:
        response = client.post("/speak", json={"text": "Hello"})
    assert response.status_code == 401, "Missing API key should return 401"

    # Valid request with API key
    with app.test_client() as client:
        response = client.post(
            "/speak",
            json={"text": "Hello"},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200, "Valid request should return 200"
    assert response.data == b"audio", "TTS audio content should match mock output"
    assert response.headers["Content-Type"] in ("application/octet-stream", "audio/wav"), "Unexpected content type"


def test_speak_requires_text_param(monkeypatch, tmp_path):
    """Ensure /speak fails if no 'text' provided."""
    app, module = _load_app(monkeypatch, tmp_path)

    monkeypatch.setattr(module.os.path, "exists", lambda path: True)

    with app.test_client() as client:
        response = client.post("/speak", json={}, headers={"X-API-Key": "secret"})

    assert response.status_code in (400, 422), "Missing text input should return 400 or 422"

            headers={"X-API-Key": "secret"},
        )
    assert response.status_code == 200
