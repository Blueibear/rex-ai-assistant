"""Tests for the rex_speak_api Flask service."""

from __future__ import annotations

import importlib
import sys
import wave
from types import ModuleType

import pytest

pytest.importorskip("flask")


class DummyTTS:
    """Mock for Coqui TTS engine."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def tts_to_file(self, *, text, speaker_wav, language, file_path, **kwargs):
        with wave.open(file_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 160)
        self.calls.append(
            {
                "text": text,
                "speaker_wav": speaker_wav,
                "language": language,
                "file_path": file_path,
                "kwargs": kwargs,
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

    # Mock generate_speech to avoid real TTS/audio dependencies
    def _fake_generate_speech(text: str, language: str, user_key: str) -> bytes:
        """Return fake WAV audio bytes for testing."""
        import io
        import wave

        # Create a minimal valid WAV file in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 160)
        return wav_io.getvalue()

    monkeypatch.setattr(module, "generate_speech", _fake_generate_speech)

    app = module.app
    app.config.update(TESTING=True)
    return app, module


@pytest.mark.unit
def test_import_without_api_key(monkeypatch):
    """Module can be imported without REX_SPEAK_API_KEY set."""
    module_name = "rex_speak_api"
    monkeypatch.delenv("REX_SPEAK_API_KEY", raising=False)
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    _mock_tts(monkeypatch)

    if module_name in sys.modules:
        del sys.modules[module_name]

    # Import should succeed without API key
    module = importlib.import_module(module_name)
    assert module.app is not None
    assert module.API_KEY is None


@pytest.mark.unit
def test_missing_api_key_prevents_start(monkeypatch):
    """Fail fast if REX_SPEAK_API_KEY is missing when calling main()."""
    module_name = "rex_speak_api"
    monkeypatch.delenv("REX_SPEAK_API_KEY", raising=False)
    monkeypatch.setenv("REX_ACTIVE_USER", "james")
    _mock_tts(monkeypatch)

    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.import_module(module_name)

    # main() raises RuntimeError (caught by exception handler → sys.exit(1))
    with pytest.raises((RuntimeError, SystemExit)):
        module.main()


@pytest.mark.integration
def test_unauthorized_requests_return_401(monkeypatch, tmp_path):
    """Requests without valid API key return 401."""
    app, module = _load_app(monkeypatch, tmp_path)

    with app.test_client() as client:
        # No API key at all
        resp = client.post("/speak", json={"text": "Hello"})
        assert resp.status_code == 401
        assert (
            "unauthorized" in resp.get_json()["error"]["message"].lower()
            or "invalid" in resp.get_json()["error"]["message"].lower()
            or "missing" in resp.get_json()["error"]["message"].lower()
        )

        # Wrong API key
        resp = client.post("/speak", json={"text": "Hello"}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

        # Empty API key
        resp = client.post("/speak", json={"text": "Hello"}, headers={"X-API-Key": ""})
        assert resp.status_code == 401


@pytest.mark.integration
def test_speak_requires_api_key(monkeypatch, tmp_path):
    app, module = _load_app(monkeypatch, tmp_path)

    with app.test_client() as client:
        # Valid key
        resp = client.post("/speak", json={"text": "Hello"}, headers={"X-API-Key": "secret"})
        assert resp.status_code == 200
        assert resp.data
        assert "audio/wav" in resp.content_type


@pytest.mark.integration
def test_speak_requires_text(monkeypatch, tmp_path):
    app, _ = _load_app(monkeypatch, tmp_path)

    with app.test_client() as client:
        resp = client.post("/speak", json={"api_key": "secret"}, headers={"X-API-Key": "secret"})

    assert resp.status_code == 400
    assert "text" in resp.get_json()["error"]["message"].lower()


@pytest.mark.integration
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
    assert "maximum length" in response.get_json()["error"]["message"]


@pytest.mark.integration
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
    assert response.data
    assert "audio/wav" in response.content_type


@pytest.mark.integration
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
    assert blocked.get_json()["error"]["message"] == "Too many requests"


@pytest.mark.integration
def test_spoofed_xff_does_not_bypass_rate_limit(monkeypatch, tmp_path):
    """X-Forwarded-For header from untrusted source should not bypass rate limiting."""
    app, module = _load_app(monkeypatch, tmp_path)

    # Ensure TRUSTED_PROXIES doesn't include our test client IP
    module.TRUSTED_PROXIES = {"192.168.1.1"}  # Different from test client

    with app.test_client() as client:
        # Make 2 requests with different spoofed X-Forwarded-For headers
        for i in range(2):
            resp = client.post(
                "/speak",
                json={"text": "hi"},
                headers={
                    "X-API-Key": "secret",
                    "X-Forwarded-For": f"10.0.0.{i}",  # Trying to spoof different IPs
                },
            )
            assert resp.status_code == 200

        # Third request should be blocked (same remote_addr, spoofed XFF ignored)
        blocked = client.post(
            "/speak",
            json={"text": "hi again"},
            headers={
                "X-API-Key": "secret",
                "X-Forwarded-For": "10.0.0.99",  # Different spoofed IP
            },
        )

    assert blocked.status_code == 429
    assert blocked.get_json()["error"]["message"] == "Too many requests"


@pytest.mark.integration
def test_trusted_proxy_xff_honored(monkeypatch, tmp_path):
    """X-Forwarded-For from trusted proxy should be honored."""
    app, module = _load_app(monkeypatch, tmp_path)

    # Configure test client IP as trusted
    # Flask test client uses 127.0.0.1 by default
    module.TRUSTED_PROXIES = {"127.0.0.1", "::1"}

    with app.test_client() as client:
        # Request 1 from IP 10.0.0.1
        resp1 = client.post(
            "/speak",
            json={"text": "hi"},
            headers={
                "X-API-Key": "secret",
                "X-Forwarded-For": "10.0.0.1",
            },
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert resp1.status_code == 200

        # Request 2 from same IP should count toward same limit
        resp2 = client.post(
            "/speak",
            json={"text": "hi"},
            headers={
                "X-API-Key": "secret",
                "X-Forwarded-For": "10.0.0.1",
            },
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert resp2.status_code == 200

        # Request 3 from same IP should be rate limited
        resp3 = client.post(
            "/speak",
            json={"text": "hi"},
            headers={
                "X-API-Key": "secret",
                "X-Forwarded-For": "10.0.0.1",
            },
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert resp3.status_code == 429

        # But request from different IP should succeed
        resp4 = client.post(
            "/speak",
            json={"text": "hi"},
            headers={
                "X-API-Key": "secret",
                "X-Forwarded-For": "10.0.0.2",  # Different client IP
            },
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert resp4.status_code == 200
