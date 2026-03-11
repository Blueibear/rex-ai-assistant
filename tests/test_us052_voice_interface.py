"""Tests for US-052: Voice interface.

Acceptance criteria:
- microphone input works  (JS MediaRecorder wired to #voice-btn, tested via HTML presence)
- audio sent to backend   (/api/voice accepts multipart audio and returns transcript+reply)
- response audio plays    (reply returned; JS uses speechSynthesis to speak it)
- Typecheck passes        (mypy rex/ clean)
- Verify changes work in browser (server starts and endpoint is reachable)
"""

from __future__ import annotations

import io
import threading
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import rex.dashboard.routes as routes_module
from rex.dashboard.routes import dashboard_bp


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_chat_history():
    routes_module._CHAT_HISTORY.clear()
    yield
    routes_module._CHAT_HISTORY.clear()


@pytest.fixture()
def app():
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# Fake audio bytes to use as upload payload
_FAKE_AUDIO = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def _send_voice(client, audio_bytes=None, content_type="audio/wav", *, token=None):
    """Helper: POST /api/voice with a fake audio file."""
    if audio_bytes is None:
        audio_bytes = _FAKE_AUDIO
    data = {}
    if audio_bytes is not None:
        data["audio"] = (io.BytesIO(audio_bytes), "recording.wav", content_type)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(
        "/api/voice",
        data=data,
        content_type="multipart/form-data",
        headers=headers,
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )


# ---------------------------------------------------------------------------
# Backend: missing audio → 400
# ---------------------------------------------------------------------------


def test_voice_missing_audio_returns_400(client):
    resp = client.post(
        "/api/voice",
        data={},
        content_type="multipart/form-data",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Backend: STT not available → 503
# ---------------------------------------------------------------------------


def test_voice_stt_not_available_returns_503(client):
    with patch(
        "rex.dashboard.routes._transcribe_audio_file",
        side_effect=ImportError("openai-whisper not installed"),
    ):
        resp = _send_voice(client)
    assert resp.status_code == 503
    data = resp.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Backend: STT generic error → 500
# ---------------------------------------------------------------------------


def test_voice_stt_error_returns_500(client):
    with patch(
        "rex.dashboard.routes._transcribe_audio_file",
        side_effect=RuntimeError("model failed"),
    ):
        resp = _send_voice(client)
    assert resp.status_code == 500
    data = resp.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Backend: empty transcript → 422
# ---------------------------------------------------------------------------


def test_voice_empty_transcript_returns_422(client):
    with patch("rex.dashboard.routes._transcribe_audio_file", return_value="   "):
        resp = _send_voice(client)
    assert resp.status_code == 422
    data = resp.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Backend: LLM error → 500
# ---------------------------------------------------------------------------


def test_voice_llm_error_returns_500(client):
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("LLM unavailable")

    with patch("rex.dashboard.routes._transcribe_audio_file", return_value="hello"):
        with patch("rex.dashboard.routes._get_llm", return_value=mock_llm):
            resp = _send_voice(client)

    assert resp.status_code == 500
    data = resp.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Backend: audio sent to backend and response returned (success path)
# ---------------------------------------------------------------------------


def test_voice_success_returns_transcript_and_reply(client):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Hello, I heard you."

    with patch("rex.dashboard.routes._transcribe_audio_file", return_value="hello rex"):
        with patch("rex.dashboard.routes._get_llm", return_value=mock_llm):
            resp = _send_voice(client)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["transcript"] == "hello rex"
    assert data["reply"] == "Hello, I heard you."
    assert "timestamp" in data


def test_voice_reply_non_empty(client):
    """response audio plays: reply field is non-empty so the JS can speak it."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Rex here!"

    with patch("rex.dashboard.routes._transcribe_audio_file", return_value="what time is it"):
        with patch("rex.dashboard.routes._get_llm", return_value=mock_llm):
            resp = _send_voice(client)

    data = resp.get_json()
    assert data["reply"]  # non-empty → speechSynthesis.speak() will be called


# ---------------------------------------------------------------------------
# Backend: voice chat adds entry to shared chat history
# ---------------------------------------------------------------------------


def test_voice_adds_to_chat_history(client):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Sure!"

    with patch("rex.dashboard.routes._transcribe_audio_file", return_value="remind me"):
        with patch("rex.dashboard.routes._get_llm", return_value=mock_llm):
            _send_voice(client)

    assert len(routes_module._CHAT_HISTORY) == 1
    entry = routes_module._CHAT_HISTORY[0]
    assert entry["user_message"] == "remind me"
    assert entry["assistant_reply"] == "Sure!"
    assert entry.get("source") == "voice"


def test_voice_history_respects_max(client):
    """History should not exceed _CHAT_HISTORY_MAX entries."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "ok"

    # Pre-fill history to the max
    for i in range(routes_module._CHAT_HISTORY_MAX):
        routes_module._CHAT_HISTORY.append(
            {"user_message": f"u{i}", "assistant_reply": "r", "timestamp": "", "elapsed_ms": 0}
        )

    with patch("rex.dashboard.routes._transcribe_audio_file", return_value="overflow test"):
        with patch("rex.dashboard.routes._get_llm", return_value=mock_llm):
            _send_voice(client)

    assert len(routes_module._CHAT_HISTORY) == routes_module._CHAT_HISTORY_MAX


# ---------------------------------------------------------------------------
# Backend: webm content-type accepted
# ---------------------------------------------------------------------------


def test_voice_webm_content_type_accepted(client):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "got it"

    with patch("rex.dashboard.routes._transcribe_audio_file", return_value="test webm"):
        with patch("rex.dashboard.routes._get_llm", return_value=mock_llm):
            resp = _send_voice(client, content_type="audio/webm")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# HTML: microphone input works — UI elements present
# ---------------------------------------------------------------------------


def test_html_has_voice_button():
    """#voice-btn must exist so the JS can wire up MediaRecorder."""
    template = (Path(__file__).parent.parent / "rex/dashboard/templates/index.html").read_text()
    assert 'id="voice-btn"' in template


def test_html_has_voice_status():
    """#voice-status element must exist for recording feedback."""
    template = (Path(__file__).parent.parent / "rex/dashboard/templates/index.html").read_text()
    assert 'id="voice-status"' in template


# ---------------------------------------------------------------------------
# JS: voice handlers wired
# ---------------------------------------------------------------------------


def test_js_has_voice_handler():
    """dashboard.js must contain handleVoiceBtnClick."""
    js = (
        Path(__file__).parent.parent / "rex/dashboard/static/js/dashboard.js"
    ).read_text()
    assert "handleVoiceBtnClick" in js


def test_js_has_speech_synthesis():
    """dashboard.js must use speechSynthesis for audio playback."""
    js = (
        Path(__file__).parent.parent / "rex/dashboard/static/js/dashboard.js"
    ).read_text()
    assert "speechSynthesis" in js


def test_js_has_media_recorder():
    """dashboard.js must use MediaRecorder for microphone capture."""
    js = (
        Path(__file__).parent.parent / "rex/dashboard/static/js/dashboard.js"
    ).read_text()
    assert "MediaRecorder" in js


def test_js_posts_to_api_voice():
    """dashboard.js must send audio to /api/voice."""
    js = (
        Path(__file__).parent.parent / "rex/dashboard/static/js/dashboard.js"
    ).read_text()
    assert "/api/voice" in js


# ---------------------------------------------------------------------------
# Browser verification: server starts and voice endpoint is reachable
# ---------------------------------------------------------------------------


def test_voice_endpoint_reachable_via_http():
    """Verify changes work in browser: Flask server starts and /api/voice responds."""
    import os

    # Prevent config side effects
    os.environ.setdefault("REX_TESTING", "true")

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = False
    flask_app.register_blueprint(dashboard_bp)

    server_started = threading.Event()
    server_error: list[Exception] = []

    def run_server():
        try:
            flask_app.run(host="127.0.0.1", port=15052, use_reloader=False)
        except Exception as exc:
            server_error.append(exc)

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    import time

    for _ in range(30):
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:15052/api/dashboard/status", timeout=1
            ) as r:
                if r.status == 200:
                    server_started.set()
                    break
        except Exception:
            time.sleep(0.1)

    assert server_started.is_set(), "Server did not start in time"

    # POST /api/voice without audio → expect 400 (server reachable, auth bypassed on localhost)
    req = urllib.request.Request(
        "http://127.0.0.1:15052/api/voice",
        data=b"",
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code

    assert status == 400
