"""Tests for US-SP-002: Route TTS output to selected smart speaker."""

from __future__ import annotations

import wave
from unittest.mock import MagicMock, patch

from rex.audio.smart_speaker_output import (
    SmartSpeakerOutput,
    _get_local_ip,
    _wav_duration_seconds,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_wav_file(tmp_path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
    """Write a minimal WAV file and return its path."""
    path = tmp_path / "test.wav"
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return str(path)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_wav_duration_seconds(tmp_path):
    path = _make_wav_file(tmp_path, duration_seconds=2.0)
    assert abs(_wav_duration_seconds(path) - 2.0) < 0.01


def test_wav_duration_seconds_missing_file():
    result = _wav_duration_seconds("/nonexistent/file.wav")
    assert result == 5.0


def test_get_local_ip_returns_string():
    ip = _get_local_ip()
    assert isinstance(ip, str)
    assert "." in ip


# ---------------------------------------------------------------------------
# SmartSpeakerOutput.play_wav — unknown provider
# ---------------------------------------------------------------------------


def test_play_wav_unknown_provider_returns_false(tmp_path):
    wav = _make_wav_file(tmp_path)
    result = SmartSpeakerOutput().play_wav(wav, provider="alexa", ip="192.168.1.1")
    assert result is False


# ---------------------------------------------------------------------------
# Sonos
# ---------------------------------------------------------------------------


def test_play_wav_sonos_no_soco_installed(tmp_path):
    """Returns False gracefully when soco is not installed."""
    wav = _make_wav_file(tmp_path)
    with patch("rex.audio.smart_speaker_output.importlib") as mock_il:
        mock_il.import_module.side_effect = ImportError("soco not installed")
        result = SmartSpeakerOutput().play_wav(wav, provider="sonos", ip="192.168.1.10")
    assert result is False


def test_play_wav_sonos_device_unreachable(tmp_path):
    """Returns False when soco.SoCo raises an exception."""
    wav = _make_wav_file(tmp_path, duration_seconds=0.1)
    mock_soco = MagicMock()
    mock_soco.SoCo.return_value.play_uri.side_effect = Exception("connection refused")
    fake_server = MagicMock()

    with (
        patch("rex.audio.smart_speaker_output.importlib") as mock_il,
        patch(
            "rex.audio.smart_speaker_output._start_http_server",
            return_value=(fake_server, "http://192.168.1.5:50000/audio.wav"),
        ),
        patch("rex.audio.smart_speaker_output._wav_duration_seconds", return_value=0.0),
    ):
        mock_il.import_module.return_value = mock_soco
        result = SmartSpeakerOutput().play_wav(wav, provider="sonos", ip="192.168.1.10")

    assert result is False
    fake_server.shutdown.assert_called_once()


def test_play_wav_sonos_success(tmp_path):
    """Returns True and shuts down the HTTP server after successful Sonos play."""
    wav = _make_wav_file(tmp_path, duration_seconds=0.1)
    mock_soco = MagicMock()
    fake_server = MagicMock()

    with (
        patch("rex.audio.smart_speaker_output.importlib") as mock_il,
        patch(
            "rex.audio.smart_speaker_output._start_http_server",
            return_value=(fake_server, "http://192.168.1.5:50000/audio.wav"),
        ),
        patch("rex.audio.smart_speaker_output._wav_duration_seconds", return_value=0.0),
        patch("time.sleep"),
    ):
        mock_il.import_module.return_value = mock_soco
        result = SmartSpeakerOutput().play_wav(wav, provider="sonos", ip="192.168.1.10")

    assert result is True
    mock_soco.SoCo.assert_called_once_with("192.168.1.10")
    mock_soco.SoCo.return_value.play_uri.assert_called_once()
    fake_server.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Bose SoundTouch
# ---------------------------------------------------------------------------


def test_play_wav_bose_no_requests_installed(tmp_path):
    """Returns False gracefully when requests is not installed."""
    wav = _make_wav_file(tmp_path)
    with patch("rex.audio.smart_speaker_output.importlib") as mock_il:
        mock_il.import_module.side_effect = ImportError("no requests")
        result = SmartSpeakerOutput().play_wav(wav, provider="bose", ip="192.168.1.20")
    assert result is False


def test_play_wav_bose_api_error(tmp_path):
    """Returns False when the SoundTouch REST call fails."""
    wav = _make_wav_file(tmp_path, duration_seconds=0.1)
    mock_requests = MagicMock()
    mock_requests.post.side_effect = Exception("connection refused")
    fake_server = MagicMock()

    with (
        patch("rex.audio.smart_speaker_output.importlib") as mock_il,
        patch(
            "rex.audio.smart_speaker_output._start_http_server",
            return_value=(fake_server, "http://192.168.1.5:50001/audio.wav"),
        ),
        patch("rex.audio.smart_speaker_output._wav_duration_seconds", return_value=0.0),
    ):
        mock_il.import_module.return_value = mock_requests
        result = SmartSpeakerOutput().play_wav(wav, provider="bose", ip="192.168.1.20")

    assert result is False
    fake_server.shutdown.assert_called_once()


def test_play_wav_bose_success(tmp_path):
    """Returns True after posting to the SoundTouch REST API."""
    wav = _make_wav_file(tmp_path, duration_seconds=0.1)
    mock_requests = MagicMock()
    mock_requests.post.return_value.raise_for_status = MagicMock()
    fake_server = MagicMock()

    with (
        patch("rex.audio.smart_speaker_output.importlib") as mock_il,
        patch(
            "rex.audio.smart_speaker_output._start_http_server",
            return_value=(fake_server, "http://192.168.1.5:50001/audio.wav"),
        ),
        patch("rex.audio.smart_speaker_output._wav_duration_seconds", return_value=0.0),
        patch("time.sleep"),
    ):
        mock_il.import_module.return_value = mock_requests
        result = SmartSpeakerOutput().play_wav(wav, provider="bose", ip="192.168.1.20")

    assert result is True
    mock_requests.post.assert_called_once()
    call_url = mock_requests.post.call_args.args[0]
    assert "192.168.1.20" in call_url
    assert "8090" in call_url
    assert "select" in call_url
    fake_server.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Integration with TextToSpeech._try_smart_speaker
# ---------------------------------------------------------------------------


def test_try_smart_speaker_no_device_returns_false(tmp_path):
    """_try_smart_speaker returns False when no output device is configured."""
    from rex.voice_loop import TextToSpeech

    with patch("rex.voice_loop.settings") as mock_settings:
        mock_settings.tts_speed = 1.0
        mock_settings.tts_provider = "xtts"
        mock_settings.tts_voice = None
        mock_settings.tts_output_device = None
        tts = TextToSpeech(language="en")

    wav = _make_wav_file(tmp_path)
    assert tts._try_smart_speaker(wav) is False


def test_try_smart_speaker_speaker_not_found_returns_false(tmp_path):
    """_try_smart_speaker returns False when the named speaker is not in the cache."""
    from rex.voice_loop import TextToSpeech

    with patch("rex.voice_loop.settings") as mock_settings:
        mock_settings.tts_speed = 1.0
        mock_settings.tts_provider = "xtts"
        mock_settings.tts_voice = None
        mock_settings.tts_output_device = "Living Room Sonos"
        tts = TextToSpeech(language="en")

    wav = _make_wav_file(tmp_path)

    with patch("rex.voice_loop.TextToSpeech._try_smart_speaker", wraps=tts._try_smart_speaker):
        with patch("rex.audio.speaker_discovery.get_speaker_discovery") as mock_disc:
            mock_disc.return_value.get_cached_speakers.return_value = []
            result = tts._try_smart_speaker(wav)

    assert result is False


def test_try_smart_speaker_routes_to_output(tmp_path):
    """_try_smart_speaker finds the speaker and delegates to SmartSpeakerOutput."""
    from rex.audio.speaker_discovery import DiscoveredSpeaker
    from rex.voice_loop import TextToSpeech

    with patch("rex.voice_loop.settings") as mock_settings:
        mock_settings.tts_speed = 1.0
        mock_settings.tts_provider = "xtts"
        mock_settings.tts_voice = None
        mock_settings.tts_output_device = "Living Room"
        tts = TextToSpeech(language="en")

    wav = _make_wav_file(tmp_path)
    speaker = DiscoveredSpeaker(
        provider="sonos", name="Living Room", ip="192.168.1.10", model="Play:1"
    )

    with (
        patch("rex.audio.speaker_discovery.get_speaker_discovery") as mock_disc,
        patch("rex.audio.smart_speaker_output.get_smart_speaker_output") as mock_sso,
    ):
        mock_disc.return_value.get_cached_speakers.return_value = [speaker]
        mock_sso.return_value.play_wav.return_value = True
        result = tts._try_smart_speaker(wav)

    assert result is True
    mock_sso.return_value.play_wav.assert_called_once_with(wav, provider="sonos", ip="192.168.1.10")
