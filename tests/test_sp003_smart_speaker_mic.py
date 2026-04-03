"""Tests for US-SP-003: Wake word detection from smart speaker microphone."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

from rex.audio.smart_speaker_mic import SmartSpeakerMic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm_bytes(n_samples: int, value: int = 0) -> bytes:
    """Return *n_samples* 16-bit little-endian PCM samples."""
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


# ---------------------------------------------------------------------------
# connect() — unsupported provider
# ---------------------------------------------------------------------------


def test_connect_unsupported_provider_returns_false():
    mic = SmartSpeakerMic(provider="bose", ip="192.168.1.20")
    assert mic.connect() is False
    assert mic._connected is False


# ---------------------------------------------------------------------------
# connect() — Sonos
# ---------------------------------------------------------------------------


def test_connect_sonos_no_requests_returns_false():
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10")
    with patch("rex.audio.smart_speaker_mic.importlib") as mock_il:
        mock_il.import_module.side_effect = ImportError("no requests")
        assert mic.connect() is False
    assert mic._connected is False


def test_connect_sonos_http_error_returns_false():
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10")
    mock_requests = MagicMock()
    mock_requests.get.side_effect = Exception("connection refused")
    with patch("rex.audio.smart_speaker_mic.importlib") as mock_il:
        mock_il.import_module.return_value = mock_requests
        assert mic.connect() is False
    assert mic._connected is False


def test_connect_sonos_http_401_returns_false():
    """HTTP errors (raise_for_status) also count as failed connection."""
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10")
    mock_requests = MagicMock()
    mock_requests.get.return_value.raise_for_status.side_effect = Exception("401 Unauthorized")
    with patch("rex.audio.smart_speaker_mic.importlib") as mock_il:
        mock_il.import_module.return_value = mock_requests
        assert mic.connect() is False
    assert mic._connected is False


def test_connect_sonos_success():
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10")
    mock_requests = MagicMock()
    with patch("rex.audio.smart_speaker_mic.importlib") as mock_il:
        mock_il.import_module.return_value = mock_requests
        result = mic.connect()

    assert result is True
    assert mic._connected is True
    mock_requests.get.assert_called_once()
    call_url = mock_requests.get.call_args.args[0]
    assert "192.168.1.10" in call_url
    assert "1400" in call_url
    assert "audio/capture" in call_url


# ---------------------------------------------------------------------------
# read_frame()
# ---------------------------------------------------------------------------


def test_read_frame_not_connected_returns_none():
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10")
    # _connected is False by default
    assert mic.read_frame(1.0) is None


def test_read_frame_returns_float32_array():
    """read_frame returns a normalised float32 numpy array."""
    import numpy as np

    sample_rate = 16000
    duration = 0.1
    n_samples = int(sample_rate * duration)
    pcm_bytes = _make_pcm_bytes(n_samples, value=16384)  # ~0.5 amplitude

    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10", sample_rate=sample_rate)
    mic._connected = True

    mock_response = MagicMock()
    mock_response.iter_content.return_value = iter([pcm_bytes])
    mic._response = mock_response

    with patch("rex.audio.smart_speaker_mic.importlib") as mock_il:
        mock_il.import_module.return_value = np
        result = mic.read_frame(duration)

    assert result is not None
    assert result.dtype == np.float32
    assert result.ndim == 1
    assert len(result) == n_samples
    # 16384 / 32768 ≈ 0.5
    assert abs(float(result[0]) - 0.5) < 0.01


def test_read_frame_stream_ends_early_returns_none_and_disconnects():
    """If the stream ends before we have enough bytes, disconnect and return None."""
    import numpy as np

    sample_rate = 16000
    duration = 1.0
    # Provide only 100 bytes when we need 32000 bytes
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10", sample_rate=sample_rate)
    mic._connected = True

    mock_response = MagicMock()
    mock_response.iter_content.return_value = iter([b"\x00" * 100])
    mic._response = mock_response

    with patch("rex.audio.smart_speaker_mic.importlib") as mock_il:
        mock_il.import_module.return_value = np
        result = mic.read_frame(duration)

    assert result is None
    assert mic._connected is False


# ---------------------------------------------------------------------------
# disconnect()
# ---------------------------------------------------------------------------


def test_disconnect_closes_response():
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10")
    mock_response = MagicMock()
    mic._response = mock_response
    mic._connected = True

    mic.disconnect()

    mock_response.close.assert_called_once()
    assert mic._connected is False
    assert mic._response is None


# ---------------------------------------------------------------------------
# build_voice_loop integration: smart mic wired when device configured
# ---------------------------------------------------------------------------


def _common_voice_loop_patches(mock_settings, wake_word_device: str | None = None):
    """Return a dict of common patch targets for build_voice_loop tests."""
    mock_settings.audio_input_device = None
    mock_settings.wake_word_input_device = wake_word_device
    mock_settings.acknowledgment_sound = "chime"
    mock_settings.tts_speed = 1.0
    mock_settings.tts_provider = "xtts"
    mock_settings.tts_voice = None
    mock_settings.tts_output_device = None
    mock_settings.use_openclaw_voice_backend = False


def test_build_voice_loop_uses_smart_mic_when_configured():
    """build_voice_loop passes SmartSpeakerMic.read_frame as recorder when configured."""

    from rex.audio.speaker_discovery import DiscoveredSpeaker

    speaker = DiscoveredSpeaker(provider="sonos", name="Office", ip="192.168.1.10", model="One")
    mock_smart_mic = MagicMock(spec=SmartSpeakerMic)
    mock_smart_mic.connect.return_value = True
    mock_smart_mic.read_frame = MagicMock()

    with (
        patch("rex.voice_loop.settings") as mock_settings,
        patch("rex.voice_loop._validate_input_device_index", return_value=None),
        patch("rex.wakeword.listener.build_default_detector"),
        patch("rex.voice_loop.SpeechToText"),
        patch("rex.voice_loop.TextToSpeech"),
        patch("rex.voice_loop.WakeAcknowledgement"),
        patch("rex.voice_loop._build_voice_id_callback", return_value=None),
        patch("rex.audio.speaker_discovery.get_speaker_discovery") as mock_disc,
        patch("rex.audio.smart_speaker_mic.SmartSpeakerMic", return_value=mock_smart_mic),
        patch("rex.voice_loop.AsyncMicrophone") as mock_async_mic,
    ):
        _common_voice_loop_patches(mock_settings, wake_word_device="Office")
        mock_disc.return_value.get_cached_speakers.return_value = [speaker]

        from rex.voice_loop import build_voice_loop

        build_voice_loop(MagicMock())

    # AsyncMicrophone should have been called with the smart mic's read_frame as recorder
    call_kwargs = mock_async_mic.call_args.kwargs
    assert call_kwargs["recorder"] is mock_smart_mic.read_frame


def test_build_voice_loop_falls_back_when_speaker_not_found():
    """build_voice_loop uses recorder=None (local mic) when speaker name not in cache."""

    with (
        patch("rex.voice_loop.settings") as mock_settings,
        patch("rex.voice_loop._validate_input_device_index", return_value=None),
        patch("rex.wakeword.listener.build_default_detector"),
        patch("rex.voice_loop.SpeechToText"),
        patch("rex.voice_loop.TextToSpeech"),
        patch("rex.voice_loop.WakeAcknowledgement"),
        patch("rex.voice_loop._build_voice_id_callback", return_value=None),
        patch("rex.audio.speaker_discovery.get_speaker_discovery") as mock_disc,
        patch("rex.voice_loop.AsyncMicrophone") as mock_async_mic,
    ):
        _common_voice_loop_patches(mock_settings, wake_word_device="Nonexistent Speaker")
        mock_disc.return_value.get_cached_speakers.return_value = []

        from rex.voice_loop import build_voice_loop

        build_voice_loop(MagicMock())

    call_kwargs = mock_async_mic.call_args.kwargs
    assert call_kwargs["recorder"] is None
