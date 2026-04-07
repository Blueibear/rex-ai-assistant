"""Tests for US-012: Fix custom voice duration validation."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from rex.custom_voices import (
    ACCEPTED_FORMATS,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    get_audio_duration,
    save_custom_voice,
)


def _make_wav(path: Path, duration_seconds: float, sample_rate: int = 16000) -> Path:
    """Write a silent WAV file of exactly *duration_seconds* length."""
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return path


# ---------------------------------------------------------------------------
# get_audio_duration — WAV files
# ---------------------------------------------------------------------------


def test_duration_short_wav(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "short.wav", duration_seconds=5.0)
    assert abs(get_audio_duration(wav) - 5.0) < 0.01


def test_duration_exact_minimum_wav(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "exact.wav", duration_seconds=MIN_DURATION_SECONDS)
    assert abs(get_audio_duration(wav) - MIN_DURATION_SECONDS) < 0.01


def test_duration_long_wav(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "long.wav", duration_seconds=30.0)
    assert abs(get_audio_duration(wav) - 30.0) < 0.01


# ---------------------------------------------------------------------------
# get_audio_duration — unsupported format
# ---------------------------------------------------------------------------


def test_unsupported_format_names_extension(tmp_path: Path) -> None:
    bad = tmp_path / "audio.xyz"
    bad.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError) as exc_info:
        get_audio_duration(bad)
    msg = str(exc_info.value)
    assert "XYZ" in msg, f"Format label missing from: {msg}"
    # Must list accepted formats
    for fmt in ACCEPTED_FORMATS:
        assert fmt.lstrip(".").upper() in msg, f"{fmt} missing from: {msg}"


def test_unsupported_format_no_extension(tmp_path: Path) -> None:
    bad = tmp_path / "audiofile"
    bad.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError) as exc_info:
        get_audio_duration(bad)
    msg = str(exc_info.value)
    assert "Unsupported format" in msg


# ---------------------------------------------------------------------------
# save_custom_voice — too short
# ---------------------------------------------------------------------------


def test_too_short_message_format(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "short.wav", duration_seconds=3.5)
    result = save_custom_voice(wav, "TestVoice", voices_dir=tmp_path / "voices")
    assert result["ok"] is False
    assert "Sample is 3.5s, minimum is" in result["error"]
    assert str(int(MIN_DURATION_SECONDS)) in result["error"]


def test_too_short_duration_field(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "short.wav", duration_seconds=2.0)
    result = save_custom_voice(wav, "Short", voices_dir=tmp_path / "voices")
    assert result["duration"] == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------------------
# save_custom_voice — too long
# ---------------------------------------------------------------------------


def test_too_long_message_format(tmp_path: Path) -> None:
    # Create a WAV that exceeds MAX_DURATION_SECONDS by patching get_audio_duration.
    from unittest.mock import patch

    wav = _make_wav(tmp_path / "long.wav", duration_seconds=10.0)
    fake_duration = MAX_DURATION_SECONDS + 60.0

    with patch("rex.custom_voices.get_audio_duration", return_value=fake_duration):
        result = save_custom_voice(wav, "Long", voices_dir=tmp_path / "voices")

    assert result["ok"] is False
    assert f"Sample is {fake_duration:.1f}s, maximum is" in result["error"]
    assert str(int(MAX_DURATION_SECONDS)) in result["error"]


# ---------------------------------------------------------------------------
# save_custom_voice — valid sample
# ---------------------------------------------------------------------------


def test_valid_sample_saves_successfully(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "voice.wav", duration_seconds=15.0)
    voices_dir = tmp_path / "voices"
    result = save_custom_voice(wav, "My Voice", voices_dir=voices_dir)
    assert result["ok"] is True
    assert Path(result["voice_id"]).exists()
    assert result["duration"] == pytest.approx(15.0, abs=0.05)
