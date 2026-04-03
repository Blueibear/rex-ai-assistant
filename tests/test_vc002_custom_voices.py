"""Tests for US-VC-002: Personalized voice creation via audio file upload."""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(tmp_path: Path, duration_seconds: float = 10.0, sample_rate: int = 16000) -> Path:
    """Write a minimal WAV file and return its path."""
    path = tmp_path / "sample.wav"
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return path


# ---------------------------------------------------------------------------
# get_audio_duration
# ---------------------------------------------------------------------------


def test_get_audio_duration_wav(tmp_path: Path) -> None:
    path = _make_wav(tmp_path, duration_seconds=5.0)
    from rex.custom_voices import get_audio_duration

    dur = get_audio_duration(path)
    assert abs(dur - 5.0) < 0.1


def test_get_audio_duration_missing_file() -> None:
    from rex.custom_voices import get_audio_duration

    with pytest.raises(ValueError, match="not found"):
        get_audio_duration("/nonexistent/file.wav")


def test_get_audio_duration_invalid_wav(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"\x00" * 10)  # not a valid WAV
    from rex.custom_voices import get_audio_duration

    with pytest.raises(ValueError):
        get_audio_duration(bad)


# ---------------------------------------------------------------------------
# save_custom_voice
# ---------------------------------------------------------------------------


def test_save_custom_voice_too_short(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path, duration_seconds=3.0)
    voices_dir = tmp_path / "voices"
    from rex.custom_voices import save_custom_voice

    result = save_custom_voice(wav, "Short Voice", voices_dir=voices_dir)

    assert result["ok"] is False
    assert "too short" in result["error"].lower()
    assert result["duration"] == pytest.approx(3.0, abs=0.1)


def test_save_custom_voice_success(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path, duration_seconds=12.0)
    voices_dir = tmp_path / "voices"
    from rex.custom_voices import save_custom_voice

    result = save_custom_voice(wav, "My Custom Voice", voices_dir=voices_dir)

    assert result["ok"] is True
    assert "voice_id" in result
    assert result["duration"] == pytest.approx(12.0, abs=0.1)
    saved_path = Path(result["voice_id"])
    assert saved_path.exists()
    assert saved_path.suffix == ".wav"
    assert saved_path.parent == voices_dir


def test_save_custom_voice_creates_voices_dir(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path, duration_seconds=11.0)
    voices_dir = tmp_path / "nested" / "voices"
    assert not voices_dir.exists()
    from rex.custom_voices import save_custom_voice

    result = save_custom_voice(wav, "Test Voice", voices_dir=voices_dir)
    assert result["ok"] is True
    assert voices_dir.exists()


def test_save_custom_voice_name_sanitization(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path, duration_seconds=10.5)
    voices_dir = tmp_path / "voices"
    from rex.custom_voices import save_custom_voice

    result = save_custom_voice(wav, "My Voice! #1", voices_dir=voices_dir)
    assert result["ok"] is True
    saved = Path(result["voice_id"])
    # Name should be filesystem-safe
    assert " " not in saved.name
    assert "#" not in saved.name


def test_save_custom_voice_missing_source(tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    from rex.custom_voices import save_custom_voice

    result = save_custom_voice("/nonexistent/audio.wav", "Test", voices_dir=voices_dir)
    assert result["ok"] is False
    assert "error" in result


def test_save_custom_voice_exactly_10s(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path, duration_seconds=10.0)
    voices_dir = tmp_path / "voices"
    from rex.custom_voices import save_custom_voice

    result = save_custom_voice(wav, "Exact Ten", voices_dir=voices_dir)
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


def test_sanitize_name_basic() -> None:
    from rex.custom_voices import _sanitize_name

    assert _sanitize_name("My Voice") == "my_voice"


def test_sanitize_name_special_chars() -> None:
    from rex.custom_voices import _sanitize_name

    result = _sanitize_name("Hello! World #2")
    assert " " not in result
    assert "!" not in result
    assert "#" not in result


def test_sanitize_name_empty_falls_back() -> None:
    from rex.custom_voices import _sanitize_name

    assert _sanitize_name("!!!") == "custom_voice"


# ---------------------------------------------------------------------------
# Bridge script (black-box via subprocess)
# ---------------------------------------------------------------------------


def test_bridge_missing_file_path(tmp_path: Path) -> None:
    """Bridge returns error when file_path is absent."""
    bridge = Path(__file__).resolve().parent.parent / "rex_voice_upload_bridge.py"
    result = subprocess.run(
        [sys.executable, str(bridge)],
        input=json.dumps({"voice_name": "Test"}),
        capture_output=True,
        text=True,
    )
    out = json.loads(result.stdout.strip())
    assert out["ok"] is False


def test_bridge_nonexistent_file(tmp_path: Path) -> None:
    """Bridge returns error for a nonexistent audio file."""
    bridge = Path(__file__).resolve().parent.parent / "rex_voice_upload_bridge.py"
    result = subprocess.run(
        [sys.executable, str(bridge)],
        input=json.dumps({"file_path": "/nonexistent/audio.wav", "voice_name": "Test"}),
        capture_output=True,
        text=True,
    )
    out = json.loads(result.stdout.strip())
    assert out["ok"] is False


def test_bridge_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bridge returns ok=True for a valid WAV with custom voices_dir patched."""
    wav = _make_wav(tmp_path, duration_seconds=12.0)
    voices_dir = tmp_path / "voices"

    # We can't easily patch the voices_dir inside the subprocess, so instead
    # run the bridge in-process by importing directly.
    import contextlib
    import io
    import sys as _sys

    with patch("rex.custom_voices._VOICES_DIR", voices_dir):
        captured_out = io.StringIO()
        original_stdin = _sys.stdin
        _sys.stdin = io.StringIO(json.dumps({"file_path": str(wav), "voice_name": "My Voice"}))
        try:
            with contextlib.redirect_stdout(captured_out):
                import rex_voice_upload_bridge

                rex_voice_upload_bridge.main()
        except SystemExit:
            pass
        finally:
            _sys.stdin = original_stdin

    out = json.loads(captured_out.getvalue().strip())
    assert out["ok"] is True
    assert "voice_id" in out
    assert Path(out["voice_id"]).exists()
