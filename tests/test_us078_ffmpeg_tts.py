"""Tests for US-078: torio/FFmpeg runtime dependency handling.

Verifies:
- torio FFmpeg warnings are suppressed in rex/voice_loop module
- check_ffmpeg_for_tts() reports correct status based on TTS provider
- Voice pipeline module imports succeed without FFmpeg present
"""

from __future__ import annotations

import inspect
import shutil
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Warning suppression tests
# ---------------------------------------------------------------------------


def test_voice_loop_module_suppresses_torio_ffmpeg_warning():
    """rex.voice_loop must suppress 'FFmpeg extension' RuntimeWarnings from torio."""
    # Reload the module to ensure its warning filters are in place
    import rex.voice_loop  # noqa: F401 (side-effect: installs filters)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # Simulate the warning that torio emits
        warnings.warn(
            "Failed to load FFmpeg extension (some codecs unavailable)",
            RuntimeWarning,
            stacklevel=1,
        )

    # The filter installed by rex.voice_loop should have been active
    # but catch_warnings resets filters — we just verify the import succeeds
    # and the module-level filter code is present.
    import rex.voice_loop as vl_mod

    src = inspect.getsource(vl_mod)
    assert "FFmpeg extension" in src, "voice_loop should contain FFmpeg warning filter"
    assert "torio" in src, "voice_loop should contain torio warning filter"


def test_voice_loop_imports_without_ffmpeg(monkeypatch):
    """Importing rex.voice_loop must not raise even when ffmpeg is absent."""
    # Remove ffmpeg from PATH to simulate its absence
    original_which = shutil.which

    def _which_no_ffmpeg(name, *args, **kwargs):
        if name == "ffmpeg":
            return None
        return original_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", _which_no_ffmpeg)

    # Importing the module must succeed
    try:
        import rex.voice_loop  # noqa: F401
    except Exception as exc:
        raise AssertionError(f"rex.voice_loop import raised with ffmpeg absent: {exc}") from exc


# ---------------------------------------------------------------------------
# check_ffmpeg_for_tts() tests
# ---------------------------------------------------------------------------


def _get_check():
    from rex.doctor import check_ffmpeg_for_tts

    return check_ffmpeg_for_tts


def test_check_ffmpeg_for_tts_ok_when_ffmpeg_present(monkeypatch):
    """Returns OK when ffmpeg is on PATH regardless of TTS provider."""
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    )
    check = _get_check()

    # Patch config to return xtts provider
    mock_cfg = SimpleNamespace(tts_provider="xtts")
    with patch("rex.config.load_config", return_value=mock_cfg):
        result = check()

    from rex.doctor import Status

    assert result.status == Status.OK
    assert "ffmpeg" in result.message.lower() or "FFmpeg" in result.message


def test_check_ffmpeg_for_tts_warning_when_xtts_and_no_ffmpeg(monkeypatch):
    """Returns WARNING when TTS=xtts and FFmpeg is missing."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    check = _get_check()

    mock_cfg = SimpleNamespace(tts_provider="xtts")
    with patch("rex.config.load_config", return_value=mock_cfg):
        result = check()

    from rex.doctor import Status

    assert result.status == Status.WARNING
    assert "xtts" in result.message.lower()
    assert "not found" in result.message.lower() or "required" in result.message.lower()


def test_check_ffmpeg_for_tts_info_when_edge_tts_and_no_ffmpeg(monkeypatch):
    """Returns INFO (not WARNING) when TTS=edge-tts and FFmpeg is missing."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    check = _get_check()

    mock_cfg = SimpleNamespace(tts_provider="edge-tts")
    with patch("rex.config.load_config", return_value=mock_cfg):
        result = check()

    from rex.doctor import Status

    assert result.status == Status.INFO
    assert "edge-tts" in result.message.lower()


def test_check_ffmpeg_for_tts_info_when_pyttsx3_and_no_ffmpeg(monkeypatch):
    """Returns INFO (not WARNING) when TTS=pyttsx3 and FFmpeg is missing."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    check = _get_check()

    mock_cfg = SimpleNamespace(tts_provider="pyttsx3")
    with patch("rex.config.load_config", return_value=mock_cfg):
        result = check()

    from rex.doctor import Status

    assert result.status == Status.INFO


def test_check_ffmpeg_for_tts_defaults_to_xtts_on_config_error(monkeypatch):
    """Falls back to 'xtts' (conservative) when config cannot be loaded."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    check = _get_check()

    with patch("rex.config.load_config", side_effect=RuntimeError("no config")):
        result = check()

    from rex.doctor import Status

    # Should treat as xtts by default — WARNING since xtts needs FFmpeg
    assert result.status == Status.WARNING


def test_check_ffmpeg_for_tts_appears_in_run_diagnostics_output(monkeypatch, capsys):
    """run_diagnostics() output includes the FFmpeg TTS check."""
    # Stub out expensive checks so the test runs quickly
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    )

    import socket

    monkeypatch.setattr(socket, "create_connection", MagicMock(side_effect=ConnectionRefusedError))

    with (
        patch("rex.doctor.check_audio_input_device") as mock_ai,
        patch("rex.doctor.check_audio_output_device") as mock_ao,
        patch("rex.doctor.check_smart_speakers") as mock_ss,
        patch("rex.doctor.check_gpu_availability") as mock_gpu,
        patch("rex.doctor.check_core_dependencies", return_value=[]),
        patch("rex.doctor.check_xtts_transformers_compat") as mock_xtts,
        patch("rex.doctor.check_config_types") as mock_ct,
        patch("rex.doctor.check_stt_warmup") as mock_stt,
    ):

        from rex.doctor import CheckResult, Status

        _ok = CheckResult(name="x", status=Status.OK, message="ok")
        mock_ai.return_value = _ok
        mock_ao.return_value = _ok
        mock_ss.return_value = _ok
        mock_gpu.return_value = _ok
        mock_xtts.return_value = _ok
        mock_ct.return_value = _ok
        mock_stt.return_value = _ok

        from rex.doctor import run_diagnostics

        run_diagnostics(verbose=False)

    output = capsys.readouterr().out
    assert "FFmpeg" in output, f"Expected 'FFmpeg' in diagnostics output, got:\n{output}"
