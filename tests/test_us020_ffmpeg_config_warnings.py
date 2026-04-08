"""Tests for US-020: Fix FFmpeg/torio errors and config coercion warnings.

AC:
- [1] If FFmpeg is not on PATH, a clear warning is logged at startup (not a crash)
- [2] rex doctor checks for FFmpeg and reports its presence/version (see US-078)
- [3] Config values that trigger coercion warnings are fixed to use correct types
- [4] No UserWarning or DeprecationWarning from config loading
- [5] Test confirms config loads without warnings
- [6] Typecheck passes
"""

from __future__ import annotations

import shutil
import warnings
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# AC #4 / AC #5: Config loading emits no UserWarning or DeprecationWarning
# ---------------------------------------------------------------------------


def test_config_load_no_user_warning() -> None:
    """load_config() must not emit UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        from rex.config import load_config  # noqa: PLC0415

        load_config()  # must not raise


def test_config_load_no_deprecation_warning() -> None:
    """load_config() must not emit DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from rex.config import load_config  # noqa: PLC0415

        load_config()  # must not raise


def test_config_load_returns_app_config() -> None:
    """load_config() returns a valid AppConfig with no warnings."""
    from rex.config import AppConfig, load_config  # noqa: PLC0415

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config()
    user_or_deprecation = [
        w for w in caught if issubclass(w.category, (UserWarning, DeprecationWarning))
    ]
    assert isinstance(cfg, AppConfig)
    assert (
        user_or_deprecation == []
    ), f"Unexpected warnings from load_config(): {user_or_deprecation}"


# ---------------------------------------------------------------------------
# AC #1: FFmpeg-absent warning logged at startup (build_voice_loop)
# ---------------------------------------------------------------------------


def _setup_mock_settings(mock_settings) -> None:
    """Configure mock settings attributes needed by build_voice_loop."""
    mock_settings.audio_input_device = None
    mock_settings.wake_word_input_device = None
    mock_settings.acknowledgment_sound = "chime"
    mock_settings.tts_speed = 1.0
    mock_settings.tts_provider = "xtts"
    mock_settings.tts_voice = None
    mock_settings.tts_output_device = None
    mock_settings.use_openclaw_voice_backend = False


def _run_build_voice_loop_with_tts_provider(tts_provider: str, ffmpeg_found: bool, caplog) -> list:
    """Run build_voice_loop with a stubbed TTS provider and return ffmpeg_missing log records.

    numpy is not installed in the test environment, so rex.wakeword.listener cannot be
    imported normally.  We inject a mock module into sys.modules to satisfy the local
    ``from .wakeword.listener import build_default_detector`` call inside build_voice_loop.
    """
    import logging  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    import rex.voice_loop as vl  # noqa: PLC0415

    mock_tts = MagicMock()
    mock_tts._provider = tts_provider
    ffmpeg_return = "/usr/bin/ffmpeg" if ffmpeg_found else None

    # Build minimal stub modules for the numpy-dependent wakeword package.
    mock_listener_mod = types.ModuleType("rex.wakeword.listener")
    mock_listener_mod.build_default_detector = MagicMock()  # type: ignore[attr-defined]
    mock_listener_mod.WakeWordListener = MagicMock()  # type: ignore[attr-defined]

    mock_wakeword_mod = types.ModuleType("rex.wakeword")
    mock_wakeword_mod.listener = mock_listener_mod  # type: ignore[attr-defined]

    stub_modules = {
        "rex.wakeword": mock_wakeword_mod,
        "rex.wakeword.listener": mock_listener_mod,
    }
    # Preserve existing entries (e.g. if numpy somehow available later)
    original = {k: sys.modules.get(k) for k in stub_modules}
    sys.modules.update(stub_modules)
    try:
        with (
            patch("rex.voice_loop.settings") as mock_settings,
            patch("rex.voice_loop._validate_input_device_index", return_value=0),
            patch("rex.voice_loop.AsyncMicrophone"),
            patch("rex.voice_loop.SpeechToText"),
            patch("rex.voice_loop.TextToSpeech", return_value=mock_tts),
            patch("rex.voice_loop.WakeAcknowledgement"),
            patch("rex.voice_loop._build_voice_id_callback", return_value=None),
            patch("rex.voice_loop.VoiceLoop"),
            patch.object(shutil, "which", return_value=ffmpeg_return),
            caplog.at_level(logging.WARNING, logger="rex.voice_loop"),
        ):
            _setup_mock_settings(mock_settings)
            vl.build_voice_loop(MagicMock())
    finally:
        for k, v in original.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return [r for r in caplog.records if getattr(r, "event", None) == "ffmpeg_missing"]


def test_ffmpeg_missing_logs_warning_when_xtts(caplog: pytest.LogCaptureFixture) -> None:
    """When TTS provider is 'xtts' and ffmpeg is absent, a WARNING is logged."""
    records = _run_build_voice_loop_with_tts_provider("xtts", ffmpeg_found=False, caplog=caplog)
    assert records, "Expected a WARNING log event 'ffmpeg_missing' when FFmpeg absent"
    assert records[0].levelname == "WARNING"


def test_ffmpeg_present_no_warning_when_xtts(caplog: pytest.LogCaptureFixture) -> None:
    """When TTS provider is 'xtts' and ffmpeg IS found, no ffmpeg_missing event is logged."""
    records = _run_build_voice_loop_with_tts_provider("xtts", ffmpeg_found=True, caplog=caplog)
    assert not records, "Should NOT log ffmpeg_missing when FFmpeg is present"


def test_ffmpeg_missing_no_warning_for_edge_tts(caplog: pytest.LogCaptureFixture) -> None:
    """When TTS provider is 'edge-tts' and ffmpeg is absent, no ffmpeg warning is logged."""
    records = _run_build_voice_loop_with_tts_provider("edge-tts", ffmpeg_found=False, caplog=caplog)
    assert not records, "Should NOT log ffmpeg_missing for edge-tts provider"


# ---------------------------------------------------------------------------
# AC #3: Config coercion: string values logged as LOGGER.warning, not exceptions
#         for valid parseable strings; invalid strings raise ConfigurationError
# ---------------------------------------------------------------------------


def test_coerce_float_invalid_raises_configuration_error() -> None:
    """_coerce_float with non-parseable string raises ConfigurationError."""
    from rex.config import ConfigurationError, _coerce_float  # noqa: PLC0415

    with pytest.raises(ConfigurationError, match="invalid value"):
        _coerce_float({"llm": {"temperature": "abc"}}, "llm.temperature", 0.7)


def test_coerce_int_invalid_raises_configuration_error() -> None:
    """_coerce_int with non-parseable string raises ConfigurationError."""
    from rex.config import ConfigurationError, _coerce_int  # noqa: PLC0415

    with pytest.raises(ConfigurationError, match="invalid value"):
        _coerce_int({"tts": {"rate": "xyz"}}, "tts.rate", 150)
