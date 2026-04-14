"""Tests for US-302: XTTS PyTorch 2.6 safe-globals allowlist."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# apply_xtts_safe_globals
# ---------------------------------------------------------------------------


def test_apply_xtts_safe_globals_returns_false_when_tts_not_installed():
    """Returns False when Coqui TTS is not installed."""
    from rex.tts_utils import apply_xtts_safe_globals

    with patch("rex.tts_utils.find_spec", return_value=None):
        result = apply_xtts_safe_globals()

    assert result is False


def test_apply_xtts_safe_globals_returns_false_when_torch_not_installed():
    """Returns False when torch is not installed."""
    from rex.tts_utils import apply_xtts_safe_globals

    def _fake_find_spec(name: str) -> object:
        if name == "torch":
            return None
        return MagicMock()

    with patch("rex.tts_utils.find_spec", side_effect=_fake_find_spec):
        result = apply_xtts_safe_globals()

    assert result is False


def test_apply_xtts_safe_globals_registers_xtts_config_and_audio_config():
    """Registers XttsConfig and XttsAudioConfig with torch.serialization."""
    from rex.tts_utils import apply_xtts_safe_globals

    mock_xtts_config = MagicMock(name="XttsConfig")
    mock_xtts_audio_config = MagicMock(name="XttsAudioConfig")

    mock_tts_module = MagicMock()
    mock_tts_module.XttsConfig = mock_xtts_config
    mock_tts_module.XttsAudioConfig = mock_xtts_audio_config

    mock_torch = MagicMock()
    registered: list[list] = []

    def _capture_safe_globals(classes: list) -> None:
        registered.append(classes)

    mock_torch.serialization.add_safe_globals.side_effect = _capture_safe_globals

    with patch("rex.tts_utils.find_spec", return_value=MagicMock()):
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "TTS": MagicMock(),
                "TTS.tts": MagicMock(),
                "TTS.tts.configs": MagicMock(),
                "TTS.tts.configs.xtts_config": mock_tts_module,
            },
        ):
            result = apply_xtts_safe_globals()

    assert result is True
    assert registered, "add_safe_globals was never called"
    registered_classes = registered[0]
    assert mock_xtts_config in registered_classes, "XttsConfig not in safe globals"
    assert mock_xtts_audio_config in registered_classes, "XttsAudioConfig not in safe globals"


def test_apply_xtts_safe_globals_called_before_torch_load_in_initialize_xtts():
    """apply_xtts_safe_globals is imported and called inside _initialize_xtts."""
    import inspect

    from rex.voice_loop import TextToSpeech

    source = inspect.getsource(TextToSpeech._initialize_xtts)
    assert (
        "apply_xtts_safe_globals" in source
    ), "_initialize_xtts must call apply_xtts_safe_globals() before loading the model"


# ---------------------------------------------------------------------------
# get_tts_engine
# ---------------------------------------------------------------------------


def test_get_tts_engine_raises_import_error_when_tts_not_installed():
    """Raises ImportError with a clear message when TTS is not installed."""
    from rex.tts_utils import get_tts_engine

    with patch("rex.tts_utils.find_spec", return_value=None):
        try:
            get_tts_engine("xtts")
            raise AssertionError("Expected ImportError was not raised")
        except ImportError as exc:
            assert "Coqui TTS" in str(exc) or "TTS" in str(
                exc
            ), f"ImportError message should mention TTS, got: {exc}"


def test_get_tts_engine_raises_value_error_for_unknown_engine():
    """Raises ValueError for unrecognised engine names."""
    from rex.tts_utils import get_tts_engine

    try:
        get_tts_engine("unknown_engine_xyz")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        assert "unknown_engine_xyz" in str(exc)


def test_get_tts_engine_xtts_returns_tts_class_when_installed():
    """Returns TTS class when Coqui TTS is available."""
    from rex.tts_utils import get_tts_engine

    mock_tts_class = MagicMock(name="TTS")
    mock_api_module = MagicMock()
    mock_api_module.TTS = mock_tts_class

    with patch("rex.tts_utils.find_spec", return_value=MagicMock()):
        with patch("rex.tts_utils.apply_xtts_safe_globals", return_value=True):
            with patch.dict(
                sys.modules,
                {
                    "TTS": MagicMock(),
                    "TTS.api": mock_api_module,
                },
            ):
                result = get_tts_engine("xtts")

    assert result is mock_tts_class


def test_get_tts_engine_calls_apply_xtts_safe_globals():
    """get_tts_engine calls apply_xtts_safe_globals before returning."""
    from rex.tts_utils import get_tts_engine

    mock_api_module = MagicMock()
    mock_api_module.TTS = MagicMock()

    safe_globals_called: list[bool] = []

    def _track_call() -> bool:
        safe_globals_called.append(True)
        return True

    with patch("rex.tts_utils.find_spec", return_value=MagicMock()):
        with patch("rex.tts_utils.apply_xtts_safe_globals", side_effect=_track_call):
            with patch.dict(
                sys.modules,
                {
                    "TTS": MagicMock(),
                    "TTS.api": mock_api_module,
                },
            ):
                get_tts_engine("xtts")

    assert safe_globals_called, "apply_xtts_safe_globals was not called by get_tts_engine"


def test_apply_xtts_safe_globals_in_tts_utils_public_api():
    """apply_xtts_safe_globals and get_tts_engine are exported in __all__."""
    from rex import tts_utils

    assert "apply_xtts_safe_globals" in tts_utils.__all__
    assert "get_tts_engine" in tts_utils.__all__
