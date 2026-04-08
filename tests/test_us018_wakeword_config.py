"""Tests for US-018: Fix wake word config mismatch and empty resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# AC #1: empty / None wake word resolves to "hey_rex"
# ---------------------------------------------------------------------------


def test_appconfig_default_wakeword():
    """AppConfig.wakeword defaults to 'hey_rex'."""
    from rex.config import AppConfig

    cfg = AppConfig()
    assert cfg.wakeword == "hey_rex"


def test_load_config_empty_wakeword_uses_hey_rex():
    """load_config normalises empty-string wake_word.wakeword to 'hey_rex'."""
    from rex.config import load_config

    cfg = load_config(json_config={"wake_word": {"wakeword": ""}}, reload=True)
    assert cfg.wakeword == "hey_rex"


def test_load_config_none_wakeword_uses_hey_rex():
    """load_config normalises null wake_word.wakeword to 'hey_rex'."""
    from rex.config import load_config

    cfg = load_config(json_config={"wake_word": {"wakeword": None}}, reload=True)
    assert cfg.wakeword == "hey_rex"


def test_load_config_valid_wakeword_preserved():
    """load_config preserves an explicitly set non-empty wake word."""
    from rex.config import load_config

    cfg = load_config(json_config={"wake_word": {"wakeword": "hey_computer"}}, reload=True)
    assert cfg.wakeword == "hey_computer"


# ---------------------------------------------------------------------------
# AC #2: missing model file raises WakeWordError with clear message
# ---------------------------------------------------------------------------


def test_build_default_detector_missing_model_file(tmp_path):
    """build_default_detector raises WakeWordError when model_path doesn't exist."""
    pytest.importorskip("numpy")

    from rex.assistant_errors import WakeWordError
    from rex.wakeword.listener import build_default_detector

    missing = str(tmp_path / "nonexistent.onnx")
    with pytest.raises(WakeWordError, match="not found"):
        build_default_detector(
            sample_rate=16000,
            chunk_duration=1.0,
            model_path=missing,
        )


def test_build_default_detector_empty_keyword_raises():
    """build_default_detector raises WakeWordError for empty keyword string."""
    pytest.importorskip("numpy")

    from rex.assistant_errors import WakeWordError
    from rex.wakeword.listener import build_default_detector

    with pytest.raises(WakeWordError, match="must not be empty"):
        build_default_detector(
            sample_rate=16000,
            chunk_duration=1.0,
            keyword="",
        )


# ---------------------------------------------------------------------------
# AC #3: rex doctor check_wakeword_config
# ---------------------------------------------------------------------------


def _mock_config(wakeword="hey_rex", wakeword_model_path=None):
    cfg = MagicMock()
    cfg.wakeword = wakeword
    cfg.wakeword_model_path = wakeword_model_path
    return cfg


def _patch_load_config(mock_cfg):
    """Context manager that patches load_config inside rex.config module."""
    import rex.config as config_mod

    return patch.object(config_mod, "load_config", return_value=mock_cfg)


def test_check_wakeword_config_ok_no_model_path():
    """check_wakeword_config returns OK when keyword is set and no custom model path."""
    from rex.doctor import Status, check_wakeword_config

    with _patch_load_config(_mock_config()):
        result = check_wakeword_config()

    assert result.status == Status.OK
    assert "hey_rex" in result.message


def test_check_wakeword_config_error_empty_keyword():
    """check_wakeword_config returns ERROR when keyword is empty."""
    from rex.doctor import Status, check_wakeword_config

    with _patch_load_config(_mock_config(wakeword="")):
        result = check_wakeword_config()

    assert result.status == Status.ERROR
    assert "empty" in result.message.lower()


def test_check_wakeword_config_error_missing_model_file(tmp_path):
    """check_wakeword_config returns ERROR when model file doesn't exist."""
    from rex.doctor import Status, check_wakeword_config

    missing = str(tmp_path / "missing.onnx")
    with _patch_load_config(_mock_config(wakeword_model_path=missing)):
        result = check_wakeword_config()

    assert result.status == Status.ERROR
    assert "not found" in result.message


def test_check_wakeword_config_ok_existing_model_file(tmp_path):
    """check_wakeword_config returns OK when model file exists."""
    from rex.doctor import Status, check_wakeword_config

    model_file = tmp_path / "rex.onnx"
    model_file.write_bytes(b"\x00" * 8)
    with _patch_load_config(_mock_config(wakeword_model_path=str(model_file))):
        result = check_wakeword_config()

    assert result.status == Status.OK
    assert "rex.onnx" in result.message
