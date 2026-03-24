"""Tests for profile loading and merging."""

from pathlib import Path

import pytest

from rex.profile_manager import apply_profile, get_active_profile_name, load_profile


def test_load_profile_default_and_james():
    profiles_dir = Path(__file__).resolve().parents[1] / "profiles"
    default_profile = load_profile("default", profiles_dir=str(profiles_dir))
    james_profile = load_profile("james", profiles_dir=str(profiles_dir))

    assert default_profile["name"] == "default"
    assert james_profile["name"] == "james"


def test_apply_profile_overrides_nested_keys():
    base_config = {
        "runtime": {"log_level": "INFO", "transcripts_enabled": True},
        "models": {"tts_provider": "xtts"},
    }
    profile = {
        "capabilities": ["local_commands"],
        "overrides": {
            "runtime": {"log_level": "DEBUG"},
            "models": {"tts_provider": "edge"},
        },
    }

    merged = apply_profile(base_config, profile)

    assert merged["runtime"]["log_level"] == "DEBUG"
    assert merged["runtime"]["transcripts_enabled"] is True
    assert merged["models"]["tts_provider"] == "edge"
    assert merged["capabilities"] == ["local_commands"]


def test_get_active_profile_default():
    assert get_active_profile_name({}) == "default"


def test_load_profile_missing_error_message(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        load_profile("missing", profiles_dir=str(tmp_path))

    message = str(exc_info.value)
    assert "active_profile" in message
    assert "missing.json" in message
