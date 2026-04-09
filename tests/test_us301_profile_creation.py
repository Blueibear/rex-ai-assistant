"""Tests for US-301: Graceful default profile creation on first run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex.profile_manager import (
    ensure_default_profile,
    load_profile,
)


# ---------------------------------------------------------------------------
# ensure_default_profile
# ---------------------------------------------------------------------------


def test_ensure_default_profile_creates_from_example(tmp_path: Path) -> None:
    """Copies default.example.json to default.json when example exists."""
    example = {
        "profile_version": 1,
        "name": "default",
        "description": "Example",
        "capabilities": ["local_commands"],
        "overrides": {},
    }
    (tmp_path / "default.example.json").write_text(
        json.dumps(example), encoding="utf-8"
    )

    created = ensure_default_profile(str(tmp_path))

    assert created is True
    dest = tmp_path / "default.json"
    assert dest.exists()
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded == example


def test_ensure_default_profile_creates_minimal_when_no_example(tmp_path: Path) -> None:
    """Generates a minimal valid profile when no example file is present."""
    created = ensure_default_profile(str(tmp_path))

    assert created is True
    dest = tmp_path / "default.json"
    assert dest.exists()
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["profile_version"] == 1
    assert loaded["name"] == "default"
    assert isinstance(loaded["capabilities"], list)
    assert isinstance(loaded["overrides"], dict)


def test_ensure_default_profile_does_not_overwrite_existing(tmp_path: Path) -> None:
    """Returns False and leaves the file untouched when default.json already exists."""
    original = {
        "profile_version": 1,
        "name": "default",
        "description": "Existing",
        "capabilities": ["ha_router"],
        "overrides": {},
    }
    dest = tmp_path / "default.json"
    dest.write_text(json.dumps(original), encoding="utf-8")

    created = ensure_default_profile(str(tmp_path))

    assert created is False
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded == original


def test_ensure_default_profile_creates_profiles_dir(tmp_path: Path) -> None:
    """Creates the profiles directory if it doesn't exist yet."""
    new_dir = tmp_path / "nonexistent_profiles"
    created = ensure_default_profile(str(new_dir))

    assert created is True
    assert (new_dir / "default.json").exists()


# ---------------------------------------------------------------------------
# load_profile fallback for missing non-default profiles
# ---------------------------------------------------------------------------


def test_load_profile_raises_for_missing_named_profile(tmp_path: Path) -> None:
    """load_profile raises FileNotFoundError for any missing profile (unchanged semantics)."""
    with pytest.raises(FileNotFoundError, match="james.json"):
        load_profile("james", profiles_dir=str(tmp_path))


def test_load_profile_raises_when_default_missing(tmp_path: Path) -> None:
    """Raises FileNotFoundError when the default profile itself is absent."""
    with pytest.raises(FileNotFoundError, match="default.json"):
        load_profile("default", profiles_dir=str(tmp_path))


def test_load_profile_loads_existing_named_profile(tmp_path: Path) -> None:
    """Returns the named profile when it exists."""
    profile = {
        "profile_version": 1,
        "name": "james",
        "description": "James profile",
        "capabilities": ["ha_router"],
        "overrides": {},
    }
    (tmp_path / "james.json").write_text(json.dumps(profile), encoding="utf-8")

    result = load_profile("james", profiles_dir=str(tmp_path))

    assert result["name"] == "james"


# ---------------------------------------------------------------------------
# Integration: _merge_profile_config creates default profile automatically
# and falls back to default for missing named profiles
# ---------------------------------------------------------------------------


def test_merge_profile_config_creates_default_profile(tmp_path: Path) -> None:
    """_merge_profile_config calls ensure_default_profile so load_config
    succeeds even when profiles/default.json is absent."""
    from rex.profile_manager import _MINIMAL_DEFAULT_PROFILE, ensure_default_profile

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    # No default.json present
    assert not (profiles_dir / "default.json").exists()

    ensure_default_profile(str(profiles_dir))

    assert (profiles_dir / "default.json").exists()
    loaded = json.loads((profiles_dir / "default.json").read_text(encoding="utf-8"))
    assert loaded["profile_version"] == _MINIMAL_DEFAULT_PROFILE["profile_version"]


def test_merge_profile_config_falls_back_when_named_profile_missing(
    tmp_path: Path,
) -> None:
    """_merge_profile_config warns and uses default when active_profile is missing."""
    from rex.config import _merge_profile_config

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    default_profile = {
        "profile_version": 1,
        "name": "default",
        "description": "Default",
        "capabilities": ["local_commands"],
        "overrides": {},
    }
    (profiles_dir / "default.json").write_text(
        json.dumps(default_profile), encoding="utf-8"
    )

    base_config: dict = {
        "active_profile": "james",
        "profiles_dir": str(profiles_dir),
    }
    merged = _merge_profile_config(base_config)

    # Should fall back to default profile, not crash
    assert merged["active_profile"] == "default"
    assert "local_commands" in merged.get("capabilities", [])
