"""Tests for the memory utility helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)


def test_resolve_user_key_from_email():
    users_map = load_users_map()
    profiles = load_all_profiles()

    result = resolve_user_key("james@example.com", users_map, profiles=profiles)
    assert result == "james"


def test_extract_voice_reference_handles_missing(monkeypatch, tmp_path):
    absolute_voice = tmp_path / "voice.wav"
    absolute_voice.write_bytes(b"fake")
    profile = {"voice": {"sample_path": str(absolute_voice)}}

    assert extract_voice_reference(profile) == str(absolute_voice.resolve())

    user_dir = tmp_path / "james"
    user_dir.mkdir()
    relative_voice = user_dir / "voice.wav"
    relative_voice.write_bytes(b"demo")
    profile_relative = {"voice_sample": "voice.wav"}

    assert extract_voice_reference(
        profile_relative,
        user_key="james",
        memory_root=str(tmp_path),
    ) == str(relative_voice.resolve())

    profile_missing = {"voice": {}}
    assert (
        extract_voice_reference(
            profile_missing,
            user_key="james",
            memory_root=str(tmp_path),
        )
        is None
    )


def test_extract_voice_reference_generates_placeholder(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    profile = {"voice_sample": "assets/voices/placeholder.wav"}

    result = extract_voice_reference(
        profile,
        memory_root=str(repo_root / "Memory"),
        repo_root=str(repo_root),
    )

    path = Path(result)
    assert path.exists()
    assert path.stat().st_size > 0


def test_load_memory_profile_enforces_size_limit(monkeypatch, tmp_path):
    directory = tmp_path / "user"
    directory.mkdir()
    core = directory / "core.json"
    core.write_text("{}")

    monkeypatch.setenv("REX_MEMORY_MAX_BYTES", "1")

    from importlib import reload
    import memory_utils as memory_utils_module

    reload(memory_utils_module)

    with pytest.raises(ValueError):
        memory_utils_module.load_memory_profile("user", memory_root=str(tmp_path))

    monkeypatch.delenv("REX_MEMORY_MAX_BYTES", raising=False)
    reload(memory_utils_module)
