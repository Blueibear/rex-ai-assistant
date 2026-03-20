"""Tests for rex.wake_acknowledgment module.

This is distinct from tests/test_wake_acknowledgment.py which imports the
root-level wake_acknowledgment.py shim.  These tests exercise the canonical
rex package module at rex/wake_acknowledgment.py.
"""

from __future__ import annotations

import os
import wave


def test_import():
    """rex.wake_acknowledgment imports without error."""
    import rex.wake_acknowledgment  # noqa: F401


def test_default_path_constant():
    """DEFAULT_WAKE_ACK_RELATIVE_PATH points into the assets directory."""
    from rex.wake_acknowledgment import DEFAULT_WAKE_ACK_RELATIVE_PATH

    assert "wake_acknowledgment.wav" in DEFAULT_WAKE_ACK_RELATIVE_PATH
    assert "assets" in DEFAULT_WAKE_ACK_RELATIVE_PATH


def test_all_exports():
    """__all__ exposes the public API."""
    import rex.wake_acknowledgment as m

    assert "ensure_wake_acknowledgment_sound" in m.__all__
    assert "DEFAULT_WAKE_ACK_RELATIVE_PATH" in m.__all__


def test_ensure_generates_valid_wav(tmp_path):
    """ensure_wake_acknowledgment_sound creates a valid mono 24 kHz WAV file."""
    from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound

    target = str(tmp_path / "ack.wav")
    result = ensure_wake_acknowledgment_sound(path=target)

    assert result == target
    assert os.path.exists(target)
    assert os.path.getsize(target) > 0

    with wave.open(target, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 24_000
        assert wf.getsampwidth() == 2


def test_ensure_reuses_existing_file(tmp_path):
    """ensure_wake_acknowledgment_sound skips regeneration when file already exists."""
    from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound

    target = str(tmp_path / "ack.wav")
    ensure_wake_acknowledgment_sound(path=target)
    size_first = os.path.getsize(target)

    ensure_wake_acknowledgment_sound(path=target)
    assert os.path.getsize(target) == size_first


def test_ensure_default_path_within_repo_root(tmp_path):
    """ensure_wake_acknowledgment_sound creates the file at the default path under repo_root."""
    from rex.wake_acknowledgment import (
        DEFAULT_WAKE_ACK_RELATIVE_PATH,
        ensure_wake_acknowledgment_sound,
    )

    repo_root = str(tmp_path)
    result = ensure_wake_acknowledgment_sound(repo_root=repo_root)
    expected = os.path.join(repo_root, DEFAULT_WAKE_ACK_RELATIVE_PATH)

    assert result == expected
    assert os.path.exists(result)


def test_cleanup_removes_legacy_files(tmp_path):
    """ensure_wake_acknowledgment_sound deletes legacy asset files."""
    from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound

    legacy_name = "rex_wake_acknowledgment.wav"
    legacy_path = tmp_path / "assets" / legacy_name
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(b"old_data")

    assert legacy_path.exists()
    ensure_wake_acknowledgment_sound(repo_root=str(tmp_path))
    assert not legacy_path.exists()
