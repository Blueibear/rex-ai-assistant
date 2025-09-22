"""Tests for the placeholder voice generator."""

from __future__ import annotations

from pathlib import Path

from placeholder_voice import ensure_placeholder_voice


def test_ensure_placeholder_voice_creates_file(tmp_path):
    target = tmp_path / "voice.wav"

    result = ensure_placeholder_voice(path=str(target))

    path = Path(result)
    assert path == target
    assert path.exists()
    assert path.stat().st_size > 0


def test_ensure_placeholder_voice_is_idempotent(tmp_path):
    target = tmp_path / "voice.wav"

    first = ensure_placeholder_voice(path=str(target))
    before = Path(first).read_bytes()

    second = ensure_placeholder_voice(path=str(target))
    after = Path(second).read_bytes()

    assert before == after
