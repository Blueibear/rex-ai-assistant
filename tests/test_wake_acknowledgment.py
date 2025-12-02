import os
import wave
from pathlib import Path

from wake_acknowledgment import (
    DEFAULT_WAKE_ACK_RELATIVE_PATH,
    ensure_wake_acknowledgment_sound,
)


def test_generates_acknowledgment_sound(tmp_path):
    target = tmp_path / "ack.wav"
    path = ensure_wake_acknowledgment_sound(path=str(target))
    assert path == str(target)
    assert target.exists()
    assert target.stat().st_size > 0

    with wave.open(str(target), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24_000
        assert wav_file.getsampwidth() == 2


def test_reuses_existing_file(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "assets").mkdir()
    target = repo_root / DEFAULT_WAKE_ACK_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    generated_first = ensure_wake_acknowledgment_sound(repo_root=str(repo_root))
    size_first = os.path.getsize(generated_first)

    generated_second = ensure_wake_acknowledgment_sound(repo_root=str(repo_root))
    assert generated_second == generated_first
    assert os.path.getsize(generated_second) == size_first


def test_cleanup_removes_legacy_assets(tmp_path):
    repo_root = tmp_path / "repo"
    legacy_one = repo_root / "assets" / "rex_wake_acknowledgment (1).wav"
    legacy_two = repo_root / "assets" / "rex_wake_acknowledgment.wav"

    for legacy in (legacy_one, legacy_two):
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"legacy")

    result = ensure_wake_acknowledgment_sound(repo_root=str(repo_root))

    generated = Path(result)
    assert generated.name == Path(DEFAULT_WAKE_ACK_RELATIVE_PATH).name
    assert generated.exists()
    assert generated.stat().st_size > 0

    assert not legacy_one.exists()
    assert not legacy_two.exists()
