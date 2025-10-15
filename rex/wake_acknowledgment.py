"""Utilities for generating the wake confirmation sound on demand."""

from __future__ import annotations

import math
import os
import struct
import wave
from typing import Optional

DEFAULT_WAKE_ACK_RELATIVE_PATH = os.path.join("assets", "wake_acknowledgment.wav")
_LEGACY_WAKE_ACK_RELATIVE_PATHS = (
    os.path.join("assets", "rex_wake_acknowledgment.wav"),
    os.path.join("assets", "rex_wake_acknowledgment (1).wav"),
)
_SAMPLE_RATE = 24_000
_DURATION_SECONDS = 0.35
_AMPLITUDE = 12_000
_FREQUENCIES = (880.0, 660.0)
_FADE_DURATION = 0.02


def _envelope(time_seconds: float) -> float:
    """Compute a simple fade-in/fade-out envelope to avoid clicks."""

    fade_in = min(time_seconds / _FADE_DURATION, 1.0)
    fade_out = min(max((_DURATION_SECONDS - time_seconds) / _FADE_DURATION, 0.0), 1.0)
    return min(fade_in, fade_out)


def _cleanup_legacy_acknowledgment_samples(base_directory: str) -> None:
    """Delete outdated wake-acknowledgment audio files if they linger."""

    for relative_path in _LEGACY_WAKE_ACK_RELATIVE_PATHS:
        candidate = os.path.join(base_directory, relative_path)
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            # If we cannot remove the file (e.g., permissions), fall back to the
            # safer option of leaving it in place. The repository integrity test
            # will still flag the tracked artifact.
            continue


def ensure_wake_acknowledgment_sound(
    path: Optional[str] = None,
    *,
    repo_root: Optional[str] = None,
) -> str:
    """Return the absolute path to the wake confirmation sound sample.

    The sample is generated as a short dual-tone sine wave if it does not
    already exist. The target location defaults to ``assets/wake_acknowledgment.wav``
    within the repository, but callers can override ``path`` to create it elsewhere
    (useful for tests).
    """

    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if path is None:
        target_path = os.path.join(repo_root, DEFAULT_WAKE_ACK_RELATIVE_PATH)
    elif os.path.isabs(path):
        target_path = path
    else:
        target_path = os.path.join(repo_root, path)

    _cleanup_legacy_acknowledgment_samples(repo_root)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    total_frames = int(_DURATION_SECONDS * _SAMPLE_RATE)

    with wave.open(target_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_SAMPLE_RATE)

        for index in range(total_frames):
            time_seconds = index / _SAMPLE_RATE
            envelope = _envelope(time_seconds)
            sample = 0.0
            for frequency in _FREQUENCIES:
                sample += math.sin(2.0 * math.pi * frequency * time_seconds)
            sample /= len(_FREQUENCIES)
            value = int(_AMPLITUDE * envelope * sample)
            wav_file.writeframes(struct.pack("<h", value))

    return target_path


__all__ = [
    "DEFAULT_WAKE_ACK_RELATIVE_PATH",
    "ensure_wake_acknowledgment_sound",
]
