"""Generate the bundled placeholder voice sample on demand."""

from __future__ import annotations

import math
import os
import struct
import wave

DEFAULT_PLACEHOLDER_RELATIVE_PATH = os.path.join("assets", "voices", "placeholder.wav")
_SAMPLE_RATE = 22_050
_DURATION_SECONDS = 1.2
_FREQUENCY_HZ = 220.0
_AMPLITUDE = 16_000


def ensure_placeholder_voice(
    path: str | None = None,
    *,
    repo_root: str | None = None,
) -> str:
    """Return the absolute path to the placeholder voice sample.

    The sample is generated as a short sine wave if it does not already exist.
    The target location defaults to ``assets/voices/placeholder.wav`` within the
    repository, but callers can override ``path`` to create it elsewhere (useful
    for tests).
    """

    if repo_root is None:
        repo_root = os.path.dirname(os.path.abspath(__file__))

    if path is None:
        target_path = os.path.join(repo_root, DEFAULT_PLACEHOLDER_RELATIVE_PATH)
    elif os.path.isabs(path):
        target_path = path
    else:
        target_path = os.path.join(repo_root, path)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    total_frames = int(_DURATION_SECONDS * _SAMPLE_RATE)

    with wave.open(target_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_SAMPLE_RATE)

        for index in range(total_frames):
            sample = math.sin(2.0 * math.pi * _FREQUENCY_HZ * (index / _SAMPLE_RATE))
            value = int(_AMPLITUDE * sample)
            wav_file.writeframes(struct.pack("<h", value))

    return target_path


__all__ = ["DEFAULT_PLACEHOLDER_RELATIVE_PATH", "ensure_placeholder_voice"]
