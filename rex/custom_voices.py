"""Custom voice management for XTTS speaker cloning (US-VC-002).

Accepts an audio file (WAV or MP3), validates it meets the minimum duration
requirement, and saves it to the ``voices/`` directory so it appears in the
XTTS voice dropdown.  XTTS v2 uses WAV files directly as speaker references —
no model training is required.
"""

from __future__ import annotations

import logging
import re
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum audio duration required for a usable XTTS speaker reference.
MIN_DURATION_SECONDS: float = 10.0

# Directory where XTTS speaker WAV files are stored (project root / voices).
_VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"


def get_audio_duration(file_path: str | Path) -> float:
    """Return the duration of an audio file in seconds.

    Supports WAV via the stdlib ``wave`` module.  For other formats (MP3,
    FLAC, OGG) falls back to ``soundfile`` when installed.

    Raises ``ValueError`` if the file cannot be read or the format is
    unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except wave.Error as exc:
            raise ValueError(f"Cannot read WAV file: {exc}") from exc

    # Non-WAV formats: try soundfile.
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.duration
    except ImportError:
        raise ValueError(
            f"soundfile is required to read {suffix.upper()} files. "
            "Install it with: pip install soundfile"
        )
    except Exception as exc:
        raise ValueError(f"Cannot read audio file: {exc}") from exc


def _sanitize_name(name: str) -> str:
    """Return a filesystem-safe stem derived from *name*."""
    sanitized = re.sub(r"[^\w\s-]", "", name).strip()
    sanitized = re.sub(r"[\s-]+", "_", sanitized).lower()
    return sanitized or "custom_voice"


def save_custom_voice(
    file_path: str | Path,
    voice_name: str,
    *,
    voices_dir: Path | None = None,
) -> dict:
    """Validate and save an audio file as a custom XTTS speaker voice.

    Args:
        file_path: Path to the source audio file (WAV or MP3).
        voice_name: Human-readable name for the voice (used as filename stem).
        voices_dir: Override the default ``voices/`` directory (for testing).

    Returns:
        A dict with keys:
        - ``ok`` (bool): True if the voice was saved successfully.
        - ``voice_id`` (str): Absolute path of the saved WAV file.
        - ``voice_name`` (str): Sanitized display name.
        - ``duration`` (float): Audio duration in seconds.
        - ``error`` (str, only on failure): Human-readable error message.
    """
    source = Path(file_path)
    directory = voices_dir or _VOICES_DIR

    # Validate duration first (cheap — avoids unnecessary file I/O).
    try:
        duration = get_audio_duration(source)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "duration": 0.0}

    if duration < MIN_DURATION_SECONDS:
        remaining = MIN_DURATION_SECONDS - duration
        return {
            "ok": False,
            "error": (
                f"Audio is too short ({duration:.1f}s). "
                f"Need {remaining:.1f}s more to reach the 10-second minimum."
            ),
            "duration": duration,
        }

    stem = _sanitize_name(voice_name)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{stem}.wav"

    # Convert to WAV if needed, or copy directly.
    suffix = source.suffix.lower()
    try:
        if suffix == ".wav":
            import shutil

            shutil.copy2(str(source), str(dest))
        else:
            # MP3 / other: re-encode to WAV via soundfile.
            try:
                import soundfile as sf

                data, sample_rate = sf.read(str(source))
                sf.write(str(dest), data, sample_rate)
            except ImportError:
                return {
                    "ok": False,
                    "error": "soundfile is required to convert MP3 to WAV.",
                    "duration": duration,
                }
    except Exception as exc:
        return {"ok": False, "error": f"Failed to save voice file: {exc}", "duration": duration}

    display_name = stem.replace("_", " ").title()
    logger.info("[custom_voices] Saved %r → %s (%.1fs)", voice_name, dest, duration)

    return {
        "ok": True,
        "voice_id": str(dest),
        "voice_name": display_name,
        "duration": duration,
    }


__all__ = [
    "MIN_DURATION_SECONDS",
    "get_audio_duration",
    "save_custom_voice",
]
