"""Shared Python compatibility policy for Rex installs and runtime checks."""

from __future__ import annotations

from typing import Iterable

SUPPORTED_MAJOR = 3
SUPPORTED_MINOR = 11
SUPPORTED_MICRO = 0

MIN_VERSION = (SUPPORTED_MAJOR, SUPPORTED_MINOR, SUPPORTED_MICRO)
MAX_VERSION_EXCLUSIVE = (SUPPORTED_MAJOR, SUPPORTED_MINOR + 1, 0)

SUPPORTED_VERSION_LABEL = "Python 3.11"
SUPPORTED_RANGE_LABEL = ">=3.11,<3.12"

DEFAULT_INSTALL_LABEL = "default install"
WINDOWS_GPU_INSTALL_LABEL = "Windows GPU + TTS install"
ML_AUDIO_EXTRAS_LABEL = "ML/audio/TTS extras"


def normalize_version(version_info: object) -> tuple[int, int, int]:
    """Normalize sys.version_info-like values into a simple version tuple."""
    if hasattr(version_info, "major") and hasattr(version_info, "minor"):
        major = int(getattr(version_info, "major"))
        minor = int(getattr(version_info, "minor"))
        micro = int(getattr(version_info, "micro", 0))
        return (major, minor, micro)

    if isinstance(version_info, Iterable):
        parts = [int(part) for part in version_info]
        if len(parts) >= 3:
            return (parts[0], parts[1], parts[2])
        if len(parts) == 2:
            return (parts[0], parts[1], 0)

    raise TypeError(f"Unsupported version info: {version_info!r}")


def format_version(version_info: object) -> str:
    """Return a human-readable version string."""
    major, minor, micro = normalize_version(version_info)
    return f"{major}.{minor}.{micro}"


def is_supported_python(version_info: object) -> bool:
    """Return True when the version is within the supported runtime window."""
    version = normalize_version(version_info)
    return MIN_VERSION <= version < MAX_VERSION_EXCLUSIVE


def unsupported_python_message(
    version_info: object,
    *,
    install_target: str = DEFAULT_INSTALL_LABEL,
) -> str:
    """Return a plain-English fail-fast message for unsupported interpreters."""
    current = format_version(version_info)
    return (
        f"Unsupported Python {current} for Rex {install_target}. "
        f"Use {SUPPORTED_VERSION_LABEL}. "
        "The current dependency stack is only validated on Python 3.11, and "
        "fresh installs on Python 3.13/3.14 are known to fail in the ML/TTS path."
    )
