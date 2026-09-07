"""Path normalization helpers for Windows service launch and registration."""

from __future__ import annotations

from os import PathLike
from pathlib import Path


def normalize_existing_executable(executable: str | PathLike[str]) -> Path:
    """Return an absolute executable path, failing closed when it is not a file."""

    resolved = Path(executable).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected executable file was not found: {resolved}")
    return resolved
