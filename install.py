"""Simple installation wizard for the Rex assistant."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_COMMANDS = ["ffmpeg"]


def _print_status(message: str) -> None:
    print(f"[rex-install] {message}")


def _check_command(name: str) -> bool:
    return shutil.which(name) is not None


def _install_requirements() -> None:
    requirements = Path("requirements.txt")
    if not requirements.exists():
        _print_status("requirements.txt not found; skipping dependency installation")
        return
    _print_status("Installing Python dependencies (this may take a while)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def main() -> int:
    missing = [cmd for cmd in REQUIRED_COMMANDS if not _check_command(cmd)]
    if missing:
        _print_status(
            "Missing required system commands: " + ", ".join(missing) + ". Install them before continuing."
        )
    else:
        _print_status("All required system commands are available.")

    try:
        _install_requirements()
    except subprocess.CalledProcessError as exc:
        _print_status(f"Dependency installation failed: {exc}")
        return exc.returncode or 1

    _print_status("Setup complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
