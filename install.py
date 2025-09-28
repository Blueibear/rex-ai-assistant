"""Simple installation wizard for the Rex assistant."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_COMMANDS = ["ffmpeg"]


def _print_status(message: str) -> None:
    print(f"[rex-install] {message}")


def _check_command(name: str) -> bool:
    return shutil.which(name) is not None


def _install_file(path: Path) -> None:
    if not path.exists():
        _print_status(f"{path.name} not found; skipping")
        return
    _print_status(f"Installing dependencies from {path.name}…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(path)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Rex dependencies.")
    parser.add_argument(
        "--with-ml",
        action="store_true",
        help="Include optional machine-learning dependencies (Whisper, Torch, XTTS).",
    )
    parser.add_argument(
        "--with-dev",
        action="store_true",
        help="Include developer tooling such as pytest and coverage plugins.",
    )
    args = parser.parse_args(argv)

    missing = [cmd for cmd in REQUIRED_COMMANDS if not _check_command(cmd)]
    if missing:
        _print_status(
            "Missing required system commands: " + ", ".join(missing) + ". Install them before continuing."
        )
    else:
        _print_status("All required system commands are available.")

    try:
        _install_file(Path("requirements.txt"))
        if args.with_ml:
            _install_file(Path("requirements-ml.txt"))
        if args.with_dev:
            _install_file(Path("requirements-dev.txt"))
    except subprocess.CalledProcessError as exc:
        _print_status(f"Dependency installation failed: {exc}")
        return exc.returncode or 1

    _print_status("Setup complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
