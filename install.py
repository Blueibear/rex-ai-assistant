"""Interactive installer for the Rex assistant."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys

from assistant_errors import AssistantError, AudioDeviceError
from audio_config import list_devices
from logging_utils import get_logger

LOGGER = get_logger(__name__)


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise AssistantError("Python 3.10 or newer is required.")
    LOGGER.info("Detected Python %s", platform.python_version())


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise AssistantError("ffmpeg is not available on PATH. Please install ffmpeg and retry.")
    LOGGER.info("ffmpeg available")


def install_requirements() -> None:
    LOGGER.info("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def run_mic_test() -> None:
    try:
        devices = list_devices()
    except AudioDeviceError as exc:
        LOGGER.warning("Unable to query audio devices: %s", exc)
        return

    for idx, device in enumerate(devices):
        LOGGER.info("Device %d: %s", idx, device["name"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rex installation helper")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install step")
    parser.add_argument("--mic-test", action="store_true", help="List available audio devices")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        check_python_version()
        check_ffmpeg()
        if not args.no_install:
            install_requirements()
        if args.mic_test:
            run_mic_test()
    except AssistantError as exc:
        LOGGER.error("Installation failed: %s", exc)
        return 1
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Command failed: %s", exc)
        return exc.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI utility
    raise SystemExit(main())
