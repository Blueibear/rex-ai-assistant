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
    LOGGER.info("✅ Python version: %s", platform.python_version())


def check_ffmpeg(install_if_missing: bool = False) -> None:
    if shutil.which("ffmpeg"):
        LOGGER.info("✅ ffmpeg is available")
        return

    LOGGER.warning("⚠️  ffmpeg not found on PATH.")

    if install_if_missing:
        install_cmd = []

        if sys.platform.startswith("linux"):
            install_cmd = ["sudo", "apt", "install", "-y", "ffmpeg"]
        elif sys.platform == "darwin":
            install_cmd = ["brew", "install", "ffmpeg"]
        elif sys.platform == "win32":
            raise AssistantError("Please install ffmpeg manually on Windows and ensure it's on your PATH.")
        else:
            raise AssistantError(f"Unsupported platform for auto-install: {sys.platform}")

        try:
            LOGGER.info("Installing ffmpeg using: %s", " ".join(install_cmd))
            subprocess.check_call(install_cmd)
            LOGGER.info("✅ ffmpeg installed.")
        except Exception as exc:
            raise AssistantError(f"Failed to install ffmpeg: {exc}") from exc
    else:
        raise AssistantError("ffmpeg is not available on PATH. Please install it and retry.")


def install_requirements() -> None:
    LOGGER.info("📦 Installing Python dependencies…")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as exc:
        raise AssistantError(f"pip install failed: {exc}") from exc
    LOGGER.info("✅ Dependencies installed successfully.")


def run_mic_test() -> None:
    try:
        devices = list_devices()
    except AudioDeviceError as exc:
        LOGGER.warning("⚠️  Unable to query audio devices: %s", exc)
        return

    if not devices:
        LOGGER.warning("⚠️  No audio devices detected.")
        return

    LOGGER.info("🎙️  Available audio devices:")
    for idx, device in enumerate(devices):
        LOGGER.info("  [%2d] %s (in: %d, out: %d)",
                    idx,
                    device["name"],
                    device["max_input_channels"],
                    device["max_output_channels"]
        )


def show_system_info() -> None:
    LOGGER.info("🧠 System info:")
    LOGGER.info("  OS        : %s", platform.system())
    LOGGER.info("  Version   : %s", platform.version())
    LOGGER.info("  Arch      : %s", platform.machine())
    LOGGER.info("  Python    : %s", platform.python_version())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rex installation helper")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install step")
    parser.add_argument("--mic-test", action="store_true", help="List available audio devices")
    parser.add_argument("--auto-install-ffmpeg", action="store_true", help="Attempt to install ffmpeg automatically")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        show_system_info()
        check_python_version()
        check_ffmpeg(install_if_missing=args.auto_install_ffmpeg)

        if not args.no_install:
            install_requirements()

        if args.mic_test:
            run_mic_test()

        LOGGER.info("✅ Rex installation completed successfully.")
        return 0

    except AssistantError as exc:
        LOGGER.error("❌ Installation failed: %s", exc)
        return 1
    except subprocess.CalledProcessError as exc:
        LOGGER.error("❌ Command failed: %s", exc)
        return exc.returncode


if __name__ == "__main__":  # pragma: no cover - CLI utility
    raise SystemExit(main())

