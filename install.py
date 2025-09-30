"""Interactive installer for the Rex assistant."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from assistant_errors import AssistantError, AudioDeviceError
from audio_config import list_devices
from logging_utils import get_logger

logger = get_logger(__name__)


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise AssistantError("Python 3.10 or newer is required.")
    logger.info("✅ Python version: %s", platform.python_version())


def check_ffmpeg(install_if_missing: bool = False) -> None:
    if shutil.which("ffmpeg"):
        logger.info("✅ ffmpeg is available")
        return

    logger.warning("⚠️  ffmpeg not found on PATH.")

    if install_if_missing:
        if sys.platform.startswith("linux"):
            cmd = ["sudo", "apt", "install", "-y", "ffmpeg"]
        elif sys.platform == "darwin":
            cmd = ["brew", "install", "ffmpeg"]
        elif sys.platform == "win32":
            raise AssistantError("Please install ffmpeg manually on Windows and ensure it's on your PATH.")
        else:
            raise AssistantError(f"Unsupported platform: {sys.platform}")

        try:
            logger.info("Installing ffmpeg: %s", " ".join(cmd))
            subprocess.check_call(cmd)
            logger.info("✅ ffmpeg installed.")
        except Exception as exc:
            raise AssistantError(f"Failed to install ffmpeg: {exc}") from exc
    else:
        raise AssistantError("ffmpeg is not installed. Please install it and retry.")


def install_requirements(include_ml: bool = False, include_dev: bool = False) -> None:
    def _install_file(path: Path) -> None:
        if not path.exists():
            logger.warning("⚠️  %s not found, skipping.", path.name)
            return
        logger.info("📦 Installing dependencies from %s…", path.name)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(path)])

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _install_file(Path("requirements.txt"))

    if include_ml:
        _install_file(Path("requirements-ml.txt"))
    if include_dev:
        _install_file(Path("requirements-dev.txt"))

    logger.info("✅ Python packages installed.")


def run_mic_test() -> None:
    try:
        devices = list_devices()
    except AudioDeviceError as exc:
        logger.warning("⚠️  Unable to query audio devices: %s", exc)
        return

    if not devices:
        logger.warning("⚠️  No audio devices detected.")
        return

    logger.info("🎙️  Available audio devices:")
    for idx, device in enumerate(devices):
        logger.info(
            "  [%2d] %s (in: %d, out: %d)",
            idx,
            device["name"],
            device["max_input_channels"],
            device["max_output_channels"]
        )


def show_system_info() -> None:
    logger.info("🧠 System Info:")
    logger.info("  OS        : %s", platform.system())
    logger.info("  Version   : %s", platform.version())
    logger.info("  Arch      : %s", platform.machine())
    logger.info("  Python    : %s", platform.python_version())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rex installation helper")
    parser.add_argument("--no-install", action="store_true", help="Skip Python package installation")
    parser.add_argument("--with-ml", action="store_true", help="Also install machine learning dependencies")
    parser.add_argument("--with-dev", action="store_true", help="Also install dev/testing dependencies")
    parser.add_argument("--mic-test", action="store_true", help="List and test audio devices")
    parser.add_argument("--auto-install-ffmpeg", action="store_true", help="Try to install ffmpeg if missing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        show_system_info()
        check_python_version()
        check_ffmpeg(install_if_missing=args.auto_install_ffmpeg)

        if not args.no_install:
            install_requirements(include_ml=args.with_ml, include_dev=args.with_dev)

        if args.mic_test:
            run_mic_test()

        logger.info("✅ Rex installation completed successfully.")
        return 0

    except AssistantError as exc:
        logger.error("❌ Installation failed: %s", exc)
        return 1
    except subprocess.CalledProcessError as exc:
        logger.error("❌ Command failed: %s", exc)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())

