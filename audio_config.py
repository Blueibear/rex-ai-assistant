"""CLI utilities for inspecting and selecting audio devices."""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env
_load_env()

import argparse
import sys

try:
    import sounddevice as sd
except ImportError:
    sd = None  # sounddevice is optional

from dotenv import set_key

from assistant_errors import AudioDeviceError
from config import ENV_MAPPING, ENV_PATH, load_config
from logging_utils import get_logger

logger = get_logger(__name__)


def update_env_value(key: str, value: str) -> None:
    """Persist an environment override to the shared .env file."""

    set_key(str(ENV_PATH), key, value)
    logger.info("Persisted %s = %s", key, value)


def _require_sounddevice() -> None:
    if sd is None:
        raise AudioDeviceError("The 'sounddevice' package is required for audio device selection.")


def list_devices() -> list[dict]:
    _require_sounddevice()
    try:
        return sd.query_devices()
    except Exception as exc:
        raise AudioDeviceError(f"Failed to query audio devices: {exc}") from exc


def _set_device(env_key: str, device_id: int) -> None:
    update_env_value(env_key, str(device_id))


def select_input(device_id: int) -> None:
    devices = list_devices()
    if device_id < 0 or device_id >= len(devices):
        raise AudioDeviceError(f"Invalid input device ID: {device_id}")

    device = devices[device_id]
    if device["max_input_channels"] < 1:
        raise AudioDeviceError(f"Device {device_id} has no input channels.")

    try:
        with sd.InputStream(device=device_id, blocksize=0):
            pass
    except Exception as exc:
        raise AudioDeviceError(f"Failed to open input device {device_id}: {exc}") from exc

    _set_device(ENV_MAPPING["audio_input_device"], device_id)


def select_output(device_id: int) -> None:
    devices = list_devices()
    if device_id < 0 or device_id >= len(devices):
        raise AudioDeviceError(f"Invalid output device ID: {device_id}")

    device = devices[device_id]
    if device["max_output_channels"] < 1:
        raise AudioDeviceError(f"Device {device_id} has no output channels.")

    try:
        with sd.OutputStream(device=device_id, blocksize=0):
            pass
    except Exception as exc:
        raise AudioDeviceError(f"Failed to open output device {device_id}: {exc}") from exc

    _set_device(ENV_MAPPING["audio_output_device"], device_id)


def _format_devices() -> str:
    devices = list_devices()
    rows = [" ID | Name                           | In | Out"]
    rows.append("-" * 50)
    for idx, device in enumerate(devices):
        rows.append(
            f"{idx:2d} | {device['name'][:30]:<30} | {device['max_input_channels']:2d} | {device['max_output_channels']:2d}"
        )
    return "\n".join(rows)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure audio devices for Rex.")
    parser.add_argument("--list", action="store_true", help="List available audio devices")
    parser.add_argument(
        "--set-input", type=int, metavar="INDEX", help="Persist default input device"
    )
    parser.add_argument(
        "--set-output", type=int, metavar="INDEX", help="Persist default output device"
    )
    parser.add_argument("--show", action="store_true", help="Show current configured devices")

    args = parser.parse_args(argv)

    try:
        if args.list:
            print(_format_devices())
            return 0

        if args.set_input is not None:
            select_input(args.set_input)
            print(f"✅ Input device set to index {args.set_input}")

        if args.set_output is not None:
            select_output(args.set_output)
            print(f"✅ Output device set to index {args.set_output}")

        if args.show:
            cfg = load_config(reload=True)
            print("Configured Audio Devices:")
            print(f"  🎤 Input Device Index : {cfg.audio_input_device}")
            print(f"  🔈 Output Device Index: {cfg.audio_output_device}")

        if not any([args.list, args.set_input is not None, args.set_output is not None, args.show]):
            parser.print_help()
            return 1

        return 0
    except AudioDeviceError as exc:
        logger.error("Audio error: %s", exc)
        print(f"❌ Error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point used by unit tests to invoke the CLI."""

    return cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
