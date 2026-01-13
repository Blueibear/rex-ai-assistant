"""CLI utilities for inspecting and selecting audio devices.

This module now uses rex_config.json for persistence instead of .env.
"""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env

_load_env()

import argparse
import sys
from typing import Dict, Optional

try:
    import sounddevice as sd
except ImportError:
    sd = None  # sounddevice is optional

from assistant_errors import AudioDeviceError
from logging_utils import get_logger
from rex.config_manager import load_config, save_config

logger = get_logger(__name__)


def _require_sounddevice() -> None:
    if sd is None:
        raise AudioDeviceError("The 'sounddevice' package is required for audio device selection.")


def list_devices() -> list[dict]:
    _require_sounddevice()
    try:
        return sd.query_devices()
    except Exception as exc:
        raise AudioDeviceError(f"Failed to query audio devices: {exc}") from exc


def get_selected_input_device_index(config: Dict) -> Optional[int]:
    """Get selected input device index from config dict.

    Args:
        config: Configuration dict (from config_manager.load_config)

    Returns:
        Device index or None
    """
    return config.get("audio", {}).get("input_device_index")


def set_selected_input_device_index(config: Dict, index: Optional[int]) -> Dict:
    """Set selected input device index in config dict.

    Args:
        config: Configuration dict
        index: Device index or None

    Returns:
        Updated config dict
    """
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["input_device_index"] = index
    return config


def get_selected_output_device_index(config: Dict) -> Optional[int]:
    """Get selected output device index from config dict.

    Args:
        config: Configuration dict (from config_manager.load_config)

    Returns:
        Device index or None
    """
    return config.get("audio", {}).get("output_device_index")


def set_selected_output_device_index(config: Dict, index: Optional[int]) -> Dict:
    """Set selected output device index in config dict.

    Args:
        config: Configuration dict
        index: Device index or None

    Returns:
        Updated config dict
    """
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["output_device_index"] = index
    return config


def select_input(device_id: int, *, config: Optional[Dict] = None) -> None:
    """Select and persist input device to rex_config.json.

    Args:
        device_id: Device index to select

    Raises:
        AudioDeviceError: If device is invalid or cannot be opened
    """
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

    # Save to rex_config.json
    if config is None:
        config = load_config()
    config = set_selected_input_device_index(config, device_id)
    save_config(config)
    logger.info(f"Selected input device {device_id}, saved to config")


def select_output(device_id: int, *, config: Optional[Dict] = None) -> None:
    """Select and persist output device to rex_config.json.

    Args:
        device_id: Device index to select

    Raises:
        AudioDeviceError: If device is invalid or cannot be opened
    """
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

    # Save to rex_config.json
    if config is None:
        config = load_config()
    config = set_selected_output_device_index(config, device_id)
    save_config(config)
    logger.info(f"Selected output device {device_id}, saved to config")


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

        config = None
        if args.set_input is not None or args.set_output is not None:
            config = load_config()

        if args.set_input is not None:
            select_input(args.set_input, config=config)
            print(f"Input device set to index {args.set_input}")

        if args.set_output is not None:
            select_output(args.set_output, config=config)
            print(f"Output device set to index {args.set_output}")

        if args.show:
            config = load_config()
            input_idx = get_selected_input_device_index(config)
            output_idx = get_selected_output_device_index(config)
            print("Configured Audio Devices:")
            print(f"  Input Device Index : {input_idx}")
            print(f"  Output Device Index: {output_idx}")

        if not any([args.list, args.set_input is not None, args.set_output is not None, args.show]):
            parser.print_help()
            return 1

        return 0
    except AudioDeviceError as exc:
        logger.error("Audio error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point used by unit tests to invoke the CLI."""

    return cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
