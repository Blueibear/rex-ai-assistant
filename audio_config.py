"""CLI utilities for inspecting and selecting audio devices."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

try:
    import sounddevice as sd
except ImportError:
    sd = None  # optional dependency

from rex.assistant_errors import AudioDeviceError
from rex.config import reload_settings, update_env_value

import logging

logger = logging.getLogger(__name__)


def _require_sounddevice() -> None:
    if sd is None:
        raise AudioDeviceError("The 'sounddevice' package is required for audio device inspection.")


def list_devices() -> List[dict]:
    """Return all available audio devices."""
    _require_sounddevice()
    try:
        devices = sd.query_devices()
        formatted = []
        for index, device in enumerate(devices):
            formatted.append(
                {
                    "index": index,
                    "name": device.get("name"),
                    "max_input_channels": device.get("max_input_channels"),
                    "max_output_channels": device.get("max_output_channels"),
                }
            )
        return formatted
    except Exception as exc:
        raise AudioDeviceError(f"Failed to query audio devices: {exc}") from exc


def _validate_device(device_id: int) -> dict:
    devices = list_devices()
    if device_id < 0 or device_id >= len(devices):
        raise AudioDeviceError(f"Invalid device index: {device_id}")
    return devices[device_id]


def _persist_device(env_key: str, device_id: int) -> None:
    update_env_value(env_key, str(device_id))
    logger.info("Saved %s = %s", env_key, device_id)


def select_input(device_id: int) -> None:
    device = _validate_device(device_id)
    if device.get("max_input_channels", 0) < 1:
        raise AudioDeviceError(f"Device {device_id} has no input channels")

    try:
        with sd.InputStream(device=device_id, blocksize=0):
            pass
    except Exception as exc:
        raise AudioDeviceError(f"Unable to open input device {device_id}: {exc}") from exc

    _persist_device("REX_INPUT_DEVICE", device_id)


def select_output(device_id: int) -> None:
    device = _validate_device(device_id)
    if device.get("max_output_channels", 0) < 1:
        raise AudioDeviceError(f"Device {device_id} has no output channels")

    try:
        with sd.OutputStream(device=device_id, blocksize=0):
            pass
    except Exception as exc:
        raise AudioDeviceError(f"Unable to open output device {device_id}: {exc}") from exc

    _persist_device("REX_OUTPUT_DEVICE", device_id)


def _print_devices() -> None:
    try:
        devices = list_devices()
        print(json.dumps(devices, indent=2))
    except AudioDeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)


def _print_active_config() -> None:
    current = reload_settings()
    print(
        json.dumps(
            {
                "input_device": getattr(current, "input_device", None),
                "output_device": getattr(current, "output_device", None),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure Rex audio devices.")
    parser.add_argument("--list", action="store_true", help="List available audio devices")
    parser.add_argument("--set-input", type=int, metavar="INDEX", help="Persist the default input device index")
    parser.add_argument("--set-output", type=int, metavar="INDEX", help="Persist the default output device index")
    parser.add_argument("--show", action="store_true", help="Display the currently configured device indices")

    args = parser.parse_args(argv)

    try:
        if args.list:
            _print_devices()
            return 0

        if args.set_input is not None:
            select_input(args.set_input)
            print(f"Set input device to index {args.set_input}")

        if args.set_output is not None:
            select_output(args.set_output)
            print(f"Set output device to index {args.set_output}")

        if args.show:
            _print_active_config()

        if not any([args.list, args.set_input is not None, args.set_output is not None, args.show]):
            parser.print_help()
            return 1

        return 0

    except AudioDeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

