"""CLI utilities for inspecting and selecting audio devices."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rex.assistant_errors import AudioDeviceError
from rex.config import reload_settings, update_env_value as _config_update_env_value

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    sd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _require_sounddevice() -> None:
    if sd is None:
        raise AudioDeviceError("sounddevice is not installed. Install it to inspect audio devices.")


def list_devices() -> list[dict]:
    """Return sounddevice's device listing as serialisable dictionaries."""

    _require_sounddevice()
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


def update_env_value(key: str, value: str) -> None:
    """Persist ``value`` for ``key`` via the central configuration layer."""

    _config_update_env_value(key, value)


def _persist_device(env_key: str, device_id: int) -> None:
    update_env_value(env_key, str(device_id))
    logger.info("Saved %s=%s", env_key, device_id)


def _validate_device(device_id: int) -> dict:
    _require_sounddevice()
    devices = sd.query_devices()
    if device_id < 0 or device_id >= len(devices):
        raise AudioDeviceError(f"Invalid device index: {device_id}")
    return devices[device_id]


def select_input(device_id: int) -> None:
    device = _validate_device(device_id)
    if device.get("max_input_channels", 0) < 1:
        raise AudioDeviceError(f"Device {device_id} has no input channels")

    try:
        with sd.InputStream(device=device_id, blocksize=0):
            pass
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise AudioDeviceError(f"Unable to open input device {device_id}: {exc}") from exc

    _persist_device("REX_INPUT_DEVICE", device_id)


def select_output(device_id: int) -> None:
    device = _validate_device(device_id)
    if device.get("max_output_channels", 0) < 1:
        raise AudioDeviceError(f"Device {device_id} has no output channels")

    try:
        with sd.OutputStream(device=device_id, blocksize=0):
            pass
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise AudioDeviceError(f"Unable to open output device {device_id}: {exc}") from exc

    _persist_device("REX_OUTPUT_DEVICE", device_id)


def _print_devices() -> None:
    try:
        devices = list_devices()
    except AudioDeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return

    print(json.dumps(devices, indent=2))


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

        if args.show:
            _print_active_config()

        updated = False
        if args.set_input is not None:
            select_input(args.set_input)
            updated = True
        if args.set_output is not None:
            select_output(args.set_output)
            updated = True

        if updated:
            print("Audio device preferences updated.")
            return 0

        if args.show:
            return 0
    except AudioDeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
