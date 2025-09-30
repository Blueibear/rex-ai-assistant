"""CLI utilities for inspecting and selecting audio devices."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Iterable, Optional, Tuple

from assistant_errors import AudioDeviceError
from config import ENV_MAPPING, load_config, update_env_value as _config_update

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    sd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional auto-detection helper
    import pyaudio  # type: ignore
except ImportError:  # pragma: no cover
    pyaudio = None

LOGGER = logging.getLogger(__name__)


def update_env_value(key: str, value: str) -> None:
    """Persist a configuration override to the shared .env file."""

    _config_update(key, value)
    LOGGER.info("Persisted %s=%s", key, value)


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


def _persist_device(env_key: str, device_id: int) -> None:
    update_env_value(env_key, str(device_id))


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
    except Exception as exc:  # hardware dependent
        raise AudioDeviceError(f"Unable to open input device {device_id}: {exc}") from exc
    _persist_device(ENV_MAPPING["audio_input_device"], device_id)


def select_output(device_id: int) -> None:
    device = _validate_device(device_id)
    if device.get("max_output_channels", 0) < 1:
        raise AudioDeviceError(f"Device {device_id} has no output channels")
    try:
        with sd.OutputStream(device=device_id, blocksize=0):
            pass
    except Exception as exc:  # hardware dependent
        raise AudioDeviceError(f"Unable to open output device {device_id}: {exc}") from exc
    _persist_device(ENV_MAPPING["audio_output_device"], device_id)


def _auto_detect_device(kind: str) -> Optional[int]:
    if pyaudio is None:  # pragma: no cover - dependency optional
        return None
    pa = pyaudio.PyAudio()
    try:
        if kind == "input":
            info = pa.get_default_input_device_info()
        else:
            info = pa.get_default_output_device_info()
        if not info:
            return None
        index = info.get("index")
        return int(index) if isinstance(index, int) else None
    except OSError:
        return None
    finally:
        pa.terminate()


def auto_configure_defaults() -> Tuple[Optional[int], Optional[int]]:
    """Use PyAudio defaults to choose sensible devices."""

    input_index = _auto_detect_device("input")
    output_index = _auto_detect_device("output")

    if input_index is not None:
        update_env_value(ENV_MAPPING["audio_input_device"], str(input_index))
    if output_index is not None:
        update_env_value(ENV_MAPPING["audio_output_device"], str(output_index))

    return input_index, output_index


def _print_devices() -> None:
    devices = list_devices()
    print(json.dumps(devices, indent=2))


def _print_active_config() -> None:
    current = load_config(reload=True)
    print(
        json.dumps(
            {
                "input_device": current.audio_input_device,
                "output_device": current.audio_output_device,
            },
            indent=2,
        )
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Configure Rex audio devices.")
    parser.add_argument("--list", action="store_true", help="List available audio devices")
    parser.add_argument("--set-input", type=int, metavar="INDEX", help="Persist the default input device index")
    parser.add_argument("--set-output", type=int, metavar="INDEX", help="Persist the default output device index")
    parser.add_argument("--show", action="store_true", help="Display the currently configured device indices")
    parser.add_argument("--auto", action="store_true", help="Auto-detect defaults via PyAudio when available")

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.list:
            _print_devices()
            return 0

        updated = False
        if args.auto:
            inferred_in, inferred_out = auto_configure_defaults()
            print(
                json.dumps(
                    {
                        "auto_detected_input": inferred_in,
                        "auto_detected_output": inferred_out,
                    },
                    indent=2,
                )
            )
            updated = updated or (inferred_in is not None or inferred_out is not None)

        if args.set_input is not None:
            select_input(args.set_input)
            updated = True
        if args.set_output is not None:
            select_output(args.set_output)
            updated = True

        if args.show:
            _print_active_config()

        if updated and not args.show:
            # After updates, surface the new config so callers get immediate feedback.
            _print_active_config()

        if not any([args.list, args.auto, args.set_input is not None, args.set_output is not None, args.show]):
            parser.print_help()
            return 1

        return 0
    except AudioDeviceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
