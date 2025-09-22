"""CLI helpers for selecting audio devices used by the assistant."""

from __future__ import annotations

import argparse
from typing import List

import sounddevice as sd

from assistant_errors import AudioDeviceError
from config import ENV_MAPPING, ENV_PATH, load_config
from logging_utils import get_logger
from dotenv import set_key

LOGGER = get_logger(__name__)


def list_devices() -> List[dict]:
    try:
        return sd.query_devices()
    except Exception as exc:
        raise AudioDeviceError(f"Failed to query audio devices: {exc}") from exc


def _set_device(env_key: str, device_id: int) -> None:
    set_key(str(ENV_PATH), env_key, str(device_id))
    LOGGER.info("Saved %s as %s", device_id, env_key)


def select_input(device_id: int) -> None:
    devices = list_devices()
    if device_id < 0 or device_id >= len(devices):
        raise AudioDeviceError(f"Invalid input device id: {device_id}")
    with sd.InputStream(device=device_id):
        pass
    _set_device(ENV_MAPPING["audio_input_device"], device_id)


def select_output(device_id: int) -> None:
    devices = list_devices()
    if device_id < 0 or device_id >= len(devices):
        raise AudioDeviceError(f"Invalid output device id: {device_id}")
    with sd.OutputStream(device=device_id):
        pass
    _set_device(ENV_MAPPING["audio_output_device"], device_id)


def _format_devices() -> str:
    devices = list_devices()
    rows = ["ID | Name | Max In | Max Out"]
    for idx, device in enumerate(devices):
        rows.append(
            f"{idx:2d} | {device['name']} | {device['max_input_channels']} | {device['max_output_channels']}"
        )
    return "\n".join(rows)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audio configuration helpers")
    parser.add_argument("--list", action="store_true", help="List available devices")
    parser.add_argument("--set-input", type=int, help="Persist the selected input device")
    parser.add_argument("--set-output", type=int, help="Persist the selected output device")
    parser.add_argument("--show", action="store_true", help="Show the active configuration values")
    args = parser.parse_args(argv)

    if args.list:
        print(_format_devices())

    if args.set_input is not None:
        select_input(args.set_input)

    if args.set_output is not None:
        select_output(args.set_output)

    if args.show:
        cfg = load_config(reload=True)
        LOGGER.info(
            "Input device: %s | Output device: %s",
            cfg.audio_input_device,
            cfg.audio_output_device,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI utility
    try:
        raise SystemExit(cli())
    except AudioDeviceError as exc:
        LOGGER.error("Audio configuration error: %s", exc)
        raise SystemExit(1) from exc
