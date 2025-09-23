"""CLI utilities for inspecting and selecting audio devices."""

from __future__ import annotations

import argparse
import json
import sys

from rex.config import reload_settings, update_env_value

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    sd = None  # type: ignore[assignment]


def list_devices() -> list[dict]:
    if sd is None:
        raise RuntimeError("sounddevice is not installed")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure Rex audio devices.")
    parser.add_argument("--list", action="store_true", help="List available audio devices")
    parser.add_argument("--set-input", type=int, metavar="INDEX", help="Persist the default input device index")
    parser.add_argument("--set-output", type=int, metavar="INDEX", help="Persist the default output device index")

    args = parser.parse_args(argv)

    if args.list:
        try:
            devices = list_devices()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(devices, indent=2))
        return 0

    updated = False
    if args.set_input is not None:
        update_env_value("REX_INPUT_DEVICE", str(args.set_input))
        updated = True
    if args.set_output is not None:
        update_env_value("REX_OUTPUT_DEVICE", str(args.set_output))
        updated = True

    if updated:
        reload_settings()
        print("Audio device preferences updated.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
