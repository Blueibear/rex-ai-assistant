"""Record a custom ONNX wake-word asset with openWakeWord."""

from __future__ import annotations

import argparse
import json

from rex.wakeword.assets import create_openwakeword_model_asset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a custom ONNX wake-word asset into config/wake_words/<slug>/model.onnx."
    )
    parser.add_argument(
        "--phrase",
        default="hey rex",
        help="Wake phrase to record (default: 'hey rex').",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to config/wake_words/<slug>/model.onnx.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = create_openwakeword_model_asset(
        phrase=args.phrase,
        target_path=args.output,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
