"""Install a supplied wake-word asset into the repo convention path."""

from __future__ import annotations

import argparse
import json

from rex.wakeword.assets import install_custom_wakeword_asset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a custom wake-word asset into config/wake_words/<slug>/ and validate it."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["custom_onnx", "custom_embedding"],
        help="Asset backend type.",
    )
    parser.add_argument(
        "--phrase",
        default="hey rex",
        help="Wake phrase for metadata/path slugging (default: 'hey rex').",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source asset file to import.",
    )
    parser.add_argument(
        "--target",
        help="Optional explicit destination path. Defaults to config/wake_words/<slug>/model.onnx or embedding.pt.",
    )
    parser.add_argument(
        "--sample",
        help="Optional WAV sample to copy alongside the asset as sample.wav.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = install_custom_wakeword_asset(
        backend=args.backend,
        phrase=args.phrase,
        source_path=args.source,
        target_path=args.target,
        sample_path=args.sample,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
