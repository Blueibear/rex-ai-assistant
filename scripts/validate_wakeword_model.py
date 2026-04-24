"""Validate a wake word model or embedding file."""

from __future__ import annotations

import argparse
from pathlib import Path

from rex.wakeword.assets import validate_custom_wakeword_asset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a wake word model file.")
    parser.add_argument(
        "--backend",
        default="custom_onnx",
        choices=["custom_onnx", "custom_embedding"],
        help="Wake word backend to validate.",
    )
    parser.add_argument(
        "--phrase",
        default="hey rex",
        help="Wake phrase used to resolve default asset paths (default: 'hey rex').",
    )
    parser.add_argument("--model-path", help="Path to custom ONNX model.")
    parser.add_argument("--embedding-path", help="Path to custom embedding .pt file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path
    embedding_path = args.embedding_path

    if args.backend == "custom_onnx" and not model_path:
        raise SystemExit("custom_onnx requires --model-path")
    if args.backend == "custom_embedding" and not embedding_path:
        raise SystemExit("custom_embedding requires --embedding-path")

    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise SystemExit(f"Model file not found: {path}")

    if embedding_path:
        path = Path(embedding_path)
        if not path.exists():
            raise SystemExit(f"Embedding file not found: {path}")

    selection = validate_custom_wakeword_asset(
        phrase=args.phrase,
        backend=args.backend,
        model_path=model_path,
        embedding_path=embedding_path,
    )

    path = selection.resolved_model_path or selection.resolved_embedding_path
    print(
        "Wake word asset valid. "
        f"backend={selection.active_backend} label={selection.active_label} path={path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
