"""Validate a wake word model or embedding file."""

from __future__ import annotations

import argparse
from pathlib import Path

from rex.wakeword.utils import load_wakeword_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a wake word model file.")
    parser.add_argument(
        "--backend",
        default="custom_onnx",
        choices=["custom_onnx", "custom_embedding"],
        help="Wake word backend to validate.",
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

    model, label = load_wakeword_model(
        backend=args.backend,
        model_path=model_path,
        embedding_path=embedding_path,
        fallback_to_builtin=False,
    )

    _ = model  # Model is validated by loading it
    print(f"Wake word model valid. Label: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

