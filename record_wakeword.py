"""Utility helpers for managing Picovoice Porcupine keywords."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pvporcupine


def list_keywords() -> None:
    keywords = pvporcupine.KEYWORDS
    print("Available Porcupine keywords:")
    for keyword in sorted(keywords):
        print(f" - {keyword}")


def export_custom(keyword_path: Path) -> None:
    if not keyword_path.is_file():
        raise SystemExit(f"Keyword file not found: {keyword_path}")
    info = {
        "keyword_path": str(keyword_path.resolve()),
        "description": "Custom Porcupine keyword",
    }
    output = keyword_path.with_suffix(".json")
    output.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Metadata exported to {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Porcupine keyword helper")
    parser.add_argument("--list", action="store_true", help="List built-in keywords")
    parser.add_argument(
        "--export",
        type=Path,
        metavar="KEYWORD.PPN",
        help="Record metadata for a custom Porcupine keyword",
    )
    args = parser.parse_args(argv)

    if args.list:
        list_keywords()
        return 0

    if args.export:
        export_custom(args.export)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
