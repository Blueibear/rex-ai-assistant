"""Manual transcription demo using Whisper."""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env
_load_env()

import argparse
import os
import sys
from pathlib import Path

import whisper


def transcribe_audio(file_path: str, model_name: str = "base", language: str | None = None) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, language=language)
    return result.get("text", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("audio", help="Path to the audio file (WAV, MP3, etc.)")
    parser.add_argument("--model", default="base", help="Whisper model to use (default: base)")
    parser.add_argument("--lang", help="Optional language hint (e.g., 'en', 'es')")
    parser.add_argument("--output", help="Optional path to save the transcript as a .txt file")

    args = parser.parse_args()

    try:
        print(f"🧠 Loading Whisper model: {args.model}")
        transcript = transcribe_audio(args.audio, model_name=args.model, language=args.lang)

        print("\n--- Transcript ---\n")
        print(transcript)

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(transcript + "\n", encoding="utf-8")
            print(f"\n📄 Transcript saved to: {output_path}")

        return 0

    except Exception as exc:
        print(f"❌ Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
