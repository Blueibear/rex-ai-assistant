"""Manual transcription demo using Whisper."""

from __future__ import annotations

import argparse
import os
import sys

import whisper


def transcribe_audio(file_path: str, model_name: str = "base", language: str | None = None) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, language=language)
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper Transcription Demo")
    parser.add_argument("audio", help="Path to the audio file (e.g., test_audio.mp3)")
    parser.add_argument("--model", default="base", help="Whisper model to use (default: base)")
    parser.add_argument("--lang", help="Language hint (e.g., 'en', 'es', etc.)")
    parser.add_argument("--output", help="Optional path to save transcription as .txt")

    args = parser.parse_args()

    try:
        print(f"🧠 Loading Whisper model: {args.model}")
        transcript = transcribe_audio(args.audio, model_name=args.model, language=args.lang)
        print("\n--- Transcript ---\n")
        print(transcript)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(transcript + "\n")
            print(f"\n📄 Transcript saved to: {args.output}")

    except Exception as exc:
        print(f"❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

