"""Manual transcription demo using Whisper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import whisper


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to the audio clip to transcribe (WAV/MP3/etc).",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model variant to load (default: base).",
    )
    args = parser.parse_args()

    if not args.audio:
        print(
            "Please provide a path to an audio file, e.g.:\n"
            "  python manual_whisper_demo.py path/to/clip.wav",
            file=sys.stderr,
        )
        return 1

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    model = whisper.load_model(args.model)
    result = model.transcribe(str(audio_path))
    print(result.get("text", "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
