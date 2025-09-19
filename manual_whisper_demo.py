"""Manual transcription demo using Whisper."""

from __future__ import annotations

import whisper


def main() -> None:
    model = whisper.load_model("base")
    result = model.transcribe("test_audio.mp3")
    print(result["text"])


if __name__ == "__main__":
    main()
