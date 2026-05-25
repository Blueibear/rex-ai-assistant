"""Wake word sample playback bridge.

Reads a JSON payload from stdin:
    {"wake_word_id": "<slug>"}

Writes a JSON response to stdout:
    {"ok": true, "audio_base64": "<base64-encoded WAV>"}
 or {"ok": false, "has_sample": false, "error": "<text>"}

Used by the Electron GUI to play back a recorded sample of a custom wake word.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

_CONFIG_DIR_DEFAULT = Path(__file__).resolve().parent / "config" / "wake_words"


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Failed to read input: {exc}"}), flush=True)
        return

    wake_word_id = payload.get("wake_word_id", "").strip()
    if not wake_word_id:
        print(json.dumps({"ok": False, "error": "wake_word_id is required"}), flush=True)
        return

    sample_path = _CONFIG_DIR_DEFAULT / wake_word_id / "sample.wav"
    if not sample_path.is_file():
        print(
            json.dumps(
                {
                    "ok": False,
                    "has_sample": False,
                    "error": f"No sample recording found for '{wake_word_id}'",
                }
            ),
            flush=True,
        )
        return

    try:
        audio_bytes = sample_path.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        print(json.dumps({"ok": True, "audio_base64": audio_b64}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Failed to read sample: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
