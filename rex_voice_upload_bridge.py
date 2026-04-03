"""Custom voice upload bridge.

Reads a JSON payload from stdin:
    {"file_path": "<absolute path to audio file>", "voice_name": "<name>"}

Writes a JSON response to stdout:
    {"ok": true, "voice_id": "<path>", "voice_name": "<name>", "duration": <seconds>}
 or {"ok": false, "error": "<text>", "duration": <seconds>}

Used by the Electron GUI main process to save uploaded audio files as custom
XTTS speaker voices.  The ``file_path`` must be an absolute path to the
temporary file written by the renderer after the user selects a file.

Voice files are saved to the ``voices/`` directory at the project root, where
``rex.tts_voices.list_voices("xtts")`` will discover them automatically.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        file_path = str(payload.get("file_path", ""))
        voice_name = str(payload.get("voice_name", "")).strip() or "Custom Voice"
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}", "duration": 0.0}), flush=True)
        sys.exit(1)

    if not file_path:
        print(
            json.dumps({"ok": False, "error": "file_path is required", "duration": 0.0}),
            flush=True,
        )
        sys.exit(1)

    try:
        from rex.custom_voices import save_custom_voice

        result = save_custom_voice(file_path, voice_name)
        print(json.dumps(result), flush=True)
        if not result.get("ok"):
            sys.exit(1)
    except Exception as exc:
        print(
            json.dumps({"ok": False, "error": str(exc), "duration": 0.0}),
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
