"""TTS voice sample synthesis bridge.

Reads a JSON payload from stdin:
    {"provider": "<tts_provider>", "voice_id": "<voice_id>"}
    Optional: {"text": "<phrase>"}  (defaults to "Hello, I'm Rex.")

Writes a JSON response to stdout:
    {"ok": true, "audio_base64": "<base64-encoded audio bytes>"}
 or {"ok": false, "error": "<text>"}

Used by the Electron GUI main process to play a voice preview.
The ``provider`` field should match the active TTS backend configured in
rex_config.json (e.g. "xtts", "edge-tts", or "pyttsx3").
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        provider = str(payload.get("provider", "xtts"))
        voice_id = str(payload.get("voice_id", ""))
        text = str(payload.get("text", "Hello, I'm Rex."))[:50]
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    if not voice_id:
        print(json.dumps({"ok": False, "error": "voice_id is required"}), flush=True)
        sys.exit(1)

    try:
        from rex.tts_voices import synthesize_sample

        audio_bytes = asyncio.run(synthesize_sample(provider, voice_id, text))
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        print(json.dumps({"ok": True, "audio_base64": audio_base64}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
