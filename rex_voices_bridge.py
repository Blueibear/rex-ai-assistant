"""TTS voice listing bridge.

Reads a JSON payload from stdin:  {"provider": "<tts_provider>"}
Writes a JSON response to stdout: {"ok": true, "voices": [...]}
                               or {"ok": false, "error": "<text>"}

Used by the Electron GUI main process to populate the voice selector dropdown.
The ``provider`` field should match the active TTS backend configured in
rex_config.json (e.g. "xtts", "edge-tts", or "pyttsx3").
"""

from __future__ import annotations

import json
import sys

from rex.bridge_utils import repo_root, resolve_python

_PYTHON_EXE = resolve_python()  # venv-aware interpreter path for subprocess calls
_REPO_ROOT = repo_root()  # absolute repo root for resolving scripts and config


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        provider = str(payload.get("provider", "xtts"))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        from rex.tts_voices import list_voices

        voices = list_voices(provider)
        print(json.dumps({"ok": True, "voices": voices}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
