"""Wake word training bridge.

Reads a JSON payload from stdin:
    {
        "phrase": "hey rex",
        "positive_samples": [[...float32 PCM frames...], ...],
        "negative_samples": [[...float32 PCM frames...], ...]
    }

Writes a JSON response to stdout:
    {"ok": true, "model_path": "...", "phrase": "hey rex"}
 or {"ok": false, "error": "<text>"}

Used by the Electron GUI main process to train a custom wake word embedding.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Failed to read input: {exc}"}), flush=True)
        return

    phrase = payload.get("phrase", "")
    positive_samples = payload.get("positive_samples", [])
    negative_samples = payload.get("negative_samples", [])

    try:
        from rex.wakeword.trainer import train_from_samples

        result = train_from_samples(phrase, positive_samples, negative_samples)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
