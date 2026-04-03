"""Wake word listing bridge.

Reads an empty JSON payload from stdin (or {}).
Writes a JSON response to stdout:
    {"ok": true, "wake_words": [{"id": "<keyword>", "name": "<display>", "engine": "openwakeword"}, ...]}
 or {"ok": false, "error": "<text>", "wake_words": []}

Used by the Electron GUI main process to populate the wake word dropdown.
Falls back to a bundled default list if openWakeWord is not installed.
"""

from __future__ import annotations

import json
import sys

# Default set of known openWakeWord keywords returned when the library is not
# installed, ensuring the dropdown is never empty.
_DEFAULT_KEYWORDS = [
    "hey jarvis",
    "hey mycroft",
    "hey rhasspy",
    "ok nabu",
    "alexa",
]


def main() -> None:
    try:
        sys.stdin.read()  # consume stdin (payload ignored)
    except Exception:
        pass

    try:
        from rex.wakeword.selection import list_openwakeword_keywords

        try:
            import openwakeword as _oww

            raw = list_openwakeword_keywords(_oww)
        except ImportError:
            raw = []

        # Supplement with defaults if the live list is empty.
        keywords = raw if raw else _DEFAULT_KEYWORDS

        wake_words = [
            {
                "id": kw.replace(" ", "_"),
                "name": kw.replace("_", " ").title(),
                "engine": "openwakeword",
            }
            for kw in keywords
        ]
        print(json.dumps({"ok": True, "wake_words": wake_words}), flush=True)
    except Exception as exc:
        # Fall back to the default list so the UI is never empty.
        wake_words = [
            {
                "id": kw.replace(" ", "_"),
                "name": kw.replace("_", " ").title(),
                "engine": "openwakeword",
            }
            for kw in _DEFAULT_KEYWORDS
        ]
        print(
            json.dumps({"ok": True, "wake_words": wake_words, "warning": str(exc)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
