"""Rex smart speaker bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list"}
    -> {"ok": true, "speakers": [{"provider": "sonos", "name": "...", "ip": "...", "model": "..."}]}
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read().strip() or "{}")
    except json.JSONDecodeError:
        payload = {}

    command = payload.get("command", "list")

    if command == "list":
        try:
            from rex.audio.speaker_discovery import SpeakerDiscoveryService

            svc = SpeakerDiscoveryService(
                refresh_interval_seconds=60.0,
                discovery_timeout_seconds=1.0,
            )
            speakers = svc.discover_now()
            result = [
                {"provider": s.provider, "name": s.name, "ip": s.ip, "model": s.model}
                for s in speakers
            ]
            print(json.dumps({"ok": True, "speakers": result}))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "speakers": [], "error": str(exc)}))
    else:
        print(json.dumps({"ok": False, "error": f"unknown command: {command}"}))


if __name__ == "__main__":
    main()
