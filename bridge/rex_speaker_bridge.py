"""Rex smart speaker bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list"}
    -> {"ok": true, "speakers": [{"provider": "sonos", "name": "...", "ip": "...", "model": "..."}]}
"""

from __future__ import annotations

import json
import sys

from rex.bridge_utils import bridge_error_response, repo_root, resolve_python

_PYTHON_EXE = resolve_python()  # venv-aware interpreter path for subprocess calls
_REPO_ROOT = repo_root()  # absolute repo root for resolving scripts and config


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}))
        return

    command = payload.get("action") or payload.get("command", "list")

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
            print(json.dumps({**bridge_error_response(exc), "speakers": []}))
    else:
        print(json.dumps({"ok": False, "error": f"unknown command: {command}"}))


if __name__ == "__main__":
    main()
