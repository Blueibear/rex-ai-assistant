"""Rex command history bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list", "limit": 50}
    -> {"ok": true, "history": [{id, timestamp, command, result, success}, ...]}
"""

from __future__ import annotations

import json
import sys

from rex.command_history import CommandHistoryStore


def main() -> None:
    try:
        raw = sys.stdin.read()
        req: dict = json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad request: {exc}"}))
        return

    command = req.get("command", "list")

    if command == "list":
        limit = int(req.get("limit", 50))
        try:
            from rex.identity import validate_user_id

            user_id = validate_user_id(str(req.get("user") or ""))
            if req.get("data_scope") != "private":
                raise PermissionError("Command history requires private Electron data scope")
            store = CommandHistoryStore()
            history = store.get_recent(limit=limit, user_id=user_id)
            print(json.dumps({"ok": True, "history": history}))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc), "history": []}))
    else:
        print(json.dumps({"ok": False, "error": f"Unknown command: {command}"}))


if __name__ == "__main__":
    main()
