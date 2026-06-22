"""Quick actions IPC bridge for the Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Supported commands:
  {"command": "list"}
    -> {"ok": true, "quick_actions": [{id, label, command}, ...]}

  {"command": "add", "label": "...", "command_text": "..."}
    -> {"ok": true, "action": {id, label, command}}
"""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any

from rex.bridge_utils import bridge_error_response, repo_root, resolve_python

_PYTHON_EXE = resolve_python()
_REPO_ROOT = repo_root()


def _resolve_user_id() -> str:
    """Return the active user ID, falling back to config then 'default'."""
    try:
        from rex.config import load_config

        config = load_config()
        runtime = getattr(config, "_raw", None)
        if runtime is None:
            # Try to get the user from config attributes directly
            uid = getattr(config, "user_id", None) or getattr(config, "default_user", None)
            if uid and str(uid) != "default":
                return str(uid)
        # Use rex.identity resolution chain
        from rex.identity import resolve_active_user

        user = resolve_active_user()
        return user if user else "default"
    except Exception:
        return "default"


def _get_quick_actions(user_id: str) -> list[dict[str, Any]]:
    from rex.identity import get_user_profile

    profile = get_user_profile(user_id) or {}
    prefs = profile.get("preferences", {})
    actions = prefs.get("quick_actions", [])
    return actions if isinstance(actions, list) else []


def _save_quick_actions(user_id: str, actions: list[dict[str, Any]]) -> None:
    from rex.identity import update_user_preferences

    update_user_preferences(user_id, {"quick_actions": actions})


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"Bad input: {exc}"}))
        sys.exit(1)

    command = str(payload.get("command", ""))
    user_id = _resolve_user_id()

    try:
        if command == "list":
            actions = _get_quick_actions(user_id)
            sys.stdout.write(json.dumps({"ok": True, "quick_actions": actions}))

        elif command == "add":
            label = str(payload.get("label", "")).strip()
            command_text = str(payload.get("command_text", "")).strip()
            if not label or not command_text:
                sys.stdout.write(
                    json.dumps({"ok": False, "error": "label and command_text are required"})
                )
                return
            actions = _get_quick_actions(user_id)
            new_action: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "label": label,
                "command": command_text,
            }
            actions.append(new_action)
            _save_quick_actions(user_id, actions)
            sys.stdout.write(json.dumps({"ok": True, "action": new_action}))

        else:
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown command: {command}"}))

    except Exception as exc:
        sys.stdout.write(json.dumps(bridge_error_response(exc)))


if __name__ == "__main__":
    main()
