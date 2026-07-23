"""Quick actions IPC bridge for the Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Supported commands:
  {"command": "list"}
    -> {"ok": true, "quick_actions": [{id, label, command}, ...]}

  {"command": "add", "label": "...", "command_text": "..."}
    -> {"ok": true, "action": {id, label, command}}

  {"command": "delete", "id": "..."}
    -> {"ok": true, "deleted": true|false}

  {"command": "run", "id": "..."}
    -> {"status": "attempted", "detail": "<reply>"} | {"status": "failed", "detail": "<error>"}
"""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any

from rex.bridge_utils import bridge_error_response


def _resolve_user_id(payload: dict[str, Any]) -> str | None:
    """Return a deliberately selected user ID, or ``None`` when missing."""
    try:
        from rex.identity import validate_user_id

        if payload.get("data_scope") != "private":
            return None
        return validate_user_id(str(payload.get("user") or ""))
    except Exception:
        return None


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
    user_id = _resolve_user_id(payload)

    try:
        if user_id is None:
            raise PermissionError("A valid active user is required for quick actions.")
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

        elif command == "delete":
            action_id = str(payload.get("id", "")).strip()
            if not action_id:
                sys.stdout.write(json.dumps({"ok": False, "error": "id is required"}))
                return
            actions = _get_quick_actions(user_id)
            filtered = [a for a in actions if a.get("id") != action_id]
            _save_quick_actions(user_id, filtered)
            sys.stdout.write(json.dumps({"ok": True, "deleted": len(filtered) < len(actions)}))

        elif command == "run":
            action_id = str(payload.get("id", "")).strip()
            if not action_id:
                sys.stdout.write(json.dumps({"status": "failed", "detail": "id is required"}))
                return
            actions = _get_quick_actions(user_id)
            action = next((a for a in actions if a.get("id") == action_id), None)
            if action is None:
                sys.stdout.write(
                    json.dumps({"status": "failed", "detail": f"action {action_id!r} not found"})
                )
                return
            command_text = str(action.get("command", "")).strip()
            if not command_text:
                sys.stdout.write(
                    json.dumps({"status": "failed", "detail": "action has no command text"})
                )
                return
            try:
                import asyncio

                from rex.assistant import Assistant
                from rex.logging_utils import configure_logging
                from rex.plugins import load_plugins, shutdown_plugins
                from rex.services import initialize_services

                configure_logging()
                initialize_services()
                plugin_specs = load_plugins()
                # Run the action as the already-resolved bridge user (issue
                # #303): Assistant no longer invents an identity on its own.
                assistant = Assistant(user_id=user_id)
                try:
                    reply = asyncio.run(assistant.generate_reply(command_text))
                    sys.stdout.write(json.dumps({"status": "attempted", "detail": str(reply)}))
                finally:
                    shutdown_plugins(plugin_specs)
            except Exception as exc:
                sys.stdout.write(json.dumps({"status": "failed", "detail": str(exc)}))

        else:
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown command: {command}"}))

    except Exception as exc:
        sys.stdout.write(json.dumps(bridge_error_response(exc)))


if __name__ == "__main__":
    main()
