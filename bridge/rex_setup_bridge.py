"""Setup IPC bridge for the Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Supported commands:
  {"command": "status"}
    -> {"needs_setup": true|false}

  {"command": "complete", "username": "...", "password": "...",
   "llm_provider": "...", "llm_api_key": "...", "tts_provider": "...",
   "ha_base_url": "...", "ha_token": "..."}
    -> {"ok": true, "user_id": "..."} | {"ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from rex.bridge_utils import bridge_error_response

logger = logging.getLogger(__name__)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"Bad input: {exc}"}))
        sys.exit(1)

    command = str(payload.get("command", ""))

    try:
        if command == "status":
            _handle_status()

        elif command == "complete":
            _handle_complete(payload)

        else:
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown command: {command}"}))

    except Exception as exc:
        sys.stdout.write(json.dumps(bridge_error_response(exc)))


def _handle_status() -> None:
    from rex.auth import _open_db  # noqa: PLC2701

    try:
        with _open_db() as conn:
            row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
            count = row[0] if row else 0
    except Exception:
        count = 0
    sys.stdout.write(json.dumps({"needs_setup": count == 0}))


def _handle_complete(payload: dict[str, Any]) -> None:
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    llm_provider = str(payload.get("llm_provider") or "local").strip()
    llm_api_key = str(payload.get("llm_api_key") or "").strip()
    tts_provider = str(payload.get("tts_provider") or "none").strip()
    ha_base_url = str(payload.get("ha_base_url") or "").strip()
    ha_token = str(payload.get("ha_token") or "").strip()

    if not username or not password:
        sys.stdout.write(json.dumps({"ok": False, "error": "username and password are required"}))
        return

    from rex.auth import create_user

    try:
        user = create_user(username, password)
    except ValueError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}))
        return

    try:
        from rex.permissions import bootstrap_admin_if_first_user

        bootstrap_admin_if_first_user(user["id"])
    except Exception:
        logger.debug("Failed to bootstrap first-user admin permissions", exc_info=True)

    try:
        from rex.config_manager import load_config as _load_json_cfg
        from rex.config_manager import save_config as _save_json_cfg

        json_cfg: dict[str, Any] = _load_json_cfg() or {}
        json_cfg.setdefault("llm", {})["provider"] = llm_provider
        if llm_provider == "ollama" and payload.get("ollama_base_url"):
            json_cfg.setdefault("llm", {})["ollama_base_url"] = payload["ollama_base_url"]
        json_cfg["tts_provider"] = tts_provider
        if ha_base_url:
            json_cfg.setdefault("home_assistant", {})["base_url"] = ha_base_url
        _save_json_cfg(json_cfg)
    except Exception:
        logger.debug("Failed to persist setup wizard configuration", exc_info=True)

    from rex.runtime_paths import env_path as resolve_env_path

    env_path = resolve_env_path()

    from rex.gui_app import _write_env_secrets  # noqa: PLC2701

    _write_env_secrets(
        env_path,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        ha_token=ha_token,
    )

    sys.stdout.write(json.dumps({"ok": True, "user_id": user["id"]}))


if __name__ == "__main__":
    main()
