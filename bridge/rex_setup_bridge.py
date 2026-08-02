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

logger = logging.getLogger(__name__)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        sys.stdout.write(json.dumps({"ok": False, "error": "Invalid setup request"}))
        sys.exit(1)

    command = str(payload.get("command", ""))

    try:
        if command == "status":
            _handle_status()

        elif command == "complete":
            _handle_complete(payload)

        else:
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown command: {command}"}))

    except Exception:
        sys.stdout.write(json.dumps({"ok": False, "error": "Secure setup persistence failed"}))


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

    from rex.auth import _open_db, create_user  # noqa: PLC2701
    from rex.credential_persistence import persist_household_secrets
    from rex.permissions import bootstrap_admin_if_first_user

    user_id: str | None = None

    try:
        secrets_to_store: dict[str, str] = {"HA_TOKEN": ha_token}
        if llm_api_key:
            logical_name = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "ollama": "OLLAMA_API_KEY",
            }.get(llm_provider)
            if logical_name is None:
                raise ValueError("Selected LLM provider does not accept an API key")
            secrets_to_store[logical_name] = llm_api_key

        user = create_user(username, password)
        user_id = str(user["id"])
        bootstrap_admin_if_first_user(user_id)

        def update_config(config: dict[str, Any]) -> None:
            config.setdefault("llm", {})["provider"] = llm_provider
            if llm_provider == "ollama" and payload.get("ollama_base_url"):
                config.setdefault("llm", {})["ollama_base_url"] = payload["ollama_base_url"]
            config["tts_provider"] = tts_provider
            if ha_base_url:
                config.setdefault("home_assistant", {})["base_url"] = ha_base_url

        persist_household_secrets(secrets_to_store, update_config=update_config)
    except Exception:
        if user_id:
            try:
                with _open_db() as conn:
                    conn.execute("DELETE FROM user_permissions WHERE user_id = ?", (user_id,))
                    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
                    conn.commit()
            except Exception:
                logger.error("Setup rollback could not remove the incomplete user")
        raise
    sys.stdout.write(json.dumps({"ok": True, "user_id": user_id}))


if __name__ == "__main__":
    main()
