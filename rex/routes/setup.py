"""Setup wizard routes — /api/setup/*."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint


def create_blueprint() -> Blueprint:
    """Return the setup wizard Blueprint."""
    bp = Blueprint("setup", __name__)

    @bp.route("/api/setup/status", methods=["GET"])
    def _setup_status() -> Any:
        """Return whether the initial setup wizard needs to run."""
        from flask import jsonify

        from rex.auth import _open_db  # noqa: PLC2701

        try:
            with _open_db() as conn:
                row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
                count = row[0] if row else 0
        except Exception:
            count = 0
        return jsonify({"needs_setup": count == 0}), 200

    @bp.route("/api/setup/complete", methods=["POST"])
    def _setup_complete() -> Any:
        """Complete the first-run wizard.

        Requires the ``X-Setup-Token`` header containing the single-use token
        generated at app start.  The token is consumed on the first successful
        call; subsequent calls return 403.
        """
        from flask import current_app, jsonify, request

        from rex.auth import create_user
        from rex.routes._helpers import _log_nonfatal_exception, _require_setup_token

        _, token_err = _require_setup_token()
        if token_err is not None:
            return token_err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        llm_provider = (data.get("llm_provider") or "local").strip()
        llm_api_key = (data.get("llm_api_key") or "").strip()
        tts_provider = (data.get("tts_provider") or "none").strip()
        ha_base_url = (data.get("ha_base_url") or "").strip()
        ha_token = (data.get("ha_token") or "").strip()

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        try:
            user = create_user(username, password)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

        try:
            from rex.permissions import bootstrap_admin_if_first_user

            bootstrap_admin_if_first_user(user["id"])
        except Exception:
            _log_nonfatal_exception("Failed to bootstrap first-user admin permissions")

        try:
            from rex.config_manager import load_config as _load_json_cfg
            from rex.config_manager import save_config as _save_json_cfg

            json_cfg: dict[str, Any] = _load_json_cfg() or {}
            json_cfg.setdefault("llm", {})["provider"] = llm_provider
            if llm_provider == "ollama" and data.get("ollama_base_url"):
                json_cfg.setdefault("llm", {})["ollama_base_url"] = data["ollama_base_url"]
            json_cfg["tts_provider"] = tts_provider
            if ha_base_url:
                json_cfg.setdefault("home_assistant", {})["base_url"] = ha_base_url
            _save_json_cfg(json_cfg)
        except Exception:
            _log_nonfatal_exception("Failed to persist setup wizard configuration")

        try:
            from rex.bridge_utils import repo_root

            env_path = repo_root() / ".env"
        except Exception:
            env_path = Path(".env")

        # Import lazily so monkeypatching rex.gui_app._write_env_secrets in tests works.
        from rex.gui_app import _write_env_secrets

        _write_env_secrets(
            env_path,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            ha_token=ha_token,
        )

        # Consume the setup token so the wizard cannot be re-run.
        current_app.config["SETUP_TOKEN"] = None

        return jsonify({"ok": True, "user_id": user["id"]}), 201

    return bp
