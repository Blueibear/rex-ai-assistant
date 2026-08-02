"""Setup wizard routes — /api/setup/*."""

from __future__ import annotations

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

        from rex.auth import _open_db, create_user  # noqa: PLC2701
        from rex.routes._helpers import _require_setup_token

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
            from rex.credential_persistence import persist_household_secrets
            from rex.permissions import bootstrap_admin_if_first_user

            bootstrap_admin_if_first_user(user["id"])
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

            def update_config(config: dict[str, Any]) -> None:
                config.setdefault("llm", {})["provider"] = llm_provider
                if llm_provider == "ollama" and data.get("ollama_base_url"):
                    config.setdefault("llm", {})["ollama_base_url"] = data["ollama_base_url"]
                config["tts_provider"] = tts_provider
                if ha_base_url:
                    config.setdefault("home_assistant", {})["base_url"] = ha_base_url

            persist_household_secrets(secrets_to_store, update_config=update_config)
        except Exception:
            try:
                with _open_db() as conn:
                    conn.execute("DELETE FROM user_permissions WHERE user_id = ?", (user["id"],))
                    conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
                    conn.commit()
            except Exception:
                pass
            return jsonify({"error": "setup could not be persisted securely"}), 500

        # Consume the setup token so the wizard cannot be re-run.
        current_app.config["SETUP_TOKEN"] = None

        return jsonify({"ok": True, "user_id": user["id"]}), 201

    return bp
