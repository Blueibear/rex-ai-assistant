"""Home Assistant and device routes — /api/ha/*, /api/devices/*."""
from __future__ import annotations

import json
from typing import Any

from flask import Blueprint


def create_blueprint() -> Blueprint:
    """Return the HA and devices Blueprint."""
    bp = Blueprint("ha", __name__)

    # ------------------------------------------------------------------
    # Device control API (US-060)
    # ------------------------------------------------------------------

    @bp.route("/api/devices", methods=["GET"])
    def _list_devices() -> Any:
        """Return approved devices from config/device_aliases.json."""
        from pathlib import Path

        from flask import jsonify

        try:
            from rex.bridge_utils import repo_root

            aliases_path = repo_root() / "config" / "device_aliases.json"
        except Exception:
            aliases_path = Path("config") / "device_aliases.json"

        try:
            raw: Any = json.loads(aliases_path.read_text(encoding="utf-8"))
            devices = raw.get("devices", []) if isinstance(raw, dict) else []
        except Exception:
            devices = []
        return jsonify({"devices": devices}), 200

    @bp.route("/api/devices/<path:entity_id>/command", methods=["POST"])
    def _device_command(entity_id: str) -> Any:
        """Send a command to a Home Assistant entity.

        Requires auth.  Body: ``{command: str, value?: any}``
        """
        import urllib.request

        from flask import jsonify, request

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        command = (data.get("command") or "").strip()
        value = data.get("value")

        if not command:
            return jsonify({"error": "command is required"}), 400

        _COMMAND_MAP: dict[str, tuple[str, str]] = {
            "turn_on": ("homeassistant", "turn_on"),
            "turn_off": ("homeassistant", "turn_off"),
            "set_brightness": ("light", "turn_on"),
            "media_play": ("media_player", "media_play"),
            "media_pause": ("media_player", "media_pause"),
            "media_next_track": ("media_player", "media_next_track"),
            "volume_set": ("media_player", "volume_set"),
        }

        if command not in _COMMAND_MAP:
            return jsonify({"error": f"unknown command: {command}"}), 400

        domain, service = _COMMAND_MAP[command]

        try:
            from rex.config import load_config

            cfg = load_config()
            ha_url = (cfg.ha_base_url or "").rstrip("/")
            ha_token = cfg.ha_token or ""
        except Exception:
            return jsonify({"error": "Home Assistant is not configured"}), 503

        if not ha_url:
            return jsonify({"error": "Home Assistant URL is not configured"}), 503

        service_url = f"{ha_url}/api/services/{domain}/{service}"
        payload: dict[str, Any] = {"entity_id": entity_id}
        if command == "set_brightness" and value is not None:
            payload["brightness"] = int(value)
        elif command == "volume_set" and value is not None:
            payload["volume_level"] = float(value)

        body = json.dumps(payload).encode()
        req = urllib.request.Request(service_url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if ha_token:
            req.add_header("Authorization", f"Bearer {ha_token}")

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                ok = resp.status in (200, 201)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 200

        return jsonify({"ok": ok}), 200

    # ------------------------------------------------------------------
    # Home Assistant setup API (US-059)
    # ------------------------------------------------------------------

    @bp.route("/api/ha/test", methods=["POST"])
    def _ha_test_connection() -> Any:
        """Test a Home Assistant connection using the supplied URL and token.

        Requires a valid authenticated session.  Body: ``{ha_base_url, ha_token}``
        """
        import urllib.parse
        import urllib.request

        from flask import jsonify, request

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        base_url = (data.get("ha_base_url") or "").rstrip("/")
        token = (data.get("ha_token") or "").strip()

        if not base_url:
            return jsonify({"ok": False, "error": "ha_base_url is required"}), 400

        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            return (
                jsonify({"ok": False, "error": "ha_base_url must use http or https scheme"}),
                400,
            )

        api_url = f"{base_url}/api/"
        req = urllib.request.Request(api_url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                ok = resp.status == 200
        except Exception:
            from flask import current_app

            current_app.logger.debug("HA connection test failed", exc_info=True)
            return jsonify({"ok": False, "error": "connection failed"}), 200

        return jsonify({"ok": ok}), 200

    @bp.route("/api/ha/save", methods=["POST"])
    def _ha_save_config() -> Any:
        """Save Home Assistant URL and token.  Requires auth.

        Body: ``{ha_base_url: str, ha_token?: str}``
        """
        import os
        from pathlib import Path

        from flask import jsonify, request

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        base_url = (data.get("ha_base_url") or "").strip()
        token = (data.get("ha_token") or "").strip()

        if not base_url:
            return jsonify({"error": "ha_base_url is required"}), 400

        try:
            from rex.config_manager import load_config as _load_json_cfg
            from rex.config_manager import save_config as _save_json_cfg

            json_cfg: dict[str, Any] = _load_json_cfg() or {}
            json_cfg.setdefault("home_assistant", {})["base_url"] = base_url
            _save_json_cfg(json_cfg)
        except Exception as exc:
            return jsonify({"error": f"failed to save config: {exc}"}), 500

        if token:
            try:
                from rex.bridge_utils import repo_root

                env_path = repo_root() / ".env"
            except Exception:
                env_path = Path(".env")

            # Import lazily so monkeypatching rex.gui_app._write_env_secrets in tests works.
            from rex.gui_app import _write_env_secrets

            _write_env_secrets(env_path, llm_provider="", llm_api_key="", ha_token=token)
            os.environ["HA_TOKEN"] = token

        try:
            from rex.config import load_config as _reload_app_config

            _reload_app_config(reload=True)
        except Exception:
            pass

        return jsonify({"ok": True}), 200

    @bp.route("/api/ha/states", methods=["GET"])
    def _ha_get_states() -> Any:
        """Return all entity states from Home Assistant."""
        import json as _json
        import ssl
        import urllib.error
        import urllib.request

        from flask import jsonify

        from rex.config import load_config

        cfg = load_config(reload=True)
        base_url = (cfg.ha_base_url or "").rstrip("/")
        token = cfg.ha_token or ""

        if not base_url:
            return (
                jsonify(
                    {
                        "ok": False,
                        "not_configured": True,
                        "error": "Home Assistant is not configured",
                    }
                ),
                200,
            )

        url = f"{base_url}/api/states"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")

        ssl_ctx: ssl.SSLContext | None = None
        if not cfg.ha_verify_ssl:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(req, timeout=cfg.ha_timeout, context=ssl_ctx) as resp:
                raw_states: list[dict[str, Any]] = _json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return jsonify({"ok": False, "error": f"HA returned HTTP {exc.code}"}), 200
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 200

        states = [
            {
                "entity_id": s.get("entity_id", ""),
                "state": str(s.get("state", "unknown")),
                "friendly_name": s.get("attributes", {}).get(
                    "friendly_name", s.get("entity_id", "")
                ),
                "last_updated": s.get("last_updated", ""),
            }
            for s in raw_states
            if isinstance(s, dict)
        ]
        return jsonify({"ok": True, "states": states}), 200

    return bp
