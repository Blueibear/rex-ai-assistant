"""Home Assistant and device routes — /api/ha/*, /api/devices/*."""

from __future__ import annotations

import http.client
import json
import ssl
import urllib.error
import urllib.parse
from typing import Any, NamedTuple

from flask import Blueprint

_ALLOWED_HTTP_SCHEMES = {"http", "https"}


class _HttpResponse(NamedTuple):
    status: int
    body: bytes


def _validate_http_url(url: str, *, field_name: str = "url") -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_HTTP_SCHEMES:
        raise ValueError(f"{field_name} must use http or https scheme")
    if not parsed.hostname:
        raise ValueError(f"{field_name} must include a host")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} must include a valid port") from exc
    return parsed


def _request_home_assistant(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 5,
    ssl_context: ssl.SSLContext | None = None,
) -> _HttpResponse:
    parsed = _validate_http_url(url, field_name="Home Assistant URL")
    target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    port = parsed.port
    request_headers = headers or {}

    if parsed.scheme.lower() == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname,
            port=port,
            timeout=timeout,
            context=ssl_context,
        )
    else:
        conn = http.client.HTTPConnection(parsed.hostname, port=port, timeout=timeout)

    try:
        conn.request(method, target, body=body, headers=request_headers)
        resp = conn.getresponse()
        response_body = resp.read()
        if resp.status >= 400:
            raise urllib.error.HTTPError(
                url,
                resp.status,
                resp.reason,
                dict(resp.getheaders()),
                None,
            )
        return _HttpResponse(status=resp.status, body=response_body)
    finally:
        conn.close()


def create_blueprint() -> Blueprint:
    """Return the HA and devices Blueprint."""
    bp = Blueprint("ha", __name__)
    _register_device_routes(bp)
    _register_ha_routes(bp)
    return bp


def _register_device_routes(bp: Blueprint) -> None:
    """Register device control API routes (US-060)."""

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
        headers = {"Content-Type": "application/json"}
        if ha_token:
            headers["Authorization"] = f"Bearer {ha_token}"

        try:
            resp = _request_home_assistant(
                service_url,
                method="POST",
                headers=headers,
                body=body,
                timeout=5,
            )
            ok = resp.status in (200, 201)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 200

        return jsonify({"ok": ok}), 200


def _register_ha_routes(bp: Blueprint) -> None:
    """Register Home Assistant setup and state API routes (US-059)."""

    @bp.route("/api/ha/test", methods=["POST"])
    def _ha_test_connection() -> Any:
        """Test a Home Assistant connection using the supplied URL and token.

        Requires a valid authenticated session.  Body: ``{ha_base_url, ha_token}``
        """
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

        try:
            _validate_http_url(base_url, field_name="ha_base_url")
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        api_url = f"{base_url}/api/"
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            resp = _request_home_assistant(api_url, headers=headers, timeout=5)
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
            from rex.routes._helpers import _log_nonfatal_exception

            _log_nonfatal_exception("Failed to reload app config after saving HA settings")

        return jsonify({"ok": True}), 200

    @bp.route("/api/ha/states", methods=["GET"])
    def _ha_get_states() -> Any:
        """Return all entity states from Home Assistant."""
        import json as _json

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

        ssl_ctx: ssl.SSLContext | None = None
        if not cfg.ha_verify_ssl:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        try:
            resp = _request_home_assistant(
                url,
                method="GET",
                headers=headers,
                timeout=cfg.ha_timeout,
                ssl_context=ssl_ctx,
            )
            raw_states: list[dict[str, Any]] = _json.loads(resp.body.decode())
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
