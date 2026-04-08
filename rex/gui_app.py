"""GUI application launcher for Rex AI Assistant.

Starts the web-based dashboard on localhost and opens it in the default browser.
Accessible via the ``rex-gui`` entry point.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from rex.audio.speaker_discovery import start_smart_speaker_discovery

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765

# Path to the pre-built React UI (rex/ui/dist/)
_UI_DIST = Path(__file__).parent / "ui" / "dist"


def _resolve_server_port() -> int:
    """Return the dashboard port, allowing a simple env override."""
    raw_port = os.getenv("REX_GUI_PORT")
    if raw_port is None:
        return _DEFAULT_PORT

    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("REX_GUI_PORT must be an integer") from exc

    if not 1 <= port <= 65535:
        raise ValueError("REX_GUI_PORT must be between 1 and 65535")

    return port


def _write_env_secrets(
    env_path: Path,
    *,
    llm_provider: str,
    llm_api_key: str,
    ha_token: str,
) -> None:
    """Write or update secrets in an .env file without overwriting unrelated lines.

    Only lines whose keys are managed here are modified; all other lines are
    left untouched.
    """
    managed: dict[str, str] = {}
    if llm_provider == "openai" and llm_api_key:
        managed["OPENAI_API_KEY"] = llm_api_key
    elif llm_provider == "anthropic" and llm_api_key:
        managed["ANTHROPIC_API_KEY"] = llm_api_key
    if ha_token:
        managed["HA_TOKEN"] = ha_token

    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated: list[str] = []
    seen_keys: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            updated.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in managed:
            updated.append(f"{key}={managed[key]}")
            seen_keys.add(key)
        else:
            updated.append(line)

    for key, value in managed.items():
        if key not in seen_keys:
            updated.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _require_auth() -> tuple[dict[str, Any], None] | tuple[None, Any]:
    """Extract and validate the Bearer token from the current request.

    Returns:
        ``(user_dict, None)`` on success.
        ``(None, flask_response)`` on failure (caller should return the response).
    """
    from flask import jsonify, request

    from rex.auth import get_current_user

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "authentication required"}), 401)
    token = auth_header[len("Bearer ") :]
    try:
        return get_current_user(token), None
    except ValueError as exc:
        return None, (jsonify({"error": str(exc)}), 401)


def _create_flask_app(ui_enabled: bool = True) -> Any:
    """Create a Flask application serving the Rex web UI and API stubs."""

    from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

    app = Flask(__name__, static_folder=None)
    app.secret_key = "rex-gui-local"  # local-only; not security-sensitive

    data_dir = Path(os.getenv("REX_DATA_DIR", "data"))
    from rex.history_store import HistoryStore

    _history_store = HistoryStore(db_path=data_dir / "history.db")

    if ui_enabled and _UI_DIST.is_dir():

        @app.route("/ui/", defaults={"filename": "index.html"})
        @app.route("/ui/<path:filename>")
        def _serve_ui(filename: str) -> Any:
            return send_from_directory(str(_UI_DIST), filename)

    else:

        @app.route("/ui/")
        def _ui_disabled() -> Any:
            return "<h1>Rex UI</h1><p>UI is disabled or not built.</p>", 200

    @app.route("/dashboard")
    def _dashboard_redirect() -> Any:
        from flask import redirect

        return redirect("/ui/")

    @app.route("/api/dashboard/status")
    def _dashboard_status_stub() -> Any:
        return jsonify({"status": "ok"}), 200

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------

    @app.route("/api/chat/history")
    def _chat_history() -> Any:
        user, err = _require_auth()
        if err:
            return err
        turns = _history_store.load_history(user["id"])
        return jsonify(turns), 200

    @app.route("/api/chat/clear", methods=["POST"])
    def _chat_clear() -> Any:
        user, err = _require_auth()
        if err:
            return err
        _history_store.clear_history(user["id"])
        return jsonify({"ok": True}), 200

    @app.route("/api/chat/send", methods=["POST"])
    def _chat_send() -> Any:
        user, err = _require_auth()
        if err:
            return err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        user_text = (data.get("message") or "").strip()

        if not user_text:
            return jsonify({"error": "empty message"}), 400

        from datetime import UTC, datetime

        _history_store.save_turn(user["id"], "user", user_text, datetime.now(UTC))

        def _stream() -> Any:
            from datetime import UTC, datetime

            reply = _generate_reply(user_text)
            _history_store.save_turn(user["id"], "assistant", reply, datetime.now(UTC))
            payload = json.dumps({"content": reply, "done": True})
            yield f"data: {payload}\n\n"

        return Response(
            stream_with_context(_stream()),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Logs API
    # ------------------------------------------------------------------

    _LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "rex.log"

    @app.route("/api/logs/stream")
    def _logs_stream() -> Any:
        """SSE endpoint that tails logs/rex.log in real time."""
        import time

        def _generate() -> Any:
            if not _LOG_FILE.exists():
                yield f"data: {json.dumps({'level': 'INFO', 'message': 'Log file not found yet.'})}\n\n"
                return
            with _LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)  # seek to end
                while True:
                    line = fh.readline()
                    if line:
                        line = line.strip()
                        if line:
                            yield f"data: {line}\n\n"
                    else:
                        time.sleep(0.25)

        return Response(
            stream_with_context(_generate()),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/logs/download")
    def _logs_download() -> Any:
        """Download the current log file."""
        if not _LOG_FILE.exists():
            return jsonify({"error": "Log file not found"}), 404
        return send_from_directory(
            str(_LOG_FILE.parent),
            _LOG_FILE.name,
            as_attachment=True,
            download_name="rex.log",
        )

    # ------------------------------------------------------------------
    # Usage API (US-046)
    # ------------------------------------------------------------------

    @app.route("/api/usage")
    def _usage_summary() -> Any:
        """Return local vs cloud LLM usage split by period."""
        from rex.llm_usage import usage_api_summary

        return jsonify(usage_api_summary()), 200

    # ------------------------------------------------------------------
    # Setup wizard API (US-058)
    # ------------------------------------------------------------------

    @app.route("/api/setup/status", methods=["GET"])
    def _setup_status() -> Any:
        """Return whether the initial setup wizard needs to run.

        ``needs_setup`` is True when no users exist in the database yet.
        """
        from rex.auth import _open_db  # noqa: PLC2701

        try:
            with _open_db() as conn:
                row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
                count = row[0] if row else 0
        except Exception:
            count = 0
        return jsonify({"needs_setup": count == 0}), 200

    @app.route("/api/setup/complete", methods=["POST"])
    def _setup_complete() -> Any:
        """Complete the first-run wizard.

        Accepts JSON body::

            {
              "username":     str,
              "password":     str,
              "llm_provider": "local" | "openai" | "anthropic" | "ollama",
              "llm_api_key":  str (optional),
              "tts_provider": "none" | "edge" | "pyttsx3" | "xtts",
              "ha_base_url":  str (optional),
              "ha_token":     str (optional)
            }

        Registers the user, writes non-secret settings to
        ``config/rex_config.json``, and writes secrets to ``.env``.
        """
        from rex.auth import create_user

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

        # Check that setup hasn't already been completed.
        try:
            from rex.auth import _open_db  # noqa: PLC2701

            with _open_db() as conn:
                row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
                if row and row[0] > 0:
                    return (
                        jsonify({"error": "setup already completed"}),
                        409,
                    )
        except Exception:
            pass

        # Create the admin user.
        try:
            user = create_user(username, password)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

        try:
            from rex.permissions import bootstrap_admin_if_first_user

            bootstrap_admin_if_first_user(user["id"])
        except Exception:
            pass

        # Write non-secret runtime settings to config/rex_config.json.
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
            pass

        # Write secrets to .env (append / overwrite existing lines).
        try:
            from rex.bridge_utils import repo_root

            env_path = repo_root() / ".env"
        except Exception:
            env_path = Path(".env")

        _write_env_secrets(
            env_path,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            ha_token=ha_token,
        )

        return jsonify({"ok": True, "user_id": user["id"]}), 201

    # ------------------------------------------------------------------
    # Auth API (US-047)
    # ------------------------------------------------------------------

    @app.route("/api/auth/register", methods=["POST"])
    def _auth_register() -> Any:
        """Register a new user. Body: {username, password}."""
        from rex.auth import create_user

        data: dict[str, Any] = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        try:
            user = create_user(username, password)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

        # Create a memory profile for the new user (best-effort).
        try:
            from rex.identity import create_user_profile

            create_user_profile(user["id"], name=username)
        except Exception:
            pass

        # Grant admin to the first registered user (best-effort).
        try:
            from rex.permissions import bootstrap_admin_if_first_user

            bootstrap_admin_if_first_user(user["id"])
        except Exception:
            pass

        return jsonify({"id": user["id"], "username": user["username"]}), 201

    # ------------------------------------------------------------------
    # Permissions API (US-052)
    # ------------------------------------------------------------------

    @app.route("/api/user/permissions", methods=["GET"])
    def _get_my_permissions() -> Any:
        """Return the authenticated user's permissions."""
        user, err = _require_auth()
        if err:
            return err
        from rex.permissions import get_permissions

        return jsonify({"permissions": get_permissions(user["id"])}), 200

    @app.route("/api/admin/permissions/grant", methods=["POST"])
    def _admin_grant_permission() -> Any:
        """Grant a permission to a user. Requires admin. Body: {user_id, permission}."""
        user, err = _require_auth()
        if err:
            return err
        from rex.permissions import Permission, check_permission, grant_permission

        if not check_permission(user["id"], Permission.admin):
            return jsonify({"error": "forbidden: admin permission required"}), 403

        data: dict[str, Any] = request.get_json(silent=True) or {}
        target_user_id = (data.get("user_id") or "").strip()
        permission_str = (data.get("permission") or "").strip()

        if not target_user_id or not permission_str:
            return jsonify({"error": "user_id and permission are required"}), 400

        try:
            grant_permission(target_user_id, permission_str)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({"ok": True}), 200

    @app.route("/api/admin/permissions/revoke", methods=["POST"])
    def _admin_revoke_permission() -> Any:
        """Revoke a permission from a user. Requires admin. Body: {user_id, permission}."""
        user, err = _require_auth()
        if err:
            return err
        from rex.permissions import Permission, check_permission, revoke_permission

        if not check_permission(user["id"], Permission.admin):
            return jsonify({"error": "forbidden: admin permission required"}), 403

        data: dict[str, Any] = request.get_json(silent=True) or {}
        target_user_id = (data.get("user_id") or "").strip()
        permission_str = (data.get("permission") or "").strip()

        if not target_user_id or not permission_str:
            return jsonify({"error": "user_id and permission are required"}), 400

        try:
            revoke_permission(target_user_id, permission_str)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({"ok": True}), 200

    # ------------------------------------------------------------------
    # Personality API (US-051)
    # ------------------------------------------------------------------

    @app.route("/api/personalities", methods=["GET"])
    def _list_personalities() -> Any:
        """Return available personalities with name, greeting, and tone keywords."""
        from rex.personality import list_personalities

        return (
            jsonify(
                [
                    {
                        "name": p.name,
                        "greeting": p.greeting,
                        "tone_keywords": p.tone_keywords,
                    }
                    for p in list_personalities()
                ]
            ),
            200,
        )

    # User preferences API (US-048)
    # ------------------------------------------------------------------

    @app.route("/api/user/preferences", methods=["GET"])
    def _get_preferences() -> Any:
        """Return the authenticated user's stored preferences."""
        user, err = _require_auth()
        if err:
            return err
        from rex.identity import get_user_profile

        profile = get_user_profile(user["id"])
        prefs = profile.get("preferences", {}) if profile else {}
        return jsonify(prefs), 200

    @app.route("/api/user/preferences", methods=["PATCH"])
    def _patch_preferences() -> Any:
        """Merge the request body into the authenticated user's preferences."""
        user, err = _require_auth()
        if err:
            return err
        updates: dict[str, Any] = request.get_json(silent=True) or {}
        from rex.identity import create_user_profile, get_user_profile, update_user_preferences

        # Ensure a profile exists before updating.
        if get_user_profile(user["id"]) is None:
            try:
                create_user_profile(user["id"], name=user["username"])
            except Exception:
                pass
        update_user_preferences(user["id"], updates)
        return jsonify({"ok": True}), 200

    # ------------------------------------------------------------------
    # Avatar API (US-049)
    # ------------------------------------------------------------------

    _AVATAR_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
    _AVATAR_SIZE = (256, 256)
    _AVATAR_DIR = data_dir / "avatars"
    _DEFAULT_AVATAR_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">'
        '<circle cx="128" cy="128" r="128" fill="#4f46e5"/>'
        '<text x="128" y="165" font-family="sans-serif" font-size="120" '
        'fill="white" text-anchor="middle">R</text>'
        "</svg>"
    )

    @app.route("/api/user/avatar", methods=["POST"])
    def _upload_avatar() -> Any:
        """Upload (or replace) the authenticated user's profile picture."""
        import io

        from PIL import Image

        user, err = _require_auth()
        if err:
            return err

        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400

        upload = request.files["file"]
        content_type = (upload.content_type or "").split(";")[0].strip()
        if content_type not in ("image/jpeg", "image/png"):
            return jsonify({"error": "only JPEG and PNG are accepted"}), 415

        raw = upload.read(_AVATAR_MAX_BYTES + 1)
        if len(raw) > _AVATAR_MAX_BYTES:
            return jsonify({"error": "file too large (max 2 MB)"}), 413

        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return jsonify({"error": "invalid image file"}), 400

        img = img.resize(_AVATAR_SIZE, Image.LANCZOS)

        _AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        avatar_path = _AVATAR_DIR / f"{user['id']}.jpg"
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        avatar_path.write_bytes(buf.getvalue())

        return jsonify({"ok": True}), 200

    @app.route("/api/user/avatar", methods=["GET"])
    def _get_avatar() -> Any:
        """Return the user's profile picture, or a default avatar."""
        from flask import send_file

        user, _ = _require_auth()
        if user is not None:
            avatar_path = _AVATAR_DIR / f"{user['id']}.jpg"
            if avatar_path.is_file():
                return send_file(str(avatar_path), mimetype="image/jpeg")

        return Response(_DEFAULT_AVATAR_SVG, mimetype="image/svg+xml")

    @app.route("/api/auth/login", methods=["POST"])
    def _auth_login() -> Any:
        """Authenticate a user and return a JWT. Body: {username, password}."""
        from rex.auth import authenticate

        data: dict[str, Any] = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        try:
            token = authenticate(username, password)
        except ValueError:
            return jsonify({"error": "invalid username or password"}), 401

        return jsonify({"token": token}), 200

    @app.route("/api/auth/logout", methods=["POST"])
    def _auth_logout() -> Any:
        """Logout endpoint — client should discard the token. Stateless."""
        return jsonify({"ok": True}), 200

    # ------------------------------------------------------------------
    # Device control API (US-060)
    # ------------------------------------------------------------------

    @app.route("/api/devices", methods=["GET"])
    def _list_devices() -> Any:
        """Return approved devices from config/device_aliases.json."""
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

    @app.route("/api/devices/<path:entity_id>/command", methods=["POST"])
    def _device_command(entity_id: str) -> Any:
        """Send a command to a Home Assistant entity.

        Requires auth.  Body: ``{command: str, value?: any}``

        Commands:
        - ``turn_on``  / ``turn_off``  — lights, switches
        - ``set_brightness`` (0–255)   — lights
        - ``media_play`` / ``media_pause`` / ``media_next_track`` — media players
        - ``volume_set`` (0.0–1.0)     — media players
        """
        import urllib.request

        user, err = _require_auth()
        if err:
            return err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        command = (data.get("command") or "").strip()
        value = data.get("value")

        if not command:
            return jsonify({"error": "command is required"}), 400

        # Map command → HA service domain + service.
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

    @app.route("/api/ha/test", methods=["POST"])
    def _ha_test_connection() -> Any:
        """Test a Home Assistant connection using the supplied URL and token.

        Body: ``{ha_base_url: str, ha_token: str}``

        Returns ``{ok: bool, error?: str}``; does **not** require auth so the
        setup wizard can call it before the first user is created.
        """
        import urllib.request

        data: dict[str, Any] = request.get_json(silent=True) or {}
        base_url = (data.get("ha_base_url") or "").rstrip("/")
        token = (data.get("ha_token") or "").strip()

        if not base_url:
            return jsonify({"ok": False, "error": "ha_base_url is required"}), 400

        api_url = f"{base_url}/api/"
        req = urllib.request.Request(api_url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                ok = resp.status == 200
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 200

        return jsonify({"ok": ok}), 200

    @app.route("/api/ha/save", methods=["POST"])
    def _ha_save_config() -> Any:
        """Save Home Assistant URL and token.  Requires auth.

        Body: ``{ha_base_url: str, ha_token?: str}``

        Writes ``ha_base_url`` to ``config/rex_config.json`` (non-secret) and
        ``ha_token`` to ``.env`` (secret).
        """
        user, err = _require_auth()
        if err:
            return err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        base_url = (data.get("ha_base_url") or "").strip()
        token = (data.get("ha_token") or "").strip()

        if not base_url:
            return jsonify({"error": "ha_base_url is required"}), 400

        # Persist non-secret settings to rex_config.json.
        try:
            from rex.config_manager import load_config as _load_json_cfg
            from rex.config_manager import save_config as _save_json_cfg

            json_cfg: dict[str, Any] = _load_json_cfg() or {}
            json_cfg.setdefault("home_assistant", {})["base_url"] = base_url
            _save_json_cfg(json_cfg)
        except Exception as exc:
            return jsonify({"error": f"failed to save config: {exc}"}), 500

        # Persist secret token to .env.
        if token:
            try:
                from rex.bridge_utils import repo_root

                env_path = repo_root() / ".env"
            except Exception:
                env_path = Path(".env")

            _write_env_secrets(env_path, llm_provider="", llm_api_key="", ha_token=token)

        return jsonify({"ok": True}), 200

    # ------------------------------------------------------------------
    # Quick actions API (US-063)
    # ------------------------------------------------------------------

    def _get_quick_actions(user_id: str) -> list[dict[str, Any]]:
        """Return the quick actions list from the user's profile."""
        from rex.identity import get_user_profile

        profile = get_user_profile(user_id) or {}
        prefs = profile.get("preferences", {})
        actions = prefs.get("quick_actions", [])
        return actions if isinstance(actions, list) else []

    def _save_quick_actions(user_id: str, actions: list[dict[str, Any]]) -> None:
        """Persist the quick actions list to the user's profile."""
        from rex.identity import update_user_preferences

        update_user_preferences(user_id, {"quick_actions": actions})

    @app.route("/api/quick-actions", methods=["GET"])
    def _list_quick_actions() -> Any:
        """Return the authenticated user's quick actions."""
        user, err = _require_auth()
        if err:
            return err
        return jsonify({"quick_actions": _get_quick_actions(user["id"])}), 200

    @app.route("/api/quick-actions", methods=["POST"])
    def _add_quick_action() -> Any:
        """Add a quick action.  Body: ``{label: str, command: str}``."""
        user, err = _require_auth()
        if err:
            return err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        label = (data.get("label") or "").strip()
        command = (data.get("command") or "").strip()

        if not label or not command:
            return jsonify({"error": "label and command are required"}), 400

        import uuid

        actions = _get_quick_actions(user["id"])
        new_action: dict[str, Any] = {"id": str(uuid.uuid4()), "label": label, "command": command}
        actions.append(new_action)
        _save_quick_actions(user["id"], actions)
        return jsonify(new_action), 201

    @app.route("/api/quick-actions/<action_id>", methods=["DELETE"])
    def _delete_quick_action(action_id: str) -> Any:
        """Remove a quick action by id."""
        user, err = _require_auth()
        if err:
            return err

        actions = _get_quick_actions(user["id"])
        new_actions = [a for a in actions if a.get("id") != action_id]
        if len(new_actions) == len(actions):
            return jsonify({"error": "not found"}), 404
        _save_quick_actions(user["id"], new_actions)
        return jsonify({"ok": True}), 200

    @app.route("/api/quick-actions/<action_id>/run", methods=["POST"])
    def _run_quick_action(action_id: str) -> Any:
        """Execute a quick action by sending its command to the assistant."""
        user, err = _require_auth()
        if err:
            return err

        actions = _get_quick_actions(user["id"])
        action = next((a for a in actions if a.get("id") == action_id), None)
        if action is None:
            return jsonify({"error": "not found"}), 404

        reply = _generate_reply(action["command"])
        return jsonify({"reply": reply}), 200

    # ------------------------------------------------------------------
    # Status / SSE API (US-062)
    # ------------------------------------------------------------------

    @app.route("/api/status/current", methods=["GET"])
    def _status_current() -> Any:
        """Return the current Rex status (public, no auth required)."""
        from rex.dashboard.sse import get_current_status

        return jsonify({"status": get_current_status()}), 200

    @app.route("/api/status/stream", methods=["GET"])
    def _status_stream() -> Any:
        """Stream Rex status changes as Server-Sent Events."""
        from flask import Response

        from rex.dashboard.sse import get_current_status, subscription

        def _generate() -> Any:
            yield f"data: {get_current_status()}\n\n"
            with subscription() as client_q:
                while True:
                    try:
                        status = client_q.get(timeout=30)
                        yield f"data: {status}\n\n"
                    except Exception:
                        yield ": ping\n\n"

        return Response(
            _generate(),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Command history API (US-061)
    # ------------------------------------------------------------------

    @app.route("/api/history", methods=["GET"])
    def _command_history() -> Any:
        """Return recent command history.  Requires auth.

        Query params:
            limit (int): Max entries to return (1–500, default 50).
        """
        user, err = _require_auth()
        if err:
            return err

        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50

        from rex.command_history import CommandHistoryStore

        store = CommandHistoryStore()
        entries = store.get_recent(limit=limit)
        return jsonify({"history": entries}), 200

    # ------------------------------------------------------------------
    # Integrations API (US-057)
    # ------------------------------------------------------------------

    @app.route("/api/integrations", methods=["GET"])
    def _list_integrations() -> Any:
        """Return configured integrations with their status (public)."""
        from rex.config import load_config

        try:
            cfg = load_config()
        except Exception:
            return jsonify({"integrations": []}), 200

        integrations = [
            {
                "name": "Home Assistant",
                "key": "home_assistant",
                "configured": bool(cfg.ha_base_url and cfg.ha_token),
            },
            {
                "name": "OpenAI",
                "key": "openai",
                "configured": bool(cfg.openai_api_key),
            },
            {
                "name": "Anthropic",
                "key": "anthropic",
                "configured": bool(cfg.anthropic_api_key),
            },
            {
                "name": "Ollama",
                "key": "ollama",
                "configured": bool(cfg.ollama_base_url),
            },
            {
                "name": "Telegram",
                "key": "telegram",
                "configured": bool(cfg.telegram_bot_token and cfg.telegram_chat_id),
            },
            {
                "name": "Music Assistant",
                "key": "music_assistant",
                "configured": bool(cfg.music_assistant_url),
            },
            {
                "name": "Email",
                "key": "email",
                "configured": cfg.email_provider not in ("none", ""),
            },
            {
                "name": "Push Notifications",
                "key": "push",
                "configured": bool(cfg.push_provider and cfg.push_token),
            },
        ]
        return jsonify({"integrations": integrations}), 200

    @app.route("/api/capabilities", methods=["GET"])
    def _list_capabilities() -> Any:
        """Return all capabilities from the capability registry (public)."""
        try:
            from rex.capabilities.registry import get_capability_registry

            registry = get_capability_registry()
            caps = [
                {
                    "name": c.name,
                    "description": c.description,
                    "category": getattr(c, "category", "General"),
                    "enabled": c.enabled,
                }
                for c in registry.list()
            ]
        except Exception:
            caps = []
        return jsonify({"capabilities": caps}), 200

    @app.route("/api/tools", methods=["GET"])
    def _list_tools() -> Any:
        """Return registered tools with health status."""
        _, err = _require_auth()
        if err is not None:
            return err
        try:
            from rex.openclaw.tool_registry import get_tool_registry

            registry = get_tool_registry()
            tool_list = registry.list_tools(include_disabled=True)
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "capabilities": t.capabilities,
                    "enabled": t.enabled,
                    "version": t.version,
                }
                for t in tool_list
            ]
        except Exception:
            tools = []
        return jsonify({"tools": tools}), 200

    return app


def _generate_reply(user_text: str) -> str:
    """Generate an LLM reply, falling back to an echo stub on any failure."""
    try:
        from rex.config import load_config
        from rex.llm_client import LanguageModel

        cfg = load_config()
        llm = LanguageModel(config=cfg)
        messages = [{"role": "user", "content": user_text}]
        return llm.generate(messages=messages)
    except Exception:
        return f"(Rex is not configured — echo) {user_text}"


def _open_browser(host: str, port: int) -> None:
    """Open the dashboard in the default browser after a short delay."""
    import time

    time.sleep(0.8)
    url = f"http://{host}:{port}/ui/"
    webbrowser.open(url)


def main() -> None:
    """Entry point for ``rex-gui``.  Starts the dashboard and opens the browser."""
    import logging

    logging.basicConfig(level=logging.WARNING)

    host = _DEFAULT_HOST
    port = _resolve_server_port()

    try:
        from rex.config import load_config

        cfg = load_config()
        ui_enabled = cfg.ui_enabled
    except Exception:
        ui_enabled = True

    app = _create_flask_app(ui_enabled=ui_enabled)
    start_smart_speaker_discovery()

    # Open the browser in a background thread so the server starts first.
    browser_thread = threading.Thread(target=_open_browser, args=(host, port), daemon=True)
    browser_thread.start()

    # Allow Ctrl-C to shut down cleanly.
    def _handle_sigint(sig: int, frame: Any) -> None:  # pragma: no cover
        print("\nShutting down Rex GUI…", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"Rex GUI starting at http://{host}:{port}/ui/", file=sys.stderr)

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except SystemExit:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
