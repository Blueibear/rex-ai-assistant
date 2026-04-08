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
