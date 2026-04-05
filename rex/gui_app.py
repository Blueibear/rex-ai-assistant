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


def _create_flask_app(ui_enabled: bool = True) -> Any:
    """Create a Flask application serving the Rex web UI and API stubs."""

    from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

    app = Flask(__name__, static_folder=None)
    app.secret_key = "rex-gui-local"  # local-only; not security-sensitive

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
        from rex import dashboard_store as ds

        return jsonify(ds.get_history()), 200

    @app.route("/api/chat/clear", methods=["POST"])
    def _chat_clear() -> Any:
        from rex import dashboard_store as ds

        ds.clear_history()
        return jsonify({"ok": True}), 200

    @app.route("/api/chat/send", methods=["POST"])
    def _chat_send() -> Any:
        from rex import dashboard_store as ds

        data: dict[str, Any] = request.get_json(silent=True) or {}
        user_text = (data.get("message") or "").strip()
        attachment_name: str | None = data.get("filename") or None

        if not user_text:
            return jsonify({"error": "empty message"}), 400

        ds.add_message("user", user_text, attachment_name)

        def _stream() -> Any:
            reply = _generate_reply(user_text)
            ds.add_message("assistant", reply)
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
