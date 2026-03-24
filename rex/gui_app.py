"""GUI application launcher for Rex AI Assistant.

Starts the web-based dashboard on localhost and opens it in the default browser.
Accessible via the ``rex-gui`` entry point.
"""

from __future__ import annotations

import signal
import sys
import threading
import webbrowser
from typing import Any

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765


def _create_flask_app() -> Any:
    """Create a Flask application with stub dashboard routes.

    # OPENCLAW-REPLACE: legacy rex.dashboard blueprint replaced with stubs;
    # full UI will be served by the OpenClaw web UI.
    """
    import logging

    from flask import Flask, jsonify

    app = Flask(__name__)
    app.secret_key = "rex-gui-local"  # local-only; not security-sensitive

    @app.route("/dashboard")
    def _dashboard_stub() -> Any:
        logging.getLogger(__name__).warning(
            "Rex GUI stub: web UI not available (migrating to OpenClaw)"
        )
        return (
            "<h1>Rex Dashboard</h1><p>Dashboard is migrating to the OpenClaw backend.</p>",
            200,
        )

    @app.route("/api/dashboard/status")
    def _dashboard_status_stub() -> Any:
        return jsonify({"status": "stub", "message": "Dashboard migrating to OpenClaw"}), 200

    return app


def _open_browser(host: str, port: int) -> None:
    """Open the dashboard in the default browser after a short delay."""
    import time

    time.sleep(0.8)
    url = f"http://{host}:{port}/dashboard"
    webbrowser.open(url)


def main() -> None:
    """Entry point for ``rex-gui``.  Starts the dashboard and opens the browser."""
    import logging

    logging.basicConfig(level=logging.WARNING)

    host = _DEFAULT_HOST
    port = _DEFAULT_PORT

    app = _create_flask_app()

    # Open the browser in a background thread so the server starts first.
    browser_thread = threading.Thread(target=_open_browser, args=(host, port), daemon=True)
    browser_thread.start()

    # Allow Ctrl-C to shut down cleanly.
    def _handle_sigint(sig: int, frame: Any) -> None:  # pragma: no cover
        print("\nShutting down Rex GUI…", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"Rex GUI starting at http://{host}:{port}/dashboard", file=sys.stderr)

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except SystemExit:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
