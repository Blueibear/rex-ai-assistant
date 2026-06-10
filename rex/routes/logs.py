"""Log streaming/download routes — /api/logs/*."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint


def create_blueprint(log_file: Path) -> Blueprint:
    """Return the logs Blueprint.

    Args:
        log_file: Path to the active runtime log file to stream/download.
    """
    bp = Blueprint("logs", __name__)

    @bp.route("/api/logs/stream")
    def _logs_stream() -> Any:
        """SSE endpoint that tails the active runtime log in real time.

        Requires a valid Bearer token. Home-directory paths are redacted
        from every streamed line before it is sent to the client.
        """
        import time

        from flask import Response, stream_with_context

        from rex.routes._helpers import _redact_log_line, _require_auth

        user, err = _require_auth()
        if err:
            return err

        def _generate() -> Any:
            if not log_file.exists():
                payload = {"level": "INFO", "message": "Active log file not found yet."}
                yield f"data: {json.dumps(payload)}\n\n"
                return
            with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)  # seek to end
                while True:
                    line = fh.readline()
                    if line:
                        line = _redact_log_line(line.strip())
                        if line:
                            yield f"data: {line}\n\n"
                    else:
                        time.sleep(0.25)

        return Response(
            stream_with_context(_generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @bp.route("/api/logs/download")
    def _logs_download() -> Any:
        """Download the current log file with home-directory paths redacted.

        Requires a valid Bearer token.  The file is streamed line-by-line so
        that home-directory paths are replaced with ``~`` before delivery.
        """
        from flask import Response, jsonify, stream_with_context

        from rex.routes._helpers import _redact_log_line, _require_auth

        user, err = _require_auth()
        if err:
            return err

        if not log_file.exists():
            return jsonify({"error": "Active log file not found"}), 404

        def _redacted_stream() -> Any:
            with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    yield _redact_log_line(line)

        return Response(
            stream_with_context(_redacted_stream()),
            content_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={log_file.name}",
                "Cache-Control": "no-cache",
            },
        )

    return bp
