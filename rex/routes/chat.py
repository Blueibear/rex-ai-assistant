"""Chat routes — /api/chat/*."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint


def create_blueprint(history_store: Any) -> Blueprint:
    """Return the chat Blueprint.

    Args:
        history_store: A ``HistoryStore`` instance for persisting chat turns.
    """
    bp = Blueprint("chat", __name__)

    @bp.route("/api/chat/history")
    def _chat_history() -> Any:
        """Return chat history for the authenticated user."""
        from flask import jsonify

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None
        turns = history_store.load_history(user["id"])
        return jsonify(turns), 200

    @bp.route("/api/chat/clear", methods=["POST"])
    def _chat_clear() -> Any:
        """Clear chat history for the authenticated user."""
        from flask import jsonify

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None
        history_store.clear_history(user["id"])
        return jsonify({"ok": True}), 200

    @bp.route("/api/chat/send", methods=["POST"])
    def _chat_send() -> Any:
        """Send a message and receive a streamed SSE reply."""
        from flask import Response, jsonify, request, stream_with_context

        from rex.gui_app import _generate_reply
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None

        data: dict[str, Any] = request.get_json(silent=True) or {}
        user_text = (data.get("message") or "").strip()

        if not user_text:
            return jsonify({"error": "empty message"}), 400

        from datetime import UTC, datetime

        history_store.save_turn(user["id"], "user", user_text, datetime.now(UTC))

        def _stream() -> Any:
            from datetime import UTC, datetime

            reply = _generate_reply(user_text, user_id=str(user["username"]))
            history_store.save_turn(user["id"], "assistant", reply, datetime.now(UTC))
            payload = json.dumps({"content": reply, "done": True})
            yield f"data: {payload}\n\n"

        return Response(
            stream_with_context(_stream()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return bp
