"""Status, quick-actions, history, and usage routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint


def create_blueprint() -> Blueprint:
    """Return the status/actions Blueprint."""
    bp = Blueprint("status", __name__)

    # ------------------------------------------------------------------
    # Status / SSE API (US-062)
    # ------------------------------------------------------------------

    @bp.route("/api/status/current", methods=["GET"])
    def _status_current() -> Any:
        """Return the current Rex status (public, no auth required)."""
        from flask import jsonify

        from rex.dashboard.sse import get_current_status

        return jsonify({"status": get_current_status()}), 200

    @bp.route("/api/status/stream", methods=["GET"])
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
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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

    @bp.route("/api/quick-actions", methods=["GET"])
    def _list_quick_actions() -> Any:
        """Return the authenticated user's quick actions."""
        from flask import jsonify

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None
        return jsonify({"quick_actions": _get_quick_actions(user["id"])}), 200

    @bp.route("/api/quick-actions", methods=["POST"])
    def _add_quick_action() -> Any:
        """Add a quick action.  Body: ``{label: str, command: str}``."""
        import uuid

        from flask import jsonify, request

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None

        data: dict[str, Any] = request.get_json(silent=True) or {}
        label = (data.get("label") or "").strip()
        command = (data.get("command") or "").strip()

        if not label or not command:
            return jsonify({"error": "label and command are required"}), 400

        actions = _get_quick_actions(user["id"])
        new_action: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "label": label,
            "command": command,
        }
        actions.append(new_action)
        _save_quick_actions(user["id"], actions)
        return jsonify(new_action), 201

    @bp.route("/api/quick-actions/<action_id>", methods=["DELETE"])
    def _delete_quick_action(action_id: str) -> Any:
        """Remove a quick action by id."""
        from flask import jsonify

        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None

        actions = _get_quick_actions(user["id"])
        new_actions = [a for a in actions if a.get("id") != action_id]
        if len(new_actions) == len(actions):
            return jsonify({"error": "not found"}), 404
        _save_quick_actions(user["id"], new_actions)
        return jsonify({"ok": True}), 200

    @bp.route("/api/quick-actions/<action_id>/run", methods=["POST"])
    def _run_quick_action(action_id: str) -> Any:
        """Execute a quick action by sending its command to the assistant."""
        from flask import jsonify

        from rex.gui_app import _generate_reply
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None

        actions = _get_quick_actions(user["id"])
        action = next((a for a in actions if a.get("id") == action_id), None)
        if action is None:
            return jsonify({"error": "not found"}), 404

        reply = _generate_reply(action["command"])
        return jsonify({"reply": reply}), 200

    # ------------------------------------------------------------------
    # Command history API (US-061)
    # ------------------------------------------------------------------

    @bp.route("/api/history", methods=["GET"])
    def _command_history() -> Any:
        """Return recent command history.  Requires auth."""
        from flask import jsonify, request

        from rex.command_history import CommandHistoryStore
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err

        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50

        store = CommandHistoryStore()
        entries = store.get_recent(limit=limit, user_id=str(user["username"]))
        return jsonify({"history": entries}), 200

    # ------------------------------------------------------------------
    # Usage API (US-046)
    # ------------------------------------------------------------------

    @bp.route("/api/usage")
    def _usage_summary() -> Any:
        """Return local vs cloud LLM usage split by period."""
        from flask import jsonify

        from rex.llm_usage import usage_api_summary

        return jsonify(usage_api_summary()), 200

    return bp
