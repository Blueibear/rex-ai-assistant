"""Auth routes — /api/auth/*."""
from __future__ import annotations

from typing import Any

from flask import Blueprint


def create_blueprint() -> Blueprint:
    """Return the auth Blueprint."""
    bp = Blueprint("auth", __name__)

    @bp.route("/api/auth/register", methods=["POST"])
    def _auth_register() -> Any:
        """Register a new user. Body: {username, password}.

        When no users exist yet, requires the ``X-Setup-Token`` header to
        prevent a malicious local page from racing the first-run setup flow.
        After the first user is created the token check no longer applies.
        """
        from flask import jsonify, request

        from rex.auth import _open_db, create_user  # noqa: PLC2701
        from rex.routes._helpers import _require_setup_token

        try:
            with _open_db() as conn:
                row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
                user_count = row[0] if row else 0
        except Exception:
            user_count = 0

        if user_count == 0:
            _, token_err = _require_setup_token()
            if token_err is not None:
                return token_err

        data: dict[str, Any] = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        try:
            user = create_user(username, password)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

        try:
            from rex.identity import create_user_profile

            create_user_profile(user["id"], name=username)
        except Exception:
            pass

        try:
            from rex.permissions import bootstrap_admin_if_first_user

            bootstrap_admin_if_first_user(user["id"])
        except Exception:
            pass

        return jsonify({"id": user["id"], "username": user["username"]}), 201

    @bp.route("/api/auth/login", methods=["POST"])
    def _auth_login() -> Any:
        """Authenticate a user and return a JWT. Body: {username, password}."""
        from flask import jsonify, request

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

    @bp.route("/api/auth/logout", methods=["POST"])
    def _auth_logout() -> Any:
        """Logout endpoint — client should discard the token. Stateless."""
        from flask import jsonify

        return jsonify({"ok": True}), 200

    return bp
