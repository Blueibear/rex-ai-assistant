"""User, permissions, personality, preferences, and avatar routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint

_AVATAR_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
_AVATAR_SIZE = (256, 256)
_DEFAULT_AVATAR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">'
    '<circle cx="128" cy="128" r="128" fill="#4f46e5"/>'
    '<text x="128" y="165" font-family="sans-serif" font-size="120" '
    'fill="white" text-anchor="middle">R</text>'
    "</svg>"
)


def create_blueprint(avatar_dir: Path) -> Blueprint:
    """Return the users Blueprint.

    Args:
        avatar_dir: Directory where user avatar images are stored.
    """
    bp = Blueprint("users", __name__)

    # ------------------------------------------------------------------
    # Permissions API (US-052)
    # ------------------------------------------------------------------

    @bp.route("/api/user/permissions", methods=["GET"])
    def _get_my_permissions() -> Any:
        """Return the authenticated user's permissions."""
        from flask import jsonify

        from rex.permissions import get_permissions
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None
        return jsonify({"permissions": get_permissions(user["id"])}), 200

    @bp.route("/api/admin/permissions/grant", methods=["POST"])
    def _admin_grant_permission() -> Any:
        """Grant a permission to a user. Requires admin. Body: {user_id, permission}."""
        from flask import jsonify, request

        from rex.permissions import Permission, check_permission, grant_permission
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None

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

    @bp.route("/api/admin/permissions/revoke", methods=["POST"])
    def _admin_revoke_permission() -> Any:
        """Revoke a permission from a user. Requires admin. Body: {user_id, permission}."""
        from flask import jsonify, request

        from rex.permissions import Permission, check_permission, revoke_permission
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None

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

    @bp.route("/api/personalities", methods=["GET"])
    def _list_personalities() -> Any:
        """Return available personalities with name, greeting, and tone keywords."""
        from flask import jsonify

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

    # ------------------------------------------------------------------
    # User preferences API (US-048)
    # ------------------------------------------------------------------

    @bp.route("/api/user/preferences", methods=["GET"])
    def _get_preferences() -> Any:
        """Return the authenticated user's stored preferences."""
        from flask import jsonify

        from rex.identity import get_user_profile
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None
        profile = get_user_profile(user["id"])
        prefs = profile.get("preferences", {}) if profile else {}
        return jsonify(prefs), 200

    @bp.route("/api/user/preferences", methods=["PATCH"])
    def _patch_preferences() -> Any:
        """Merge the request body into the authenticated user's preferences."""
        from flask import jsonify, request

        from rex.identity import create_user_profile, get_user_profile, update_user_preferences
        from rex.routes._helpers import _require_auth

        user, err = _require_auth()
        if err:
            return err
        assert user is not None
        updates: dict[str, Any] = request.get_json(silent=True) or {}

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

    @bp.route("/api/user/avatar", methods=["POST"])
    def _upload_avatar() -> Any:
        """Upload (or replace) the authenticated user's profile picture."""
        import io

        from flask import jsonify, request

        from rex.routes._helpers import _require_auth

        try:
            from PIL import Image
        except ImportError:
            return jsonify({"error": "Pillow is not installed"}), 503

        user, err = _require_auth()
        if err:
            return err
        assert user is not None

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

        img = img.resize(_AVATAR_SIZE, Image.Resampling.LANCZOS)

        avatar_dir.mkdir(parents=True, exist_ok=True)
        avatar_path = avatar_dir / f"{user['id']}.jpg"
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        avatar_path.write_bytes(buf.getvalue())

        return jsonify({"ok": True}), 200

    @bp.route("/api/user/avatar", methods=["GET"])
    def _get_avatar() -> Any:
        """Return the user's profile picture, or a default avatar."""
        from flask import Response, send_file

        from rex.routes._helpers import _require_auth

        user, _ = _require_auth()
        if user is not None:
            avatar_path = avatar_dir / f"{user['id']}.jpg"
            if avatar_path.is_file():
                return send_file(str(avatar_path), mimetype="image/jpeg")

        return Response(_DEFAULT_AVATAR_SVG, mimetype="image/svg+xml")

    return bp
