"""Mobile authentication routes (issue #323, Session 1).

Implements exactly:

- ``POST /mobile/auth/login``
- ``POST /mobile/auth/refresh``
- ``POST /mobile/auth/logout``
- ``POST /mobile/auth/logout-all``
- ``GET  /mobile/auth/session``

Errors never reveal whether a username, session, or account exists, and raw
credentials/tokens are never logged.
"""

from __future__ import annotations

import hashlib
from typing import Any

from flask import Blueprint, g, jsonify

from rex.identity import validate_user_id
from rex.mobile_api import errors as merr
from rex.mobile_api import sessions as msessions
from rex.mobile_api import users as musers
from rex.mobile_api.auth import issue_access_token, require_mobile_auth
from rex.mobile_api.db import connect
from rex.mobile_api.errors import MobileApiError
from rex.mobile_api.services import MobileApiServices
from rex.mobile_api.sessions import DeviceSessionError
from rex.mobile_api.validation import (
    parse_device_info,
    parse_json_body,
    parse_refresh_token_field,
    require_string_field,
)

_LOGIN_FAILED = "Invalid username or password."


def _login_rate_key() -> str:
    """Rate-limit key for login: hashed remote address + hashed username.

    Combining both reduces per-account brute force without revealing account
    existence. Neither the raw IP address nor the raw username ever appears
    in limiter storage or logs — both components are one-way hash digests.
    """
    from flask import request  # noqa: PLC0415
    from flask_limiter.util import get_remote_address  # noqa: PLC0415

    username = ""
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        raw = payload.get("username")
        if isinstance(raw, str):
            username = raw.strip().lower()
    username_digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]
    address = get_remote_address() or "unknown"
    address_digest = hashlib.sha256(address.encode("utf-8")).hexdigest()[:16]
    return f"{address_digest}|{username_digest}"


def build_auth_blueprint(services: MobileApiServices, limiter: Any) -> Blueprint:
    """Build the ``/mobile/auth`` blueprint with route-specific rate limits."""
    bp = Blueprint("mobile_auth", __name__, url_prefix="/mobile/auth")
    cfg = services.config

    def _token_pair_response(
        session_id: str,
        user_id: str,
        username: str,
        refresh_token: str,
        refresh_expires_at: Any,
    ) -> Any:
        now = services.clock()
        access_token = issue_access_token(
            secret=services.jwt_secret,
            user_id=user_id,
            session_id=session_id,
            ttl_seconds=cfg.access_token_ttl_seconds,
            now=now,
            token_id=services.id_generator(),
        )
        refresh_expires_in = max(0, int((refresh_expires_at - now).total_seconds()))
        return jsonify(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": cfg.access_token_ttl_seconds,
                "refresh_expires_in": refresh_expires_in,
                "session_id": session_id,
                "user": musers.build_user_projection(services.db_path, user_id, username),
            }
        )

    @bp.post("/login")
    @limiter.limit(cfg.rate_limit_login, key_func=_login_rate_key)
    def login() -> Any:
        payload = parse_json_body()
        username = require_string_field(payload, "username")
        password = require_string_field(payload, "password")
        device = parse_device_info(payload)

        user = musers.verify_user_credentials(services.db_path, username, password)
        if user is None:
            raise MobileApiError(merr.AUTH_INVALID_CREDENTIALS, _LOGIN_FAILED, 401)

        created = services.session_store.create_session(user["id"], device)
        return _token_pair_response(
            created.session_id,
            created.user_id,
            str(user["username"]),
            created.refresh_token,
            created.refresh_expires_at,
        )

    @bp.post("/device-challenge")
    @limiter.limit(cfg.rate_limit_refresh)
    @require_mobile_auth
    def device_challenge() -> Any:
        principal = g.mobile_principal
        payload = parse_json_body()
        if set(payload) != {"device_id", "grant_id"}:
            raise MobileApiError(merr.BAD_REQUEST, "Device challenge fields are invalid.", 400)
        device_id = require_string_field(payload, "device_id", max_length=128)
        grant_id = require_string_field(payload, "grant_id", max_length=128)
        try:
            challenge = services.session_store.create_device_session_challenge(
                bootstrap_session_id=principal.session_id,
                user_id=principal.user_id,
                device_id=device_id,
                grant_id=grant_id,
            )
        except DeviceSessionError as exc:
            raise MobileApiError(merr.PAIRING_INVALID, str(exc), 403) from exc
        return jsonify(
            {
                "challenge_id": challenge.challenge_id,
                "bootstrap_session_id": challenge.bootstrap_session_id,
                "user_id": challenge.user_id,
                "device_id": challenge.device_id,
                "grant_id": challenge.grant_id,
                "grant_version": challenge.grant_version,
                "desktop_id": challenge.desktop_id,
                "nonce": challenge.nonce_b64,
                "expires_at": challenge.expires_at.isoformat(),
            }
        )

    @bp.post("/activate-device")
    @limiter.limit(cfg.rate_limit_refresh)
    @require_mobile_auth
    def activate_device() -> Any:
        principal = g.mobile_principal
        payload = parse_json_body()
        if set(payload) != {"challenge_id", "signature"}:
            raise MobileApiError(merr.BAD_REQUEST, "Device activation fields are invalid.", 400)
        challenge_id = require_string_field(payload, "challenge_id", max_length=128)
        signature = require_string_field(payload, "signature", max_length=256)
        try:
            created = services.session_store.activate_device_session(
                bootstrap_session_id=principal.session_id,
                user_id=principal.user_id,
                challenge_id=challenge_id,
                signature_b64=signature,
                transport_binding=services.transport_binding,
            )
        except DeviceSessionError as exc:
            raise MobileApiError(merr.PAIRING_INVALID, str(exc), 403) from exc
        return _token_pair_response(
            created.session_id,
            created.user_id,
            principal.username,
            created.refresh_token,
            created.refresh_expires_at,
        )

    @bp.post("/refresh")
    @limiter.limit(cfg.rate_limit_refresh)
    def refresh() -> Any:
        payload = parse_json_body()
        raw_token = parse_refresh_token_field(payload)

        result = services.session_store.rotate_refresh_token(raw_token)
        if result.status == msessions.REUSED:
            raise MobileApiError(
                merr.AUTH_REFRESH_REUSED,
                "Refresh token reuse detected; session revoked.",
                401,
            )
        if result.status == msessions.EXPIRED:
            raise MobileApiError(merr.AUTH_TOKEN_EXPIRED, "Refresh token expired.", 401)
        if result.status in (msessions.SESSION_REVOKED, msessions.USER_INACTIVE):
            raise MobileApiError(merr.AUTH_SESSION_REVOKED, "Session is no longer valid.", 401)
        if result.status != msessions.ROTATED:
            raise MobileApiError(merr.AUTH_TOKEN_INVALID, "Invalid refresh token.", 401)

        assert result.user_id is not None  # narrowed by ROTATED status
        assert result.session_id is not None
        assert result.refresh_token is not None
        try:
            user_id = validate_user_id(result.user_id)
        except ValueError as exc:
            raise MobileApiError(merr.AUTH_TOKEN_INVALID, "Invalid refresh token.", 401) from exc
        conn = connect(services.db_path)
        try:
            user = musers.get_user(conn, user_id)
        finally:
            conn.close()
        if not musers.is_user_active(user):
            raise MobileApiError(merr.AUTH_SESSION_REVOKED, "Session is no longer valid.", 401)
        assert user is not None  # narrowed by is_user_active
        return _token_pair_response(
            result.session_id,
            user_id,
            str(user["username"]),
            result.refresh_token,
            result.refresh_expires_at,
        )

    @bp.post("/logout")
    @require_mobile_auth(allow_revoked_session=True)
    def logout() -> Any:
        # Idempotent: a repeated logout with the same (fully validated) token
        # finds the session already revoked and returns the same success.
        principal = g.mobile_principal
        services.session_store.revoke_session(principal.session_id, "logout")
        return jsonify({"ok": True})

    @bp.post("/logout-all")
    @require_mobile_auth
    def logout_all() -> Any:
        principal = g.mobile_principal
        count = services.session_store.revoke_all_sessions_for_user(principal.user_id, "logout_all")
        return jsonify({"ok": True, "revoked_sessions": count})

    @bp.get("/session")
    @require_mobile_auth
    def current_session() -> Any:
        principal = g.mobile_principal
        return jsonify(
            {
                "session_id": principal.session_id,
                "paired": principal.paired,
                "device_id": principal.paired_device_id,
                "grant_id": principal.grant_id,
                "grant_version": principal.grant_version,
                "desktop_id": principal.desktop_id,
                "scopes": sorted(principal.scopes),
                "strong_auth_at": principal.strong_auth_at,
                "user": musers.build_user_projection(
                    services.db_path, principal.user_id, principal.username
                ),
            }
        )

    return bp


__all__ = ["build_auth_blueprint"]
