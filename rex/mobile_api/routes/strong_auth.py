"""Strong-authentication challenge and proof routes for paired mobile devices."""

from __future__ import annotations

from typing import Any, NoReturn

from flask import Blueprint, g, jsonify

from rex.mobile_api import errors as merr
from rex.mobile_api.auth import require_mobile_auth
from rex.mobile_api.errors import MobileApiError
from rex.mobile_api.services import MobileApiServices
from rex.mobile_api.strong_auth import StrongAuthError, public_challenge_payload
from rex.mobile_api.validation import parse_json_body, require_string_field

_BAD_REQUEST_REASONS = {
    "invalid_action",
    "invalid_payload",
    "strong_auth_not_applicable",
    "invalid_challenge",
}
_PERMISSION_REASONS = {
    "paired_session_required",
    "scope_denied",
}
_REQUIRED_REASONS = {
    "approval_required",
    "approval_expired",
    "approval_replayed",
    "challenge_expired",
    "challenge_replayed",
}


def _raise_api_error(exc: StrongAuthError) -> NoReturn:
    if exc.reason in _BAD_REQUEST_REASONS:
        raise MobileApiError(merr.BAD_REQUEST, str(exc), 400) from exc
    if exc.reason in _PERMISSION_REASONS:
        raise MobileApiError(merr.PERMISSION_DENIED, str(exc), 403) from exc
    if exc.reason in _REQUIRED_REASONS:
        raise MobileApiError(merr.STRONG_AUTH_REQUIRED, str(exc), 403) from exc
    raise MobileApiError(merr.STRONG_AUTH_INVALID, str(exc), 403) from exc


def _require_exact_fields(
    payload: dict[str, Any],
    *,
    required: set[str],
) -> None:
    unknown = set(payload) - required
    missing = required - set(payload)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise MobileApiError(merr.BAD_REQUEST, f"Unsupported field(s): {names}.", 400)
    if missing:
        names = ", ".join(sorted(missing))
        raise MobileApiError(merr.BAD_REQUEST, f"Missing field(s): {names}.", 400)


def build_strong_auth_blueprint(services: MobileApiServices) -> Blueprint:
    bp = Blueprint("mobile_strong_auth", __name__)

    @bp.post("/mobile/auth/strong-auth/challenge")
    @require_mobile_auth
    def create_challenge():
        payload = parse_json_body()
        _require_exact_fields(payload, required={"action_name", "action"})
        action = payload.get("action")
        if not isinstance(action, dict):
            raise MobileApiError(
                merr.BAD_REQUEST,
                "Field 'action' must be an object.",
                400,
            )
        try:
            challenge = services.strong_auth_authority.create_challenge(
                g.mobile_principal,
                action_name=payload.get("action_name"),
                payload=action,
            )
        except StrongAuthError as exc:
            _raise_api_error(exc)
        return jsonify(public_challenge_payload(challenge))

    @bp.post("/mobile/auth/strong-auth/verify")
    @require_mobile_auth
    def verify_challenge():
        payload = parse_json_body()
        _require_exact_fields(payload, required={"challenge_id", "signature"})
        challenge_id = require_string_field(payload, "challenge_id", max_length=128).strip()
        signature = require_string_field(payload, "signature", max_length=256).strip()
        try:
            approval = services.strong_auth_authority.verify_challenge(
                g.mobile_principal,
                challenge_id=challenge_id,
                signature_b64=signature,
            )
        except StrongAuthError as exc:
            _raise_api_error(exc)
        return jsonify(
            {
                "approval_id": approval.approval_id,
                "action_name": approval.action_name,
                "action_hash": approval.action_hash,
                "risk_level": approval.risk_level,
                "expires_at": approval.expires_at,
                "single_use": True,
            }
        )

    return bp


__all__ = ["build_strong_auth_blueprint"]
