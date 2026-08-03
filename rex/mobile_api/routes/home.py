"""Home Assistant mobile command route with S8 one-time strong authentication."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify

from rex.mobile_api import errors as merr
from rex.mobile_api.auth import require_mobile_auth, revalidate_principal
from rex.mobile_api.authorization import ROUTE_SCOPES
from rex.mobile_api.errors import MobileApiError
from rex.mobile_api.services import MobileApiServices
from rex.mobile_api.strong_auth import StrongAuthError
from rex.mobile_api.validation import parse_json_body, require_string_field
from rex.openclaw.tools.ha_tool import ha_call_service

_ACTION_NAME = "home_assistant_call_service"
_REQUIRED_FIELDS = {"domain", "service", "entity_id", "strong_auth_approval_id"}
_OPTIONAL_FIELDS = {"data"}
_MAX_ACTION_FIELD = 128


def _translate_strong_auth_error(exc: StrongAuthError) -> MobileApiError:
    code = (
        merr.STRONG_AUTH_REQUIRED
        if exc.reason in {"approval_required", "approval_expired", "approval_replayed"}
        else merr.STRONG_AUTH_INVALID
    )
    return MobileApiError(code, str(exc), 403)


def _parse_action(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    unknown = set(payload) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    missing = _REQUIRED_FIELDS - set(payload)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise MobileApiError(merr.BAD_REQUEST, f"Unsupported field(s): {names}.", 400)
    if missing:
        names = ", ".join(sorted(missing))
        raise MobileApiError(merr.BAD_REQUEST, f"Missing field(s): {names}.", 400)
    domain = require_string_field(payload, "domain", max_length=_MAX_ACTION_FIELD).strip().lower()
    service = require_string_field(payload, "service", max_length=_MAX_ACTION_FIELD).strip().lower()
    entity_id = require_string_field(
        payload,
        "entity_id",
        max_length=_MAX_ACTION_FIELD,
    ).strip()
    approval_id = require_string_field(
        payload,
        "strong_auth_approval_id",
        max_length=128,
    ).strip()
    data = payload.get("data", {})
    if not isinstance(data, dict):
        raise MobileApiError(merr.BAD_REQUEST, "Field 'data' must be an object.", 400)
    action = {
        "domain": domain,
        "service": service,
        "entity_id": entity_id,
        "data": data,
    }
    return action, approval_id


def _execute_verified_home_action(
    *,
    action: dict[str, Any],
    user_id: str,
    request_id: str,
) -> dict[str, Any]:
    context = {"user_id": user_id, "request_id": request_id}
    result = ha_call_service(
        action["domain"],
        action["service"],
        action["entity_id"],
        data=action["data"],
        context=context,
    )
    if result.get("status") == "confirmation_required":
        token = result.get("confirmation_token")
        if not isinstance(token, str) or not token:
            raise MobileApiError(
                merr.INTERNAL_ERROR,
                "The verified action could not be confirmed safely.",
                500,
            )
        result = ha_call_service(
            action["domain"],
            action["service"],
            action["entity_id"],
            data=action["data"],
            context={**context, "confirmation_token": token},
        )
    sanitized = dict(result)
    sanitized.pop("confirmation_token", None)
    return sanitized


def _http_status_for_result(result: dict[str, Any]) -> int:
    status = result.get("status")
    if status == "verified":
        return 200
    if status == "attempted_unverified":
        return 202
    if status == "denied":
        return 403
    if status == "failed":
        return 503
    return 500


def build_home_blueprint(services: MobileApiServices) -> Blueprint:
    bp = Blueprint("mobile_home", __name__)

    @bp.post("/mobile/home/command")
    @require_mobile_auth(required_scope=ROUTE_SCOPES["home.control"])
    def command():
        action, approval_id = _parse_action(parse_json_body())
        principal = g.mobile_principal
        try:
            approval = services.strong_auth_authority.consume_approval(
                principal,
                approval_id=approval_id,
                action_name=_ACTION_NAME,
                payload=action,
            )
        except StrongAuthError as exc:
            raise _translate_strong_auth_error(exc) from exc

        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["home.control"],
        )
        result = _execute_verified_home_action(
            action=action,
            user_id=principal.user_id,
            request_id=str(g.request_id),
        )
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["home.control"],
        )
        body = {
            "action_name": _ACTION_NAME,
            "action_hash": approval.action_hash,
            "risk_level": approval.risk_level,
            "approval_consumed": True,
            "result": result,
        }
        return jsonify(body), _http_status_for_result(result)

    return bp


__all__ = ["build_home_blueprint"]
