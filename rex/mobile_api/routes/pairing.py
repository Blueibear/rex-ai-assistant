"""Mobile-side pairing submission and status routes (S5).

Challenge creation and approval are intentionally absent from HTTP. Those
operations are available only to the local Electron desktop authority.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify

from rex.mobile_api import errors as merr
from rex.mobile_api.errors import MobileApiError
from rex.mobile_api.pairing import PairingError
from rex.mobile_api.services import MobileApiServices
from rex.mobile_api.validation import parse_json_body, require_string_field


def build_pairing_blueprint(services: MobileApiServices, limiter: Any) -> Blueprint:
    bp = Blueprint("mobile_pairing", __name__, url_prefix="/mobile/pairing")

    @bp.post("/submit")
    @limiter.limit("10 per minute")
    def submit() -> Any:
        payload = parse_json_body()
        try:
            result = services.pairing_authority.submit_proof(payload)
        except PairingError as exc:
            raise MobileApiError(merr.PAIRING_INVALID, str(exc), 400) from exc
        return (
            jsonify(
                {
                    "request_id": result.request_id,
                    "poll_token": result.poll_token,
                    "status": result.status,
                }
            ),
            202,
        )

    @bp.post("/status")
    @limiter.limit("30 per minute")
    def status() -> Any:
        payload = parse_json_body()
        request_id = require_string_field(payload, "request_id", max_length=64)
        poll_token = require_string_field(payload, "poll_token", max_length=256)
        if set(payload) != {"request_id", "poll_token"}:
            raise MobileApiError(merr.BAD_REQUEST, "Pairing status fields are invalid.", 400)
        try:
            return jsonify(services.pairing_authority.poll_status(request_id, poll_token))
        except PairingError as exc:
            raise MobileApiError(merr.PAIRING_INVALID, str(exc), 401) from exc

    return bp


__all__ = ["build_pairing_blueprint"]
