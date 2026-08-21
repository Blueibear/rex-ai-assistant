"""Authenticated mobile access to the canonical per-user output-routing policy."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify

from bridge.rex_output_routing_bridge import handle_output_routing_request
from rex.media.accounts import MediaAccountStore
from rex.media.registry import AudioTargetRegistry
from rex.mobile_api import errors as merr
from rex.mobile_api.auth import require_mobile_auth, revalidate_principal
from rex.mobile_api.authorization import ROUTE_SCOPES
from rex.mobile_api.errors import MobileApiError
from rex.mobile_api.services import MobileApiServices
from rex.mobile_api.validation import parse_json_body
from rex.output_routing.service import OutputRoutingService, _policy_to_dict


def _build_routing_backend() -> tuple[
    AudioTargetRegistry,
    OutputRoutingService,
    MediaAccountStore,
]:
    from bridge.rex_speaker_bridge import _build_speaker_runtime

    registry, _groups, _refresh = _build_speaker_runtime()
    return registry, OutputRoutingService(registry), MediaAccountStore()


def _translate_backend(body: dict[str, Any]) -> MobileApiError:
    message = str(body.get("error") or "Output-routing settings request failed.")
    denied = any(
        marker in message.casefold()
        for marker in ("unauthorized", "must belong", "user-bound", "another user")
    )
    return MobileApiError(
        merr.FORBIDDEN if denied else merr.BAD_REQUEST,
        message,
        403 if denied else 400,
    )


def _run(
    user_id: str,
    command: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry, routing, accounts = _build_routing_backend()
    body, code = handle_output_routing_request(
        {"command": command, **(payload or {})},
        registry=registry,
        routing=routing,
        media_accounts=accounts,
        bound_user_id=user_id,
    )
    if code != 0 or not body.get("ok"):
        raise _translate_backend(body)
    return body


def build_settings_blueprint(services: MobileApiServices) -> Blueprint:
    bp = Blueprint("mobile_settings", __name__)

    @bp.get("/mobile/settings/output-routing")
    @require_mobile_auth(required_scope=ROUTE_SCOPES["settings.read"])
    def get_output_routing():
        principal = g.mobile_principal
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["settings.read"],
        )
        body = _run(principal.user_id, "get_policy")
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["settings.read"],
        )
        return jsonify(body), 200

    @bp.put("/mobile/settings/output-routing")
    @require_mobile_auth(required_scope=ROUTE_SCOPES["settings.write"])
    def update_output_routing():
        principal = g.mobile_principal
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["settings.write"],
        )
        requested = parse_json_body()
        if "user_id" in requested and requested["user_id"] != principal.user_id:
            raise MobileApiError(
                merr.FORBIDDEN,
                "Routing settings cannot be changed for another user.",
                403,
            )
        requested.pop("user_id", None)

        registry, routing, accounts = _build_routing_backend()
        current = _policy_to_dict(routing.get_policy(principal.user_id))
        merged = {**current, **requested}
        if isinstance(requested.get("quiet_hours"), dict):
            merged["quiet_hours"] = {
                **current.get("quiet_hours", {}),
                **requested["quiet_hours"],
            }
        body, code = handle_output_routing_request(
            {"command": "update_policy", "policy": merged},
            registry=registry,
            routing=routing,
            media_accounts=accounts,
            bound_user_id=principal.user_id,
        )
        if code != 0 or not body.get("ok"):
            raise _translate_backend(body)
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["settings.write"],
        )
        return jsonify(body), 200

    @bp.get("/mobile/settings/output-routing/targets")
    @require_mobile_auth(required_scope=ROUTE_SCOPES["settings.read"])
    def output_targets():
        principal = g.mobile_principal
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["settings.read"],
        )
        return jsonify(_run(principal.user_id, "list_targets")), 200

    @bp.get("/mobile/settings/output-routing/accounts")
    @require_mobile_auth(required_scope=ROUTE_SCOPES["settings.read"])
    def media_accounts():
        principal = g.mobile_principal
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["settings.read"],
        )
        return jsonify(_run(principal.user_id, "list_media_accounts")), 200

    @bp.post("/mobile/settings/output-routing/test")
    @require_mobile_auth(required_scope=ROUTE_SCOPES["settings.write"])
    def test_output_target():
        principal = g.mobile_principal
        revalidate_principal(
            services,
            principal,
            required_scope=ROUTE_SCOPES["settings.write"],
        )
        payload = parse_json_body()
        target_id = payload.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise MobileApiError(merr.BAD_REQUEST, "target_id is required.", 400)
        return jsonify(_run(principal.user_id, "test_playback", {"target_id": target_id})), 200

    return bp


__all__ = ["build_settings_blueprint"]
