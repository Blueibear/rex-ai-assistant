"""Authenticated mobile access to the canonical per-user output-routing policy."""

from __future__ import annotations

from functools import wraps
from typing import Any

from flask import Blueprint, g, jsonify

from bridge.rex_output_routing_bridge import handle_output_routing_request
from rex.context.privacy import ContextPrivacyService, get_context_privacy_service
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


def _build_context_privacy_service() -> ContextPrivacyService:
    return get_context_privacy_service()


def _privacy_owner(principal_user_id: str, payload: dict[str, Any]) -> str:
    requested = payload.get("user_id")
    if requested is not None and requested != principal_user_id:
        raise MobileApiError(
            merr.FORBIDDEN,
            "Privacy settings cannot be changed for another user.",
            403,
        )
    return principal_user_id


def _translate_privacy_error(exc: Exception) -> MobileApiError:
    if isinstance(exc, PermissionError):
        return MobileApiError(
            merr.FORBIDDEN,
            "Privacy settings are user-bound.",
            403,
        )
    return MobileApiError(merr.BAD_REQUEST, str(exc), 400)


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


def _privacy_principal(services: MobileApiServices, required_scope: str):
    principal = g.mobile_principal
    revalidate_principal(services, principal, required_scope=required_scope)
    return principal


@require_mobile_auth(required_scope=ROUTE_SCOPES["settings.read"])
def _get_context_privacy_view(services: MobileApiServices):
    principal = _privacy_principal(services, ROUTE_SCOPES["settings.read"])
    try:
        state = _build_context_privacy_service().get_state(principal.user_id)
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        raise _translate_privacy_error(exc) from exc
    return jsonify({"ok": True, **state}), 200


@require_mobile_auth(required_scope=ROUTE_SCOPES["settings.write"])
def _update_location_assist_view(services: MobileApiServices):
    principal = _privacy_principal(services, ROUTE_SCOPES["settings.write"])
    payload = parse_json_body()
    owner = _privacy_owner(principal.user_id, payload)
    enabled = payload.get("location_assist")
    if not isinstance(enabled, bool):
        raise MobileApiError(merr.BAD_REQUEST, "location_assist must be boolean.", 400)
    try:
        location = _build_context_privacy_service().set_location_assist(
            owner_user_id=owner,
            enabled=enabled,
            actor_user_id=principal.user_id,
        )
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        raise _translate_privacy_error(exc) from exc
    revalidate_principal(services, principal, required_scope=ROUTE_SCOPES["settings.write"])
    return jsonify({"ok": True, "location": location}), 200


@require_mobile_auth(required_scope=ROUTE_SCOPES["settings.write"])
def _update_location_share_view(services: MobileApiServices):
    principal = _privacy_principal(services, ROUTE_SCOPES["settings.write"])
    payload = parse_json_body()
    owner = _privacy_owner(principal.user_id, payload)
    recipient = payload.get("recipient_user_id")
    enabled = payload.get("enabled")
    if not isinstance(recipient, str) or not recipient:
        raise MobileApiError(merr.BAD_REQUEST, "recipient_user_id is required.", 400)
    if not isinstance(enabled, bool):
        raise MobileApiError(merr.BAD_REQUEST, "enabled must be boolean.", 400)
    try:
        location = _build_context_privacy_service().set_location_share(
            owner_user_id=owner,
            recipient_user_id=recipient,
            enabled=enabled,
            actor_user_id=principal.user_id,
        )
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        raise _translate_privacy_error(exc) from exc
    revalidate_principal(services, principal, required_scope=ROUTE_SCOPES["settings.write"])
    return jsonify({"ok": True, "location": location}), 200


@require_mobile_auth(required_scope=ROUTE_SCOPES["settings.write"])
def _update_proactive_assistance_view(services: MobileApiServices):
    principal = _privacy_principal(services, ROUTE_SCOPES["settings.write"])
    payload = parse_json_body()
    owner = _privacy_owner(principal.user_id, payload)
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise MobileApiError(merr.BAD_REQUEST, "enabled must be boolean.", 400)
    try:
        saved = _build_context_privacy_service().set_proactive_assistance(
            owner_user_id=owner,
            enabled=enabled,
            actor_user_id=principal.user_id,
        )
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        raise _translate_privacy_error(exc) from exc
    revalidate_principal(services, principal, required_scope=ROUTE_SCOPES["settings.write"])
    return jsonify({"ok": True, "proactive_assistance": saved}), 200


@require_mobile_auth(required_scope=ROUTE_SCOPES["settings.write"])
def _update_context_source_view(services: MobileApiServices):
    principal = _privacy_principal(services, ROUTE_SCOPES["settings.write"])
    payload = parse_json_body()
    owner = _privacy_owner(principal.user_id, payload)
    source_id = payload.get("source_id")
    enabled = payload.get("enabled")
    if not isinstance(source_id, str) or not source_id:
        raise MobileApiError(merr.BAD_REQUEST, "source_id is required.", 400)
    if not isinstance(enabled, bool):
        raise MobileApiError(merr.BAD_REQUEST, "enabled must be boolean.", 400)
    try:
        source = _build_context_privacy_service().set_source_context(
            owner_user_id=owner,
            source_id=source_id,
            enabled=enabled,
            actor_user_id=principal.user_id,
        )
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        raise _translate_privacy_error(exc) from exc
    revalidate_principal(services, principal, required_scope=ROUTE_SCOPES["settings.write"])
    return jsonify({"ok": True, "source": source}), 200


@require_mobile_auth(required_scope=ROUTE_SCOPES["settings.write"])
def _update_context_upload_view(services: MobileApiServices):
    principal = _privacy_principal(services, ROUTE_SCOPES["settings.write"])
    payload = parse_json_body()
    owner = _privacy_owner(principal.user_id, payload)
    doc_id = payload.get("doc_id")
    audience = payload.get("audience_scope")
    context_enabled = payload.get("context_enabled")
    if not isinstance(doc_id, str) or not doc_id:
        raise MobileApiError(merr.BAD_REQUEST, "doc_id is required.", 400)
    if audience not in {"private", "household"}:
        raise MobileApiError(
            merr.BAD_REQUEST,
            "audience_scope must be private or household.",
            400,
        )
    if not isinstance(context_enabled, bool):
        raise MobileApiError(merr.BAD_REQUEST, "context_enabled must be boolean.", 400)
    try:
        upload = _build_context_privacy_service().update_upload_policy(
            owner_user_id=owner,
            doc_id=doc_id,
            audience_scope=audience,
            context_enabled=context_enabled,
            actor_user_id=principal.user_id,
        )
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        raise _translate_privacy_error(exc) from exc
    revalidate_principal(services, principal, required_scope=ROUTE_SCOPES["settings.write"])
    return jsonify({"ok": True, "upload": upload}), 200


def _bind_settings_view(view, services: MobileApiServices):
    @wraps(view)
    def bound_view():
        return view(services)

    return bound_view


def _register_context_privacy_routes(bp: Blueprint, services: MobileApiServices) -> None:
    routes = (
        ("/mobile/settings/context", "get_context_privacy", _get_context_privacy_view, ["GET"]),
        (
            "/mobile/settings/context/location",
            "update_location_assist",
            _update_location_assist_view,
            ["PUT"],
        ),
        (
            "/mobile/settings/context/location-share",
            "update_location_share",
            _update_location_share_view,
            ["PUT"],
        ),
        (
            "/mobile/settings/context/proactive",
            "update_proactive_assistance",
            _update_proactive_assistance_view,
            ["PUT"],
        ),
        (
            "/mobile/settings/context/source",
            "update_context_source",
            _update_context_source_view,
            ["PUT"],
        ),
        (
            "/mobile/settings/context/upload",
            "update_context_upload",
            _update_context_upload_view,
            ["PUT"],
        ),
    )
    for rule, endpoint, view, methods in routes:
        bp.add_url_rule(rule, endpoint, _bind_settings_view(view, services), methods=methods)


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

    _register_context_privacy_routes(bp, services)

    return bp


__all__ = ["build_settings_blueprint"]
