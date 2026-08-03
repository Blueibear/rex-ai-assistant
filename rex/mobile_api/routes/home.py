"""Home Assistant mobile command route with S8 one-time strong authentication."""

from __future__ import annotations

import logging
import re
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

_SUPPORTED_DOMAINS = {
    "light": "light",
    "switch": "switch",
    "input_boolean": "switch",
    "fan": "switch",
    "climate": "thermostat",
    "lock": "lock",
    "cover": "switch",
    "scene": "scene",
    "media_player": "speaker",
}
_ACTIVE_STATES = {"on", "open", "opening", "unlocked", "heating", "cooling", "playing"}
_INACTIVE_STATES = {"off", "closed", "closing", "locked", "idle", "paused", "standby"}
_SLUG_RE = re.compile(r"[^a-z0-9]+")
logger = logging.getLogger(__name__)

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


def _load_home_entities() -> tuple[bool, list[dict[str, Any]]]:
    """Load the household HA entity inventory without exposing credentials."""
    from rex.config import load_config
    from rex.ha.discovery import discover_devices, load_ignored_devices

    config = load_config(reload=True)
    configured = bool(config.ha_base_url and config.ha_token)
    if not configured:
        return False, []
    ignored = set(load_ignored_devices())
    entities = discover_devices(
        config.ha_base_url,
        config.ha_token,
        verify_ssl=config.ha_verify_ssl,
        timeout=config.ha_timeout,
    )
    return True, [entry for entry in entities if entry.get("entity_id") not in ignored]


def _device_type(domain: str, entity_id: str, name: str) -> str | None:
    mapped = _SUPPORTED_DOMAINS.get(domain)
    if mapped is None:
        return None
    if domain == "cover":
        if "garage" in f"{entity_id} {name}".lower():
            return "garage"
        return None
    return mapped


def _device_state(raw_state: object) -> str:
    state = str(raw_state or "unknown").strip().lower()
    if state in _ACTIVE_STATES:
        return "on"
    if state in _INACTIVE_STATES:
        return "off"
    return "unknown"


def _risk_level(device_type: str, entity_id: str, name: str) -> str:
    if device_type in {"lock", "garage"}:
        return "high"
    if device_type == "thermostat" or "camera" in f"{entity_id} {name}".lower():
        return "medium"
    return "low"


def _room_id(value: object) -> str:
    raw = str(value or "home").strip().lower()
    slug = _SLUG_RE.sub("-", raw).strip("-")
    return slug[:64] or "home"


def _project_home_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    room_counts: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        domain = str(entity.get("domain") or "").strip().lower()
        name = str(entity.get("friendly_name") or entity_id).strip() or entity_id
        if not entity_id or not domain or not entity_id.startswith(f"{domain}."):
            continue
        device_type = _device_type(domain, entity_id, name)
        if device_type is None:
            continue
        room_name = str(entity.get("area_name") or "Home").strip() or "Home"
        room = _room_id(room_name)
        state = _device_state(entity.get("state"))
        device = {
            "id": entity_id,
            "name": name[:160],
            "type": device_type,
            "room": room,
            "state": state,
            "riskLevel": _risk_level(device_type, entity_id, name),
        }
        alias = entity.get("alias")
        if isinstance(alias, str) and alias.strip():
            device["alias"] = alias.strip()[:160]
        devices.append(device)
        aggregate = room_counts.setdefault(
            room,
            {"id": room, "name": room_name[:80], "deviceCount": 0, "activeCount": 0},
        )
        aggregate["deviceCount"] += 1
        if state == "on":
            aggregate["activeCount"] += 1
    devices.sort(key=lambda item: (item["room"], item["name"].casefold(), item["id"]))
    rooms = sorted(room_counts.values(), key=lambda item: (item["name"].casefold(), item["id"]))
    rooms.insert(
        0,
        {
            "id": "all",
            "name": "All",
            "deviceCount": len(devices),
            "activeCount": sum(1 for item in devices if item["state"] == "on"),
        },
    )
    return {"rooms": rooms, "devices": devices}


def build_home_blueprint(services: MobileApiServices) -> Blueprint:
    bp = Blueprint("mobile_home", __name__)

    @bp.get("/mobile/home/entities")
    @require_mobile_auth(required_scope=ROUTE_SCOPES["home.read"])
    def entities():
        principal = g.mobile_principal
        revalidate_principal(services, principal, required_scope=ROUTE_SCOPES["home.read"])
        try:
            configured, raw_entities = _load_home_entities()
        except Exception as exc:
            logger.warning(
                "Mobile Home Assistant inventory fetch failed (%s)",
                type(exc).__name__,
            )
            raise MobileApiError(
                merr.BACKEND_UNAVAILABLE,
                "Home Assistant is unavailable.",
                503,
                retryable=True,
            ) from exc
        revalidate_principal(services, principal, required_scope=ROUTE_SCOPES["home.read"])
        projection = _project_home_entities(raw_entities)
        return jsonify({"configured": configured, **projection}), 200

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
