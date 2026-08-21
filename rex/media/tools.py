"""Canonical ToolRegistry handlers for provider-neutral media orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from rex.identity import validate_user_id
from rex.media.accounts import MediaAccountRef
from rex.output_routing.models import OutputKind
from rex.output_routing.runtime import get_output_routing_service, user_local_now
from rex.runtime.invocation import current_turn_invocation

from .parser import MediaCommand, MediaCommandAction, parse_media_command
from .service import MediaService, MediaServiceResult

_service_lock = RLock()
_media_service: MediaService | None = None


@dataclass(frozen=True, slots=True)
class _MediaPolicyDecision:
    command: MediaCommand
    origin_device_id: str | None
    account_ref: MediaAccountRef | None
    route_volume: int | None


def set_media_service(service: MediaService | None) -> None:
    """Install the process-local canonical media service used by tool handlers."""
    global _media_service
    with _service_lock:
        _media_service = service

    from rex.output_routing.runtime import set_output_registry_provider

    if service is None:
        set_output_registry_provider(None)
        return

    def current_registry():
        refresher = getattr(service, "_refresh_registry", None)
        if callable(refresher):
            refresher()
        registry = getattr(service, "_registry", None)
        if registry is None:
            raise RuntimeError("Canonical media registry is unavailable")
        return registry

    set_output_registry_provider(current_registry)


def _get_media_service() -> MediaService:
    with _service_lock:
        service = _media_service
    if service is None:
        raise RuntimeError("Canonical media service is not configured")
    return service


def _command_from_request(
    *,
    transcript: str,
    action: str | None,
    query: str | None,
    target_text: str | None,
    level: int | None,
) -> MediaCommand:
    if action:
        return MediaCommand(action=action, query=query, target_text=target_text, level=level)
    parsed = parse_media_command(transcript)
    if parsed is None:
        raise ValueError("I couldn't identify a supported media command")
    return MediaCommand(
        action=parsed.action,
        query=query if query is not None else parsed.query,
        target_text=target_text if target_text is not None else parsed.target_text,
        level=level if level is not None else parsed.level,
    )


def _apply_output_policy(
    command: MediaCommand,
    *,
    owner: str,
    origin_device_id: str | None,
) -> _MediaPolicyDecision:
    """Resolve one media route/account from trusted turn provenance."""
    invocation = current_turn_invocation()
    trusted_origin = invocation.device_id if invocation.device_id is not None else origin_device_id
    try:
        routing = get_output_routing_service()
    except RuntimeError:
        return _MediaPolicyDecision(command, trusted_origin, None, None)

    route = routing.resolve(
        user_id=owner,
        output_kind=OutputKind.MEDIA,
        explicit_target_text=command.target_text,
        origin_device_id=trusted_origin,
        at=user_local_now(owner),
    )
    if route.suppressed:
        raise ValueError("Media output is suppressed by the current quiet-hours policy")
    if route.requires_confirmation:
        raise ValueError("Media rerouting requires confirmation before playback continues")
    if route.target_id is None:
        raise ValueError(f"Media route is unavailable: {route.reason}")

    routed_command = MediaCommand(
        action=command.action,
        query=command.query,
        target_text=route.target_id,
        level=command.level,
    )
    account_ref = routing.resolve_media_account(
        active_user_id=owner,
        identity_resolution=invocation.identity_resolution,
        requested_account_id=None,
        operation=MediaCommandAction(command.action).value,
    )
    return _MediaPolicyDecision(
        routed_command,
        trusted_origin,
        account_ref,
        route.target_volume,
    )


def _state_payload(state: Any) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "target_id": state.target_id,
        "playback": state.playback.value,
        "volume_percent": state.volume_percent,
        "muted": state.muted,
        "position_seconds": state.position_seconds,
        "current_item_id": state.current_item_id,
        "current_item_title": state.current_item_title,
        "observed_at": state.observed_at.isoformat(),
    }


def _verification_payload(result: MediaServiceResult, user_id: str) -> dict[str, Any] | None:
    if result.requested_target_id is None or result.verification_expected is None:
        return None
    provider = result.requested_target_id.split(":", 1)[0]
    return {
        "target_id": result.requested_target_id,
        "provider": provider,
        "user_id": user_id,
        "expected": dict(result.verification_expected),
    }


def media_read(
    *,
    transcript: str = "",
    action: str | None = None,
    query: str | None = None,
    target_text: str | None = None,
    level: int | None = None,
    origin_device_id: str | None = None,
    _user_id: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    owner = validate_user_id(_user_id)
    command = _command_from_request(
        transcript=transcript,
        action=action,
        query=query,
        target_text=target_text,
        level=level,
    )
    if command.action is not MediaCommandAction.QUERY_STATE:
        raise ValueError("This media request changes playback; use media_manage")
    decision = _apply_output_policy(
        command,
        owner=owner,
        origin_device_id=origin_device_id,
    )
    result = _get_media_service().execute(
        decision.command,
        user_id=owner,
        origin_device_id=decision.origin_device_id,
        account_ref=decision.account_ref,
        route_volume=decision.route_volume,
    )
    if result.outcome != "read" or result.state is None:
        raise ValueError(result.message or f"Media read failed: {result.outcome}")
    return {"target_id": result.requested_target_id, "state": _state_payload(result.state)}


def media_manage(
    *,
    transcript: str = "",
    action: str | None = None,
    query: str | None = None,
    target_text: str | None = None,
    level: int | None = None,
    origin_device_id: str | None = None,
    _user_id: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    owner = validate_user_id(_user_id)
    command = _command_from_request(
        transcript=transcript,
        action=action,
        query=query,
        target_text=target_text,
        level=level,
    )
    if command.action is MediaCommandAction.QUERY_STATE:
        raise ValueError("This media request is read-only; use media_read")
    decision = _apply_output_policy(
        command,
        owner=owner,
        origin_device_id=origin_device_id,
    )
    result = _get_media_service().execute(
        decision.command,
        user_id=owner,
        origin_device_id=decision.origin_device_id,
        account_ref=decision.account_ref,
        route_volume=decision.route_volume,
    )
    if result.outcome not in {"verified", "attempted_unverified"}:
        return {
            "status": "failed",
            "lifecycle_state": "failed",
            "success": False,
            "error": result.message or result.outcome,
            "target_id": result.requested_target_id,
        }
    verification = _verification_payload(result, owner)
    payload: dict[str, Any] = {
        "status": result.outcome,
        "lifecycle_state": result.outcome,
        "target_id": result.requested_target_id,
        "state": _state_payload(result.state),
        "verification": verification,
    }
    if result.message:
        payload["message"] = result.message
    if result.outcome == "verified":
        payload["success"] = True
    return payload


def verify_media_mutation(args: dict[str, Any], output: Any) -> bool:
    """Independently re-read current state before allowing a verified claim."""
    if not isinstance(output, dict):
        return False
    verification = output.get("verification")
    if not isinstance(verification, dict):
        return False
    target_id = verification.get("target_id")
    provider = verification.get("provider")
    user_id = verification.get("user_id")
    expected = verification.get("expected")
    if not isinstance(target_id, str) or not target_id:
        return False
    if not isinstance(provider, str) or not provider:
        return False
    if not isinstance(user_id, str) or not user_id:
        return False
    if not isinstance(expected, dict) or not expected:
        return False
    try:
        return _get_media_service().reverify(
            target_id=target_id,
            provider=provider,
            user_id=user_id,
            expected=expected,
        )
    except Exception:
        return False


__all__ = ["media_manage", "media_read", "set_media_service", "verify_media_mutation"]
