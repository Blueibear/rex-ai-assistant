"""Canonical ToolRegistry handlers for provider-neutral media orchestration."""

from __future__ import annotations

from threading import RLock
from typing import Any

from rex.identity import validate_user_id

from .parser import MediaCommand, MediaCommandAction, parse_media_command
from .service import MediaService, MediaServiceResult

_service_lock = RLock()
_media_service: MediaService | None = None


def set_media_service(service: MediaService | None) -> None:
    """Install the process-local canonical media service used by tool handlers."""
    global _media_service
    with _service_lock:
        _media_service = service

    # Output routing consumes the same authorization-aware registry snapshot as
    # media execution. Keep this adapter here so Assistant does not duplicate a
    # second target-discovery stack merely to configure routing policy.
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
        command = MediaCommand(action=action, query=query, target_text=target_text, level=level)
    else:
        parsed = parse_media_command(transcript)
        if parsed is None:
            raise ValueError("I couldn't identify a supported media command")
        command = MediaCommand(
            action=parsed.action,
            query=query if query is not None else parsed.query,
            target_text=target_text if target_text is not None else parsed.target_text,
            level=level if level is not None else parsed.level,
        )
    return command


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
    result = _get_media_service().execute(command, user_id=owner, origin_device_id=origin_device_id)
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
    result = _get_media_service().execute(command, user_id=owner, origin_device_id=origin_device_id)
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