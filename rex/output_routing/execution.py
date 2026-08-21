"""Execution-facing helpers that apply output policy without granting authority."""

from __future__ import annotations

from datetime import datetime

from rex.media.parser import MediaCommand

from .models import OutputKind, ResolvedRoute
from .service import OutputRoutingService


def resolve_spoken_response(
    routing: OutputRoutingService,
    *,
    user_id: str,
    origin_device_id: str | None,
    at: datetime,
) -> ResolvedRoute:
    """Prefer the authorized endpoint that heard the voice request."""
    registry = getattr(routing, "_registry", None)
    if registry is not None and origin_device_id is not None:
        origin = registry.resolve(None, user_id=user_id, origin_device_id=origin_device_id)
        if origin.target is not None:
            policy = routing.get_policy(user_id)
            return ResolvedRoute(
                output_kind=OutputKind.SPOKEN_RESPONSE,
                target_id=origin.target.id,
                reason="request_origin",
                target_volume=policy.volume_for(OutputKind.SPOKEN_RESPONSE),
            )
    return routing.resolve(
        user_id=user_id,
        output_kind=OutputKind.SPOKEN_RESPONSE,
        explicit_target_text=None,
        origin_device_id=origin_device_id,
        at=at,
    )


def resolve_media_command(
    routing: OutputRoutingService,
    command: MediaCommand,
    *,
    user_id: str,
    origin_device_id: str | None,
    at: datetime,
) -> MediaCommand:
    """Fill only an omitted media target from canonical output policy."""
    if command.target_text is not None:
        return command
    route = routing.resolve(
        user_id=user_id,
        output_kind=OutputKind.MEDIA,
        explicit_target_text=None,
        origin_device_id=origin_device_id,
        at=at,
    )
    if route.target_id is None:
        return command
    return MediaCommand(
        action=command.action,
        query=command.query,
        target_text=route.target_id,
        level=command.level,
    )


__all__ = ["resolve_media_command", "resolve_spoken_response"]