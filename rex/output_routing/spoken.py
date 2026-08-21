"""Spoken-response delivery through canonical output-routing policy."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .delivery import DeliveryResult
from .execution import resolve_spoken_response
from .service import OutputRoutingService


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def deliver_spoken_response(
    text: str,
    *,
    routing: OutputRoutingService,
    user_id: str,
    origin_device_id: str | None,
    at: datetime,
    remote_sender: Callable[[str, str], Any],
    local_speak: Callable[[str], Any],
) -> DeliveryResult:
    """Deliver one spoken response without silently duplicating audio."""
    route = resolve_spoken_response(
        routing,
        user_id=user_id,
        origin_device_id=origin_device_id,
        at=at,
    )
    if route.suppressed:
        # Interactive replies are explicitly required output. Quiet hours may
        # influence automatic speech, but must not make a direct user request
        # appear to hang. Preserve local reply rather than silently suppressing.
        await _await_if_needed(local_speak(text))
        return DeliveryResult(True, None, "local_required", None)
    if route.target_id is None:
        await _await_if_needed(local_speak(text))
        return DeliveryResult(True, None, "local_default", None)

    try:
        delivered = bool(await _await_if_needed(remote_sender(route.target_id, text)))
    except Exception:
        delivered = False
    if delivered:
        return DeliveryResult(True, route.target_id, route.reason, route.target_volume)
    return DeliveryResult(False, route.target_id, "delivery_failed", route.target_volume)


async def send_remote_spoken_text(target_id: str, text: str) -> bool:
    """Use supported text-addressable remote speech transports."""
    if not target_id.startswith("ha:"):
        return False
    from rex.ha_tts.client import build_ha_tts_client

    client = build_ha_tts_client()
    if client is None:
        return False
    entity_id = target_id.split(":", 1)[1]
    result = await _await_if_needed(client.speak(text, entity_id=entity_id))
    return bool(getattr(result, "ok", False))


__all__ = ["deliver_spoken_response", "send_remote_spoken_text"]