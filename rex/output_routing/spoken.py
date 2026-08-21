"""Spoken-response delivery through canonical output-routing policy."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from .delivery import DeliveryResult
from .execution import resolve_spoken_response
from .service import OutputRoutingService


class SpokenVolumeController(Protocol):
    """Minimal verified-volume contract for spoken-output transports."""

    def read_volume(self, target_id: str) -> float | None: ...

    def set_and_verify_volume(self, target_id: str, volume: int | float) -> bool: ...


class HomeAssistantSpokenVolumeController:
    """Verified temporary volume control for Home Assistant media players."""

    def __init__(self, bridge_factory: Callable[[], Any] | None = None) -> None:
        self._bridge_factory = bridge_factory
        self._bridge_instance: Any | None = None

    def _bridge(self) -> Any:
        if self._bridge_instance is None:
            factory = self._bridge_factory
            if factory is None:
                from rex.ha_bridge import HABridge

                factory = HABridge
            self._bridge_instance = factory()
        return self._bridge_instance

    @staticmethod
    def _entity_id(target_id: str) -> str | None:
        if not target_id.startswith("ha:media_player."):
            return None
        return target_id.split(":", 1)[1]

    def read_volume(self, target_id: str) -> float | None:
        entity_id = self._entity_id(target_id)
        if entity_id is None:
            return None
        try:
            state = self._bridge().get_entity_state(entity_id)
        except Exception:
            return None
        if not isinstance(state, dict):
            return None
        attributes = state.get("attributes")
        if not isinstance(attributes, dict):
            return None
        raw = attributes.get("volume_level")
        if raw is None:
            raw = attributes.get("volume")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        if not math.isfinite(value) or not 0 <= value <= 1:
            return None
        return value * 100

    def set_and_verify_volume(self, target_id: str, volume: int | float) -> bool:
        entity_id = self._entity_id(target_id)
        if entity_id is None:
            return False
        if isinstance(volume, bool) or not isinstance(volume, (int, float)):
            return False
        value = float(volume)
        if not math.isfinite(value) or not 0 <= value <= 100:
            return False
        try:
            accepted, _detail = self._bridge().execute_media_service(
                entity_id,
                "volume_set",
                volume_level=value / 100,
            )
        except Exception:
            return False
        if not accepted:
            return False
        observed = self.read_volume(target_id)
        return observed is not None and abs(observed - value) <= 0.5


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
    volume_controller: SpokenVolumeController | None = None,
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
    if route.target_id.startswith("local:"):
        await _await_if_needed(local_speak(text))
        return DeliveryResult(True, route.target_id, route.reason, route.target_volume)

    controller = volume_controller
    if controller is None and route.target_volume is not None and route.target_id.startswith("ha:"):
        controller = HomeAssistantSpokenVolumeController()

    original_volume: float | None = None
    volume_changed = False
    if route.target_volume is not None and controller is not None:
        try:
            original_volume = controller.read_volume(route.target_id)
        except Exception:
            original_volume = None
        if original_volume is not None and abs(original_volume - route.target_volume) > 0.5:
            try:
                volume_changed = controller.set_and_verify_volume(
                    route.target_id,
                    route.target_volume,
                )
            except Exception:
                volume_changed = False
            if not volume_changed:
                try:
                    controller.set_and_verify_volume(route.target_id, original_volume)
                except Exception:
                    pass
                return DeliveryResult(
                    False,
                    route.target_id,
                    "temporary_volume_unverified",
                    route.target_volume,
                )

    try:
        delivered = bool(await _await_if_needed(remote_sender(route.target_id, text)))
    except Exception:
        delivered = False

    restoration_verified = True
    if volume_changed and original_volume is not None and controller is not None:
        try:
            restoration_verified = controller.set_and_verify_volume(
                route.target_id,
                original_volume,
            )
        except Exception:
            restoration_verified = False

    if delivered:
        reason = route.reason if restoration_verified else "delivered_volume_restore_unverified"
        return DeliveryResult(True, route.target_id, reason, route.target_volume)
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


__all__ = [
    "HomeAssistantSpokenVolumeController",
    "SpokenVolumeController",
    "deliver_spoken_response",
    "send_remote_spoken_text",
]
