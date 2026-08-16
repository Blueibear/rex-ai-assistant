"""Canonical adapters over Rex's existing media provider integrations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from rex.audio.speaker_discovery import DiscoveredSpeaker

from .models import (
    AudioTarget,
    MediaAction,
    MediaActionAcknowledgement,
    MediaCapability,
    MediaState,
    MediaStateSnapshot,
    TargetKind,
)

_HA_CAPABILITIES = frozenset(
    {
        MediaCapability.PLAY,
        MediaCapability.PAUSE,
        MediaCapability.NEXT,
        MediaCapability.SET_VOLUME,
    }
)
_HA_SERVICES = {
    MediaAction.PLAY: "media_play",
    MediaAction.PAUSE: "media_pause",
    MediaAction.NEXT: "media_next_track",
    MediaAction.SET_VOLUME: "volume_set",
}
_MUSIC_ASSISTANT_ACTIONS = frozenset(
    {
        MediaAction.PLAY,
        MediaAction.PAUSE,
        MediaAction.RESUME,
        MediaAction.NEXT,
        MediaAction.SET_VOLUME,
    }
)
_PLAYBACK_STATES = {
    "buffering": MediaState.BUFFERING,
    "idle": MediaState.IDLE,
    "off": MediaState.STOPPED,
    "paused": MediaState.PAUSED,
    "playing": MediaState.PLAYING,
    "stopped": MediaState.STOPPED,
    "unavailable": MediaState.UNAVAILABLE,
}


class _SpeakerDiscovery(Protocol):
    def discover_now(self) -> list[DiscoveredSpeaker]: ...


class _HomeAssistantBridge(Protocol):
    def list_entities(self) -> list[dict[str, Any]]: ...

    def get_entity_state(self, entity_id: str) -> dict[str, Any] | None: ...

    def execute_media_service(
        self,
        entity_id: str,
        service: str,
        *,
        volume_level: float | None = None,
    ) -> tuple[bool, str]: ...


class _MusicAssistantClient(Protocol):
    def play(self, query: str, room: str | None = None) -> dict: ...

    def pause(self, room: str | None = None) -> dict: ...

    def resume(self, room: str | None = None) -> dict: ...

    def skip(self, room: str | None = None) -> dict: ...

    def set_volume(self, level: int, room: str | None = None) -> dict: ...


def _as_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


class SmartSpeakerAdapter:
    """Expose current Sonos/Bose discovery without claiming media controls."""

    provider = "smart_speaker"

    def __init__(self, discovery: _SpeakerDiscovery) -> None:
        self._discovery = discovery

    def discover_targets(self) -> tuple[AudioTarget, ...]:
        targets = (
            AudioTarget(
                id=f"{speaker.provider}:{speaker.ip}",
                native_id=speaker.ip,
                provider=speaker.provider,
                kind=TargetKind.SPEAKER,
                display_name=speaker.name,
                aliases=(),
                room=None,
                capabilities=frozenset(),
                online=True,
                health="discovered",
            )
            for speaker in self._discovery.discover_now()
        )
        return tuple(sorted(targets, key=lambda target: target.id))

    def execute_action(
        self,
        target: AudioTarget,
        action: MediaAction,
        *,
        value: str | int | float | None = None,
    ) -> MediaActionAcknowledgement:
        del target, action, value
        return MediaActionAcknowledgement(
            accepted=False,
            detail="Smart-speaker media mutations are unsupported",
        )

    def get_state(self, target: AudioTarget) -> MediaStateSnapshot:
        return MediaStateSnapshot(
            target_id=target.id,
            playback=MediaState.UNKNOWN,
            observed_at=datetime.now(tz=UTC),
        )


class HomeAssistantMediaAdapter:
    """Adapt Home Assistant media-player entities to canonical contracts."""

    provider = "ha"

    def __init__(self, bridge: _HomeAssistantBridge) -> None:
        self._bridge = bridge

    def discover_targets(self) -> tuple[AudioTarget, ...]:
        targets = []
        for entity in self._bridge.list_entities():
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id.startswith("media_player."):
                continue
            attributes = entity.get("attributes")
            if not isinstance(attributes, Mapping):
                attributes = {}
            friendly_name = attributes.get("friendly_name")
            state = str(entity.get("state", "unknown")).casefold()
            targets.append(
                AudioTarget(
                    id=f"ha:{entity_id}",
                    native_id=entity_id,
                    provider=self.provider,
                    kind=TargetKind.SPEAKER,
                    display_name=(friendly_name if isinstance(friendly_name, str) else entity_id),
                    aliases=(),
                    room=None,
                    capabilities=_HA_CAPABILITIES,
                    online=state != "unavailable",
                    health=state,
                )
            )
        return tuple(sorted(targets, key=lambda target: target.id))

    def execute_action(
        self,
        target: AudioTarget,
        action: MediaAction,
        *,
        value: str | int | float | None = None,
    ) -> MediaActionAcknowledgement:
        service = _HA_SERVICES.get(action)
        if service is None:
            return MediaActionAcknowledgement(
                accepted=False,
                detail=f"Home Assistant action {action.value} is unsupported",
            )

        data: dict[str, Any] = {"entity_id": target.native_id}
        if action is MediaAction.SET_VOLUME:
            volume = _as_number(value)
            if volume is None or not 0 <= volume <= 100:
                return MediaActionAcknowledgement(
                    accepted=False,
                    detail="Home Assistant volume requires a value from 0 to 100",
                )
            data["volume_level"] = volume / 100
        elif value is not None:
            return MediaActionAcknowledgement(
                accepted=False,
                detail=f"Home Assistant action {action.value} does not accept a value",
            )

        try:
            accepted, detail = self._bridge.execute_media_service(
                target.native_id,
                service,
                volume_level=data.get("volume_level"),
            )
        except Exception as exc:
            return MediaActionAcknowledgement(accepted=False, detail=str(exc))
        return MediaActionAcknowledgement(accepted=accepted, detail=detail)

    def get_state(self, target: AudioTarget) -> MediaStateSnapshot:
        state = self._bridge.get_entity_state(target.native_id)
        if state is None:
            return MediaStateSnapshot(
                target_id=target.id,
                playback=MediaState.UNKNOWN,
                observed_at=datetime.now(tz=UTC),
            )
        attributes = state.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
        volume = _as_number(attributes.get("volume_level"))
        if volume is None:
            volume = _as_number(attributes.get("volume"))
        playback = _PLAYBACK_STATES.get(
            str(state.get("state", "unknown")).casefold(),
            MediaState.UNKNOWN,
        )
        return MediaStateSnapshot(
            target_id=target.id,
            playback=playback,
            observed_at=datetime.now(tz=UTC),
            volume_percent=volume * 100 if volume is not None else None,
            position_seconds=_as_number(attributes.get("media_position")),
            current_item_id=_as_optional_string(attributes.get("media_content_id")),
            current_item_title=_as_optional_string(attributes.get("media_title")),
        )


class MusicAssistantAdapter:
    """Wrap only operations already exposed by ``MusicAssistantClient``."""

    provider = "music_assistant"

    def __init__(self, client: _MusicAssistantClient) -> None:
        self._client = client

    def discover_targets(self) -> tuple[AudioTarget, ...]:
        return ()

    def execute_action(
        self,
        target: AudioTarget,
        action: MediaAction,
        *,
        value: str | int | float | None = None,
    ) -> MediaActionAcknowledgement:
        if action not in _MUSIC_ASSISTANT_ACTIONS:
            return MediaActionAcknowledgement(
                accepted=False,
                detail=f"Music Assistant action {action.value} is unsupported",
            )
        try:
            if action is MediaAction.PLAY:
                if not isinstance(value, str) or not value.strip():
                    return MediaActionAcknowledgement(
                        accepted=False,
                        detail="Music Assistant play requires a query",
                    )
                self._client.play(value, room=target.native_id)
            elif action is MediaAction.PAUSE:
                self._client.pause(room=target.native_id)
            elif action is MediaAction.RESUME:
                self._client.resume(room=target.native_id)
            elif action is MediaAction.NEXT:
                self._client.skip(room=target.native_id)
            elif action is MediaAction.SET_VOLUME:
                volume = _as_number(value)
                if volume is None or not 0 <= volume <= 100:
                    return MediaActionAcknowledgement(
                        accepted=False,
                        detail="Music Assistant volume requires a value from 0 to 100",
                    )
                self._client.set_volume(int(volume), room=target.native_id)
        except Exception as exc:
            return MediaActionAcknowledgement(accepted=False, detail=str(exc))
        return MediaActionAcknowledgement(
            accepted=True,
            detail="Music Assistant accepted the command",
        )

    def get_state(self, target: AudioTarget) -> MediaStateSnapshot:
        return MediaStateSnapshot(
            target_id=target.id,
            playback=MediaState.UNKNOWN,
            observed_at=datetime.now(tz=UTC),
        )


__all__ = ["HomeAssistantMediaAdapter", "MusicAssistantAdapter", "SmartSpeakerAdapter"]
