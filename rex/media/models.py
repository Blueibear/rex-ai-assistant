"""Canonical media types shared by target providers and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class TargetKind(StrEnum):
    """Kind of addressable audio output."""

    SPEAKER = "speaker"
    GROUP = "group"
    LOCAL = "local"


class MediaCapability(StrEnum):
    """Operation that an audio target reports it can perform."""

    PLAY = "play"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    SEEK = "seek"
    SET_VOLUME = "set_volume"


class MediaAction(StrEnum):
    """Canonical media operation requested from a provider adapter."""

    PLAY = "play"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    SEEK = "seek"
    SET_VOLUME = "set_volume"


class MediaState(StrEnum):
    """Canonical playback state reported by a provider adapter."""

    UNKNOWN = "unknown"
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    BUFFERING = "buffering"
    STOPPED = "stopped"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class AudioTarget:
    """Immutable, provider-neutral audio output description."""

    id: str
    native_id: str
    provider: str
    kind: TargetKind
    display_name: str
    aliases: tuple[str, ...]
    room: str | None
    capabilities: frozenset[MediaCapability]
    online: bool
    health: str


@dataclass(frozen=True, slots=True)
class TargetResolution:
    """Deterministic outcome of resolving a target query."""

    target: AudioTarget | None
    reason: str
    ambiguous_ids: tuple[str, ...] = ()


@runtime_checkable
class TargetProviderAdapter(Protocol):
    """Provider boundary used by canonical media orchestration."""

    provider: str

    def discover_targets(self) -> tuple[AudioTarget, ...]:
        """Return the provider's current immutable target snapshot."""
        ...

    def execute_action(
        self,
        target: AudioTarget,
        action: MediaAction,
        *,
        value: str | int | float | None = None,
    ) -> MediaState:
        """Attempt an action and return the provider's resulting state."""
        ...

    def get_state(self, target: AudioTarget) -> MediaState:
        """Read the current canonical state for a target."""
        ...


__all__ = [
    "AudioTarget",
    "MediaAction",
    "MediaCapability",
    "MediaState",
    "TargetKind",
    "TargetProviderAdapter",
    "TargetResolution",
]
