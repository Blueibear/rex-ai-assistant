"""Canonical media types shared by target providers and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    MUTE = "mute"


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
    MUTE = "mute"
    UNMUTE = "unmute"


class MediaState(StrEnum):
    """Canonical playback state within an independently read snapshot."""

    UNKNOWN = "unknown"
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    BUFFERING = "buffering"
    STOPPED = "stopped"
    UNAVAILABLE = "unavailable"


class MediaMutationOutcome(StrEnum):
    """Truthful canonical outcome assigned by the orchestration lifecycle."""

    VERIFIED = "verified"
    ATTEMPTED_UNVERIFIED = "attempted_unverified"
    FAILED = "failed"


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
class MediaActionAcknowledgement:
    """Provider acknowledgement of a command, never proof of its postcondition."""

    accepted: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class MediaStateSnapshot:
    """State obtained independently from a provider after or between commands."""

    target_id: str
    playback: MediaState
    observed_at: datetime
    volume_percent: float | None = None
    muted: bool | None = None
    position_seconds: float | None = None
    current_item_id: str | None = None
    current_item_title: str | None = None


@dataclass(frozen=True, slots=True)
class MediaMutationResult:
    """Provider-neutral mutation outcome produced by a verification lifecycle."""

    target_id: str
    action: MediaAction
    outcome: MediaMutationOutcome
    acknowledgement: MediaActionAcknowledgement
    requested_value: str | int | float | None = None
    observed_state: MediaStateSnapshot | None = None
    verification_evidence: tuple[str, ...] = ()


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
    ) -> MediaActionAcknowledgement:
        """Attempt an action and return acknowledgement, not verification."""
        ...

    def get_state(self, target: AudioTarget) -> MediaStateSnapshot:
        """Independently read target state without reusing a command response."""
        ...


__all__ = [
    "AudioTarget",
    "MediaAction",
    "MediaActionAcknowledgement",
    "MediaCapability",
    "MediaMutationOutcome",
    "MediaMutationResult",
    "MediaState",
    "MediaStateSnapshot",
    "TargetKind",
    "TargetProviderAdapter",
    "TargetResolution",
]
