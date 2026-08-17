"""Canonical media orchestration contracts."""

from .adapters import HomeAssistantMediaAdapter, MusicAssistantAdapter, SmartSpeakerAdapter
from .groups import SpeakerGroup, SpeakerGroupStore
from .models import (
    AudioTarget,
    MediaAction,
    MediaActionAcknowledgement,
    MediaCapability,
    MediaMutationOutcome,
    MediaMutationResult,
    MediaState,
    MediaStateSnapshot,
    TargetKind,
    TargetProviderAdapter,
    TargetResolution,
)
from .registry import AudioTargetRegistry

__all__ = [
    "AudioTarget",
    "AudioTargetRegistry",
    "HomeAssistantMediaAdapter",
    "MediaAction",
    "MediaActionAcknowledgement",
    "MediaCapability",
    "MediaMutationOutcome",
    "MediaMutationResult",
    "MediaState",
    "MediaStateSnapshot",
    "MusicAssistantAdapter",
    "SmartSpeakerAdapter",
    "SpeakerGroup",
    "SpeakerGroupStore",
    "TargetKind",
    "TargetProviderAdapter",
    "TargetResolution",
]
