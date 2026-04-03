"""Audio helpers for Rex."""

from .speaker_discovery import (
    DiscoveredSpeaker,
    SpeakerDiscoveryService,
    get_speaker_discovery,
    start_smart_speaker_discovery,
)

__all__ = [
    "DiscoveredSpeaker",
    "SpeakerDiscoveryService",
    "get_speaker_discovery",
    "start_smart_speaker_discovery",
]
