"""Custom exception hierarchy for the Rex assistant.

The assistant touches a number of subsystems (wake-word detection,
configuration loading, external APIs, audio IO).  Having a shared set of
exception types makes it easier to provide consistent error handling and
logging across the project while still giving callers the granularity they
need to react to specific failures.
"""

from __future__ import annotations


class AssistantError(Exception):
    """Base class for all custom Rex exceptions."""


class ConfigurationError(AssistantError):
    """Raised when configuration loading or validation fails."""


class WakeWordError(AssistantError):
    """Raised when wake-word models cannot be loaded or evaluated."""


class SpeechRecognitionError(AssistantError):
    """Raised when speech-to-text processing fails."""


class TextToSpeechError(AssistantError):
    """Raised when text-to-speech synthesis cannot be completed."""


class PluginError(AssistantError):
    """Raised when dynamic plugins cannot be imported or registered."""


class AudioDeviceError(AssistantError):
    """Raised when audio input/output devices are unavailable or invalid."""


class AuthenticationError(AssistantError):
    """Raised when API authentication requirements are not met."""


__all__ = [
    "AssistantError",
    "ConfigurationError",
    "WakeWordError",
    "SpeechRecognitionError",
    "TextToSpeechError",
    "PluginError",
    "AudioDeviceError",
    "AuthenticationError",
]
