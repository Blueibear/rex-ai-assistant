"""Shared exception types for the Rex assistant stack."""

from __future__ import annotations


class AssistantError(Exception):
    """Base class for all custom Rex exceptions."""


class ConfigurationError(AssistantError):
    """Raised when application configuration is invalid or incomplete."""


class AudioDeviceError(AssistantError):
    """Raised when microphone or speaker hardware is unavailable."""


class WakeWordError(AssistantError):
    """Raised when wake-word detection fails."""


class SpeechToTextError(AssistantError):
    """Raised when speech-to-text transcription fails."""


# Alias for backward compatibility
class SpeechRecognitionError(SpeechToTextError):
    """Alias for SpeechToTextError - kept for backward compatibility."""


class TextToSpeechError(AssistantError):
    """Raised when text-to-speech synthesis fails."""


class PluginError(AssistantError):
    """Raised when a plugin fails to load or execute."""


# Alias for backward compatibility
PluginExecutionError = PluginError


class AuthenticationError(AssistantError):
    """Raised when API authentication fails."""


__all__ = [
    "AssistantError",
    "ConfigurationError",
    "AudioDeviceError",
    "WakeWordError",
    "SpeechToTextError",
    "SpeechRecognitionError",
    "TextToSpeechError",
    "PluginError",
    "PluginExecutionError",
    "AuthenticationError",
]
