"""Shared exception types for the Rex assistant stack."""

from __future__ import annotations


class AssistantError(Exception):
    """Base class for recoverable assistant errors."""


class ConfigurationError(AssistantError):
    """Raised when application configuration is invalid or incomplete."""


class AudioDeviceError(AssistantError):
    """Raised when microphone or speaker hardware is unavailable."""


class WakeWordError(AssistantError):
    """Raised when wake-word detection fails."""


class SpeechToTextError(AssistantError):
    """Raised when speech-to-text transcription fails."""


class TextToSpeechError(AssistantError):
    """Raised when text-to-speech synthesis fails."""


class PluginError(AssistantError):
    """Raised when a plugin fails to load or execute."""


class PluginExecutionError(PluginError):
    """Backward-compatible alias for plugin execution failures."""


class PluginError(AssistantError):
    """Raised when dynamic plugins cannot be imported or registered."""


class AuthenticationError(AssistantError):
    """Raised when API authentication fails."""


__all__ = [
    "AssistantError",
    "ConfigurationError",
    "AudioDeviceError",
    "WakeWordError",
    "SpeechToTextError",
    "TextToSpeechError",
    "PluginError",
    "PluginExecutionError",
    "PluginError",
    "AuthenticationError",
]
