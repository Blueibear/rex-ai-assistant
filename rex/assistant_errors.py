"""Shared exception types for the Rex assistant stack.

This module has NO imports to prevent circular dependencies.
"""

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


class PluginExecutionError(AssistantError):
    """Raised when a plugin misbehaves during processing."""


class AuthenticationError(AssistantError):
    """Raised when API authentication fails."""


# Legacy alias
SpeechRecognitionError = SpeechToTextError


__all__ = [
    "AssistantError",
    "ConfigurationError",
    "AudioDeviceError",
    "WakeWordError",
    "SpeechToTextError",
    "SpeechRecognitionError",
    "TextToSpeechError",
    "PluginExecutionError",
    "AuthenticationError",
]
