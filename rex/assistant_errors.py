"""Shared exception types for the Rex assistant stack.

This module has NO imports to prevent circular dependencies.
"""

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


class AudioFormatError(SpeechToTextError):
    """Raised when STT input is not a valid WAV payload."""


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


class IntegrationNotConfiguredError(AssistantError):
    """Raised when an optional integration is requested but not configured."""


class IdentityRequiredError(AssistantError):
    """Raised when a private assistant operation runs without a validated user identity.

    A missing identity must never silently become another profile (issue
    #303): callers either bind an explicit ``user_id`` at construction time
    or supply a validated ``active_user_id`` per request.  The message must
    stay deterministic and free of private details (paths, credentials,
    other user IDs).
    """


__all__ = [
    "AssistantError",
    "ConfigurationError",
    "AudioDeviceError",
    "WakeWordError",
    "SpeechToTextError",
    "AudioFormatError",
    "SpeechRecognitionError",
    "TextToSpeechError",
    "PluginError",
    "PluginExecutionError",
    "AuthenticationError",
    "IntegrationNotConfiguredError",
    "IdentityRequiredError",
]
