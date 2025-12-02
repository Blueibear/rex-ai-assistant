"""Backward compatibility wrapper - imports from rex.assistant_errors.

New code should import directly from rex.assistant_errors.
"""

from __future__ import annotations

# Re-export all exception types from the rex package
from rex.assistant_errors import (
    AssistantError,
    AudioDeviceError,
    AuthenticationError,
    ConfigurationError,
    PluginError,
    PluginExecutionError,
    SpeechRecognitionError,
    SpeechToTextError,
    TextToSpeechError,
    WakeWordError,
)

__all__ = [
    "AssistantError",
    "ConfigurationError",
    "WakeWordError",
    "SpeechRecognitionError",
    "SpeechToTextError",
    "TextToSpeechError",
    "PluginError",
    "PluginExecutionError",
    "AudioDeviceError",
    "AuthenticationError",
]
