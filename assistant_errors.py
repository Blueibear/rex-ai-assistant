"""Backward compatibility wrapper - imports from rex.assistant_errors.

New code should import directly from rex.assistant_errors.

.. deprecated::
    Import from ``rex.assistant_errors`` instead. This shim will be removed in a future cycle.
"""

from __future__ import annotations

import warnings

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

warnings.warn(
    "Importing from root-level 'assistant_errors' is deprecated. "
    "Use 'from rex.assistant_errors import ...' instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
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
