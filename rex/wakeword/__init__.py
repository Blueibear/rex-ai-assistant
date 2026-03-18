"""Wake word detection subpackage for Rex AI Assistant."""

from __future__ import annotations

from rex.wakeword.listener import WakeWordListener, build_default_detector
from rex.wakeword.utils import detect_wakeword, load_wakeword_model

__all__ = [
    "WakeWordListener",
    "build_default_detector",
    "detect_wakeword",
    "load_wakeword_model",
]
