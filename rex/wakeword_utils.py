"""Wake-word detection utilities for Rex."""

from __future__ import annotations

from .wakeword.utils import detect_wakeword, load_wakeword_model

__all__ = ["load_wakeword_model", "detect_wakeword"]
