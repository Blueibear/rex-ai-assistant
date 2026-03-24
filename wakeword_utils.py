"""Wake-word detection utilities for Rex.

Compatibility shim - imports from rex.wakeword_utils.
All configuration now comes from rex_config.json, not environment variables.
"""

from __future__ import annotations

# Re-export all functionality from rex.wakeword_utils
from rex.wakeword_utils import (
    detect_wakeword,
    load_wakeword_model,
)

__all__ = ["load_wakeword_model", "detect_wakeword"]
