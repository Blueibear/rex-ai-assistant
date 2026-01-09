"""Compatibility shims for third-party library incompatibilities.

This module patches known issues between library versions used in Rex.
It must be imported early (from rex/__init__.py) before libraries are used.
"""

from __future__ import annotations

import sys
from types import ModuleType


def ensure_transformers_compatibility() -> None:
    """Ensure transformers compatibility with Coqui TTS.

    Coqui TTS 0.22.0 tries to import BeamSearchScorer from transformers,
    but transformers 4.57.3 moved it to transformers.generation.
    This shim makes it available at the old location.
    """
    try:
        import transformers
        from transformers.generation import BeamSearchScorer

        # Patch BeamSearchScorer back into the main transformers namespace
        if not hasattr(transformers, 'BeamSearchScorer'):
            transformers.BeamSearchScorer = BeamSearchScorer

    except ImportError:
        # If transformers isn't installed or BeamSearchScorer doesn't exist,
        # skip the patch - TTS will fail later with a clearer error
        pass


# Apply compatibility shims immediately when this module is imported
ensure_transformers_compatibility()


__all__ = ["ensure_transformers_compatibility"]
