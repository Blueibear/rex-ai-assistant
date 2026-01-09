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

        # Check if BeamSearchScorer is already available (might be an older version)
        if hasattr(transformers, 'BeamSearchScorer'):
            return

        # Try multiple import locations (transformers has moved it around in different versions)
        BeamSearchScorer = None

        # Try transformers.generation.beam_search (4.38+)
        try:
            from transformers.generation.beam_search import BeamSearchScorer
        except (ImportError, AttributeError):
            pass

        # Try transformers.generation (older 4.x versions)
        if BeamSearchScorer is None:
            try:
                from transformers.generation import BeamSearchScorer
            except (ImportError, AttributeError):
                pass

        # Try transformers.generation_utils (very old versions)
        if BeamSearchScorer is None:
            try:
                from transformers.generation_utils import BeamSearchScorer
            except (ImportError, AttributeError):
                pass

        if BeamSearchScorer is None:
            # Couldn't find it anywhere, give up
            return

        # Patch it into the transformers module namespace (multiple methods for robustness)
        transformers.BeamSearchScorer = BeamSearchScorer
        transformers.__dict__['BeamSearchScorer'] = BeamSearchScorer

        # Also patch it into sys.modules['transformers'] to ensure it's visible to all imports
        if 'transformers' in sys.modules:
            sys.modules['transformers'].BeamSearchScorer = BeamSearchScorer
            sys.modules['transformers'].__dict__['BeamSearchScorer'] = BeamSearchScorer

    except (ImportError, AttributeError) as e:
        # If transformers isn't installed or BeamSearchScorer doesn't exist,
        # skip the patch - TTS will fail later with a clearer error
        pass


# Apply compatibility shims immediately when this module is imported
ensure_transformers_compatibility()


__all__ = ["ensure_transformers_compatibility"]
