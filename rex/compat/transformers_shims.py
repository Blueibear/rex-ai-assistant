"""Compatibility shims for Hugging Face Transformers library.

Starting in transformers 4.38+, BeamSearchScorer was moved from top-level
exports to internal modules. This causes import errors in libraries like
Coqui TTS that still expect it at the top level.

This module patches the transformers package to restore backward compatibility.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def ensure_transformers_compatibility() -> None:
    """Ensure transformers backward compatibility for legacy dependencies.

    This function patches the transformers module to expose BeamSearchScorer
    at the top level, which was removed in transformers 4.38+ but is still
    expected by some libraries like Coqui TTS.

    Must be called before any code that imports BeamSearchScorer from transformers.
    """
    try:
        import transformers

        # Check if BeamSearchScorer is already available at top level
        if hasattr(transformers, "BeamSearchScorer"):
            logger.debug("BeamSearchScorer already available in transformers top-level")
            return

        # Try to import from known internal locations (order matters)
        beam_search_scorer = None
        import_locations = [
            "transformers.generation.beam_search",
            "transformers.generation_beam_search",
            "transformers.generation",
        ]

        for location in import_locations:
            try:
                if location in sys.modules:
                    module = sys.modules[location]
                else:
                    module = __import__(location, fromlist=["BeamSearchScorer"])

                if hasattr(module, "BeamSearchScorer"):
                    beam_search_scorer = getattr(module, "BeamSearchScorer")
                    logger.info(f"Found BeamSearchScorer in {location}")
                    break
            except (ImportError, AttributeError):
                continue

        if beam_search_scorer is None:
            # Last resort: try to import directly
            try:
                from transformers.generation.beam_search import BeamSearchScorer
                beam_search_scorer = BeamSearchScorer
                logger.info("Found BeamSearchScorer via direct import")
            except ImportError:
                logger.warning(
                    "Could not find BeamSearchScorer in transformers. "
                    "TTS or other libraries may fail to import."
                )
                return

        # Monkey-patch transformers to expose BeamSearchScorer at top level
        setattr(transformers, "BeamSearchScorer", beam_search_scorer)
        logger.info("Successfully patched transformers.BeamSearchScorer for backward compatibility")

    except ImportError:
        logger.warning("transformers not installed, skipping compatibility shim")
    except Exception as e:
        logger.error(f"Failed to apply transformers compatibility shim: {e}")


# Auto-apply shim when module is imported
ensure_transformers_compatibility()
