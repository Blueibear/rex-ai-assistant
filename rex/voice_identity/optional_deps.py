"""Import guards for optional heavy voice-identity dependencies.

The base install does not include speechbrain or resemblyzer.  Any code
path that needs them must call the helpers here, which return ``None``
and log a clear message when the packages are missing.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Install the voice-id extras to enable real speaker embeddings: " "pip install '.[voice-id]'"
)


def import_speechbrain():
    """Attempt to import speechbrain. Returns the module or ``None``."""
    if find_spec("speechbrain") is None:
        logger.info("speechbrain is not installed. %s", _INSTALL_HINT)
        return None
    import speechbrain  # type: ignore[import-untyped]

    return speechbrain


def import_resemblyzer():
    """Attempt to import resemblyzer. Returns the module or ``None``."""
    if find_spec("resemblyzer") is None:
        logger.info("resemblyzer is not installed. %s", _INSTALL_HINT)
        return None
    import resemblyzer  # type: ignore[import-untyped]

    return resemblyzer


def check_voice_id_available() -> bool:
    """Return True if at least one real embedding backend is available."""
    return find_spec("speechbrain") is not None or find_spec("resemblyzer") is not None
