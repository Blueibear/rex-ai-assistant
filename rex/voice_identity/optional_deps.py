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
    import speechbrain

    return speechbrain


def import_resemblyzer():
    """Attempt to import resemblyzer. Returns the module or ``None``."""
    if find_spec("resemblyzer") is None:
        logger.info("resemblyzer is not installed. %s", _INSTALL_HINT)
        return None
    import resemblyzer

    return resemblyzer


def check_voice_id_available() -> bool:
    """Return True if at least one real embedding backend is available."""
    return find_spec("speechbrain") is not None or find_spec("resemblyzer") is not None


def get_embedding_backend(model_id: str = "synthetic", dim: int = 192):
    """Return the appropriate embedding backend for the given model ID.

    For ``model_id="synthetic"``, returns a
    :class:`~rex.voice_identity.embedding_backends.SyntheticEmbeddingBackend`
    using stdlib-only hashing.

    For ``model_id="speechbrain"`` or ``"ecapa"``, checks that speechbrain is
    installed and returns a
    :class:`~rex.voice_identity.embedding_backends.SpeechBrainBackend`.
    The model is **not** loaded until the first call to ``embed()``.

    Args:
        model_id: Backend identifier.  Valid values: ``"synthetic"``,
            ``"speechbrain"``, ``"ecapa"``.
        dim: Embedding dimensionality (only used by the synthetic backend).

    Returns:
        An object implementing the embedding backend interface.

    Raises:
        ImportError: If a real backend is requested but the required package
            is missing.  The error message includes the install hint.
        ValueError: If ``model_id`` is not recognised.
    """
    from rex.voice_identity.embedding_backends import (
        SpeechBrainBackend,
        SyntheticEmbeddingBackend,
    )

    if model_id == "synthetic":
        return SyntheticEmbeddingBackend(dim=dim)

    if model_id in ("speechbrain", "ecapa"):
        if find_spec("speechbrain") is None:
            raise ImportError(f"speechbrain is required for model_id={model_id!r}. {_INSTALL_HINT}")
        return SpeechBrainBackend()

    raise ValueError(
        f"Unknown voice identity model_id: {model_id!r}. "
        "Valid values: 'synthetic', 'speechbrain'."
    )
