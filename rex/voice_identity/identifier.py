"""Real-time speaker identification from raw audio.

Public API
----------
:func:`identify_speaker`
    Embeds a raw audio array, compares it against all enrolled users via
    cosine similarity, and returns the best-matching user ID, or ``None``
    if no enrolled user exceeds *threshold*.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rex.voice_identity.embedding_backends import SyntheticEmbeddingBackend
from rex.voice_identity.embeddings_store import EmbeddingsStore
from rex.voice_identity.recognizer import SpeakerRecognizer
from rex.voice_identity.types import VoiceIdentityConfig

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_DIR = Path("Memory")
_DEFAULT_THRESHOLD = 0.75


def identify_speaker(
    audio: np.ndarray,
    *,
    base_dir: Path | str | None = None,
    backend: object | None = None,
    threshold: float = _DEFAULT_THRESHOLD,
) -> str | None:
    """Identify which enrolled user is speaking.

    Computes a speaker embedding from *audio*, compares it against every
    enrolled user's stored embedding, and returns the matched user ID if
    the highest cosine-similarity score is at or above *threshold*.

    Args:
        audio: Raw audio array (float32, mono).
        base_dir: Root directory for per-user data.  Defaults to ``Memory/``.
        backend: Embedding backend with ``embed(bytes) -> list[float]`` and
            a ``model_id`` property.  Defaults to
            :class:`~rex.voice_identity.embedding_backends.SyntheticEmbeddingBackend`.
        threshold: Minimum cosine similarity to accept a match.  Values in
            ``[0, 1]``; default is ``0.75``.

    Returns:
        The matched ``user_id`` string, or ``None`` if no enrolled user
        meets the threshold.
    """
    resolved_base = Path(base_dir) if base_dir is not None else _DEFAULT_MEMORY_DIR
    if backend is None:
        backend = SyntheticEmbeddingBackend()

    # Load all enrolled users; bail early if nobody is enrolled.
    store = EmbeddingsStore(resolved_base)
    enrolled = store.load_all()
    if not enrolled:
        logger.debug("identify_speaker: no enrolled users; returning None")
        return None

    # Compute an embedding for the incoming audio.
    try:
        audio_bytes = np.asarray(audio, dtype=np.float32).tobytes()
        input_vec: list[float] = backend.embed(audio_bytes)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("identify_speaker: embedding failed: %s", exc)
        return None

    # SpeakerRecognizer uses cosine similarity; set both thresholds to the
    # same value so anything below threshold → UNKNOWN (review == unknown).
    config = VoiceIdentityConfig(
        accept_threshold=threshold,
        review_threshold=threshold,
    )
    recognizer = SpeakerRecognizer(config)
    result = recognizer.recognize(input_vec, enrolled)

    if result.best_user_id is not None:
        logger.debug(
            "identify_speaker: matched user=%r score=%.3f threshold=%.3f",
            result.best_user_id,
            result.score,
            threshold,
        )
        return result.best_user_id

    logger.debug(
        "identify_speaker: no match (best_score=%.3f threshold=%.3f)",
        result.score,
        threshold,
    )
    return None
