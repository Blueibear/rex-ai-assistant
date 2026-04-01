"""Voice enrollment — capture audio samples and store speaker embeddings.

Public API
----------
:func:`enroll_user`
    Average the per-sample embeddings, persist as both JSON (via
    :class:`~rex.voice_identity.embeddings_store.EmbeddingsStore`) and as a
    NumPy ``.npy`` file for external tooling.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from rex.voice_identity.embedding_backends import SyntheticEmbeddingBackend
from rex.voice_identity.embeddings_store import EmbeddingsStore
from rex.voice_identity.types import VoiceEmbedding

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_DIR = Path("Memory")
_MIN_SAMPLES = 3
_NPY_FILENAME = "voice_embedding.npy"


def enroll_user(
    user_id: str,
    audio_samples: list[np.ndarray],
    *,
    base_dir: Path | str | None = None,
    backend: object | None = None,
) -> None:
    """Enroll *user_id* using the supplied audio samples.

    Computes a speaker embedding for each sample, averages them, normalises
    to unit length, and persists the result in two forms:

    * **JSON** — via :class:`~rex.voice_identity.embeddings_store.EmbeddingsStore`
      at ``<base_dir>/<user_id>/voice_embeddings.json`` (canonical store).
    * **NumPy .npy** — at ``<base_dir>/<user_id>/voice_embedding.npy``
      for external tooling.

    Args:
        user_id: Identifier for the user being enrolled.  Must be a simple
            name (no path separators).
        audio_samples: List of raw audio arrays (float32, mono).  At least
            :data:`_MIN_SAMPLES` (3) samples are required.
        base_dir: Root directory for per-user data.  Defaults to ``Memory/``.
        backend: Embedding backend exposing ``embed(bytes) -> list[float]``
            and a ``model_id`` property.  Defaults to
            :class:`~rex.voice_identity.embedding_backends.SyntheticEmbeddingBackend`.

    Raises:
        ValueError: If fewer than :data:`_MIN_SAMPLES` audio samples are
            provided.
    """
    if len(audio_samples) < _MIN_SAMPLES:
        raise ValueError(
            f"enroll_user requires at least {_MIN_SAMPLES} audio samples; "
            f"got {len(audio_samples)}"
        )

    resolved_base = Path(base_dir) if base_dir is not None else _DEFAULT_MEMORY_DIR
    if backend is None:
        backend = SyntheticEmbeddingBackend()

    # Compute per-sample embeddings and accumulate the sum.
    embedding_sum: list[float] | None = None
    dim: int = 0
    for sample in audio_samples:
        audio_bytes = sample.astype(np.float32).tobytes()
        vec: list[float] = backend.embed(audio_bytes)  # type: ignore[attr-defined]
        if embedding_sum is None:
            dim = len(vec)
            embedding_sum = list(vec)
        else:
            embedding_sum = [embedding_sum[j] + vec[j] for j in range(dim)]

    assert embedding_sum is not None  # at least _MIN_SAMPLES samples guaranteed above
    n = len(audio_samples)
    averaged = [v / n for v in embedding_sum]

    # Normalise to unit length.
    mag = math.sqrt(sum(v * v for v in averaged))
    if mag > 0.0:
        averaged = [v / mag for v in averaged]

    model_id: str = getattr(backend, "model_id", "unknown")
    voice_emb = VoiceEmbedding(
        vector=averaged,
        model_id=model_id,
        sample_count=n,
    )

    # Persist via EmbeddingsStore (JSON — canonical store shared with recognizer).
    store = EmbeddingsStore(resolved_base)
    store.save(user_id, voice_emb)

    # Also persist as .npy for tooling that consumes the numpy format directly.
    npy_path = resolved_base / user_id / _NPY_FILENAME
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), np.array(averaged, dtype=np.float32))

    logger.info(
        "Enrolled user %r: %d samples, embedding dim=%d, model=%s",
        user_id,
        n,
        dim,
        model_id,
    )
