"""Embedding backends for voice identity.

Provides a clean interface for generating speaker embedding vectors from raw
audio bytes.  Two implementations are available:

1. ``SyntheticEmbeddingBackend`` (default) — stdlib-only, deterministic,
   suitable for testing and development without optional deps.
2. ``SpeechBrainBackend`` — optional, requires the ``voice-id`` extras
   (``pip install '.[voice-id]'``).  Loads the ECAPA-TDNN model on first use.

Design notes
------------
* No heavy imports happen at module level or ``__init__`` time.
* The real backend only loads a model when ``embed()`` is first called.
* All backends return a list of floats (unit-normalised).
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_INSTALL_HINT = "Install the voice-id extras: pip install '.[voice-id]'"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _bytes_to_unit_vector(data: bytes, dim: int) -> list[float]:
    """Convert bytes to a deterministic unit-normalised float vector.

    Uses repeated SHA-256 hashing to fill the required number of dimensions.
    Pure stdlib — no external dependencies.

    Args:
        data: Arbitrary bytes to derive a vector from.
        dim: Target vector dimensionality.

    Returns:
        A list of ``dim`` floats representing a unit vector.
    """
    floats: list[float] = []
    seed = data
    while len(floats) < dim:
        digest = hashlib.sha256(seed).digest()
        seed = digest  # next round uses previous hash as input
        # Interpret each 4-byte group as an unsigned int, map to [-0.5, 0.5]
        for i in range(0, len(digest) - 3, 4):
            val = int.from_bytes(digest[i : i + 4], "little") / (2**32)
            floats.append(val - 0.5)
    vector = floats[:dim]
    # Normalise to unit length
    mag = math.sqrt(sum(x * x for x in vector))
    if mag > 0.0:
        vector = [x / mag for x in vector]
    return vector


# ---------------------------------------------------------------------------
# Synthetic backend (stdlib-only, always available)
# ---------------------------------------------------------------------------


class SyntheticEmbeddingBackend:
    """Deterministic embedding backend using stdlib SHA-256 hashing.

    Suitable for testing, development, and enrollment without optional deps.
    Two calls with identical ``audio_bytes`` produce the same embedding.

    Args:
        dim: Output embedding dimensionality.  Default is 192 to match the
            scaffolding default in :class:`~rex.voice_identity.types.VoiceIdentityConfig`.
    """

    def __init__(self, dim: int = 192) -> None:
        self._dim = dim

    @property
    def model_id(self) -> str:
        """Identifier for this backend."""
        return "synthetic"

    def embed(self, audio_bytes: bytes, sample_rate: int = 16000) -> list[float]:
        """Generate a deterministic embedding from raw audio bytes.

        Args:
            audio_bytes: Raw audio data (any format; used as hash input).
            sample_rate: Ignored by this backend; accepted for interface
                compatibility with real backends.

        Returns:
            A unit-normalised vector of length ``self._dim``.
        """
        return _bytes_to_unit_vector(audio_bytes, self._dim)


# ---------------------------------------------------------------------------
# SpeechBrain backend (optional, lazy model loading)
# ---------------------------------------------------------------------------


class SpeechBrainBackend:
    """Optional embedding backend using SpeechBrain ECAPA-TDNN.

    Requires the voice-id extras: ``pip install '.[voice-id]'``

    The model is loaded lazily on the first ``embed()`` call.  No network
    access or model download occurs during import or ``__init__``.

    Args:
        model_source: HuggingFace model ID or local path.  Defaults to the
            SpeechBrain ECAPA-TDNN speaker recognition model.
        savedir: Directory for caching downloaded model files.
    """

    _DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(
        self,
        model_source: str | None = None,
        savedir: str | Path = "pretrained_models/spkrec-ecapa-voxceleb",
    ) -> None:
        self._model_source = model_source or self._DEFAULT_SOURCE
        self._savedir = str(savedir)
        self._model = None  # loaded lazily on first embed()

    @property
    def model_id(self) -> str:
        """Identifier for this backend."""
        return "speechbrain"

    def _load_model(self) -> None:
        """Load the SpeechBrain ECAPA model on first use."""
        from importlib.util import find_spec

        if find_spec("speechbrain") is None:
            raise ImportError(f"speechbrain is not installed. {_INSTALL_HINT}")
        try:
            from speechbrain.pretrained import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=self._model_source,
                savedir=self._savedir,
                run_opts={"device": "cpu"},
            )
            logger.info("SpeechBrain ECAPA model loaded from %s", self._model_source)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SpeechBrain model from {self._model_source!r}: {exc}. "
                "Ensure the model is downloaded or that network access is available."
            ) from exc

    def embed(self, audio_bytes: bytes, sample_rate: int = 16000) -> list[float]:
        """Generate a speaker embedding using SpeechBrain ECAPA-TDNN.

        Loads the model on first call.

        Args:
            audio_bytes: Raw float32 LE PCM audio bytes.
            sample_rate: Sample rate of the audio.

        Returns:
            Speaker embedding as a list of floats.

        Raises:
            ImportError: If speechbrain is not installed.
            RuntimeError: If model loading or inference fails.
        """
        if self._model is None:
            self._load_model()
        try:
            import numpy as np
            import torch

            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            audio_tensor = torch.tensor(audio_np).unsqueeze(0)
            lengths = torch.tensor([1.0])
            with torch.no_grad():
                embedding = self._model.encode_batch(audio_tensor, lengths)  # type: ignore[attr-defined]
            return embedding.squeeze().tolist()  # type: ignore[no-any-return]
        except Exception as exc:
            raise RuntimeError(f"SpeechBrain embedding failed: {exc}") from exc
