"""Speaker recognition using cosine similarity on embedding vectors.

This module uses pure-Python math (no numpy required) so it works in the
base install.  A real embedding model can be plugged in later; the
recognizer itself only cares about pre-computed vectors.
"""

from __future__ import annotations

import logging
import math

from rex.voice_identity.types import (
    RecognitionDecision,
    RecognitionResult,
    VoiceEmbedding,
    VoiceIdentityConfig,
)

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using pure Python.

    Returns a value in [-1, 1].  Returns 0.0 if either vector has zero
    magnitude (degenerate case).
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class SpeakerRecognizer:
    """Compare an input embedding against enrolled users.

    Parameters:
        config: Voice identity configuration with thresholds.
    """

    def __init__(self, config: VoiceIdentityConfig) -> None:
        self._config = config

    def recognize(
        self,
        input_embedding: list[float],
        enrolled: dict[str, VoiceEmbedding],
    ) -> RecognitionResult:
        """Match *input_embedding* against all *enrolled* user embeddings.

        Returns a :class:`RecognitionResult` with the best match and the
        decision (recognized / review / unknown).
        """
        if not enrolled:
            return RecognitionResult(
                decision=RecognitionDecision.UNKNOWN,
                score=0.0,
                accept_threshold=self._config.accept_threshold,
                review_threshold=self._config.review_threshold,
            )

        best_user: str | None = None
        best_score: float = -1.0

        for user_id, emb in enrolled.items():
            try:
                score = _cosine_similarity(input_embedding, emb.vector)
            except ValueError as exc:
                logger.warning("Skipping user %s: %s", user_id, exc)
                continue
            if score > best_score:
                best_score = score
                best_user = user_id

        decision = self._decide(best_score)

        return RecognitionResult(
            decision=decision,
            best_user_id=best_user if decision != RecognitionDecision.UNKNOWN else None,
            score=best_score,
            accept_threshold=self._config.accept_threshold,
            review_threshold=self._config.review_threshold,
        )

    def _decide(self, score: float) -> RecognitionDecision:
        """Map a similarity score to a recognition decision."""
        if score >= self._config.accept_threshold:
            return RecognitionDecision.RECOGNIZED
        if score >= self._config.review_threshold:
            return RecognitionDecision.REVIEW
        return RecognitionDecision.UNKNOWN
