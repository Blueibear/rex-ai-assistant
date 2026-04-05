"""Data types for voice identity recognition.

All types are pure-Python dataclasses with no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RecognitionDecision(StrEnum):
    """Outcome of a speaker recognition attempt."""

    RECOGNIZED = "recognized"
    REVIEW = "review"
    UNKNOWN = "unknown"


@dataclass
class VoiceEmbedding:
    """A single speaker embedding vector with metadata.

    Attributes:
        vector: List of floats representing the speaker embedding.
        model_id: Identifier for the model that produced this embedding.
        sample_count: Number of audio samples used to compute the embedding.
        updated_at: ISO-8601 timestamp of the last update.
    """

    vector: list[float]
    model_id: str = "synthetic"
    sample_count: int = 1
    updated_at: str = ""


@dataclass
class RecognitionResult:
    """Result of comparing an input embedding against enrolled users.

    Attributes:
        decision: One of recognized, review, or unknown.
        best_user_id: The user ID of the best match, or None.
        score: Cosine similarity score of the best match.
        accept_threshold: Threshold used for the ``recognized`` decision.
        review_threshold: Threshold used for the ``review`` decision.
    """

    decision: RecognitionDecision
    best_user_id: str | None = None
    score: float = 0.0
    accept_threshold: float = 0.0
    review_threshold: float = 0.0


@dataclass
class VoiceIdentityConfig:
    """Configuration for the voice identity subsystem.

    Loaded from ``voice_identity`` section of ``config/rex_config.json``.

    Attributes:
        enabled: Whether voice identity is active.
        accept_threshold: Minimum cosine similarity for ``recognized``.
        review_threshold: Minimum cosine similarity for ``review``.
            Scores below this are ``unknown``.
        embedding_dim: Expected dimensionality of embedding vectors.
        model_id: Identifier for the active embedding model.
    """

    enabled: bool = False
    accept_threshold: float = 0.85
    review_threshold: float = 0.65
    embedding_dim: int = 192
    model_id: str = "synthetic"
