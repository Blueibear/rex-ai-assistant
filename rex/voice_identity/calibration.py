"""Voice identity calibration.

Computes recommended thresholds from enrolled embeddings so that recognition
can be reliable in practice.

Calibration logic
-----------------
* **No enrolled users** — return a report indicating calibration cannot run.
* **Single enrolled user** — without a second user for inter-user comparison,
  conservative defaults are recommended with a note to enroll more users.
* **Multiple enrolled users** — compute pairwise inter-user cosine similarities.
  Set ``review_threshold`` just below the minimum inter-user similarity (to
  prevent false-accept between different users) and ``accept_threshold`` a
  fixed gap above that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rex.voice_identity.embeddings_store import EmbeddingsStore
from rex.voice_identity.recognizer import _cosine_similarity
from rex.voice_identity.types import VoiceIdentityConfig

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Result of a calibration run.

    Attributes:
        enrolled_users: User IDs that were used in calibration.
        recommended_accept_threshold: Recommended minimum score for RECOGNIZED.
        recommended_review_threshold: Recommended minimum score for REVIEW.
        min_inter_user_similarity: Minimum pairwise cosine similarity between
            enrolled users.  ``None`` if fewer than 2 users are enrolled.
        notes: Human-readable notes about the calibration.
    """

    enrolled_users: list[str] = field(default_factory=list)
    recommended_accept_threshold: float = 0.85
    recommended_review_threshold: float = 0.65
    min_inter_user_similarity: float | None = None
    notes: list[str] = field(default_factory=list)


def calibrate(store: EmbeddingsStore, config: VoiceIdentityConfig) -> CalibrationReport:
    """Compute recommended thresholds from enrolled embeddings.

    Args:
        store: Embeddings store to read enrolled users from.
        config: Current voice identity configuration (used as fallback for
            default thresholds when calibration cannot be computed).

    Returns:
        :class:`CalibrationReport` with recommended thresholds and notes.
    """
    enrolled = store.load_all()
    report = CalibrationReport(enrolled_users=sorted(enrolled.keys()))

    if len(enrolled) == 0:
        report.notes.append(
            "No users enrolled. Enroll at least one user with 'rex voice-id enroll'."
        )
        report.recommended_accept_threshold = config.accept_threshold
        report.recommended_review_threshold = config.review_threshold
        return report

    if len(enrolled) == 1:
        user = next(iter(enrolled))
        report.notes.append(
            f"Only one user enrolled ({user!r}). Enroll a second user for "
            "inter-user separation calibration. Using conservative defaults."
        )
        report.recommended_accept_threshold = 0.85
        report.recommended_review_threshold = 0.65
        return report

    # --- Multiple users: compute pairwise inter-user similarities ---
    users = sorted(enrolled.keys())
    inter_similarities: list[float] = []

    for i, uid_a in enumerate(users):
        for uid_b in users[i + 1 :]:
            vec_a = enrolled[uid_a].vector
            vec_b = enrolled[uid_b].vector
            try:
                sim = _cosine_similarity(vec_a, vec_b)
                inter_similarities.append(sim)
                logger.debug(
                    "Inter-user similarity %s vs %s: %.4f",
                    uid_a,
                    uid_b,
                    sim,
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping similarity between %s and %s: %s",
                    uid_a,
                    uid_b,
                    exc,
                )

    if not inter_similarities:
        report.notes.append(
            "Could not compute pairwise similarities (possible dimension mismatch). "
            "Re-enroll users with the same backend and model."
        )
        report.recommended_accept_threshold = config.accept_threshold
        report.recommended_review_threshold = config.review_threshold
        return report

    min_sim = min(inter_similarities)
    report.min_inter_user_similarity = min_sim

    # Conservative threshold derivation:
    #   review_threshold  = min_inter_user_similarity - margin
    #   accept_threshold  = review_threshold + gap
    # The margin keeps false-accepts below the worst observed inter-user
    # similarity, and the gap ensures meaningful confidence before auto-accept.
    _margin = 0.05
    _gap = 0.15

    raw_review = min_sim - _margin
    raw_accept = raw_review + _gap

    # Clamp to sensible ranges
    recommended_review = round(max(0.50, min(0.90, raw_review)), 3)
    recommended_accept = round(max(recommended_review + 0.05, min(0.95, raw_accept)), 3)

    report.recommended_review_threshold = recommended_review
    report.recommended_accept_threshold = recommended_accept

    n_pairs = len(inter_similarities)
    n_users = len(users)
    report.notes.append(
        f"Minimum inter-user similarity: {min_sim:.4f} "
        f"({n_pairs} pair(s) across {n_users} enrolled users)."
    )
    report.notes.append(
        f"Recommended thresholds: " f"accept={recommended_accept}, review={recommended_review}."
    )

    if min_sim > 0.85:
        report.notes.append(
            "WARNING: Enrolled users have high inter-user similarity "
            f"(min={min_sim:.4f}). This may cause false-accept between users. "
            "Consider enrolling with a real embedding backend (speechbrain) for "
            "better speaker separation."
        )

    return report
