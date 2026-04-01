"""Unit tests for rex.voice_identity.identifier.identify_speaker."""

from __future__ import annotations

import numpy as np

from rex.voice_identity.embedding_backends import SyntheticEmbeddingBackend
from rex.voice_identity.enrollment import enroll_user
from rex.voice_identity.identifier import identify_speaker


def _make_audio(seed: int, length: int = 16000) -> np.ndarray:
    """Return a deterministic float32 audio array seeded by *seed*."""
    rng = np.random.default_rng(seed)
    return rng.random(length).astype(np.float32)


def _enroll(tmp_path, user_id: str, seed: int) -> None:
    """Enroll *user_id* with three samples derived from *seed*."""
    samples = [_make_audio(seed + i) for i in range(3)]
    enroll_user(user_id, samples, base_dir=tmp_path, backend=SyntheticEmbeddingBackend())


# ---------------------------------------------------------------------------
# Enrolled user is recognised above threshold
# ---------------------------------------------------------------------------


def test_enrolled_user_recognized(tmp_path):
    """An enrolled user's audio should be matched by identify_speaker."""
    _enroll(tmp_path, "alice", seed=1)

    # Use the same seed → same deterministic audio → same embedding.
    audio = _make_audio(seed=1)
    result = identify_speaker(audio, base_dir=tmp_path, threshold=0.5)
    assert result == "alice"


# ---------------------------------------------------------------------------
# Unrecognised speaker returns None (no enrolled users)
# ---------------------------------------------------------------------------


def test_unrecognized_speaker_no_enrollments(tmp_path):
    """identify_speaker returns None when no users are enrolled."""
    audio = _make_audio(seed=99)
    result = identify_speaker(audio, base_dir=tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# Below-threshold match returns None
# ---------------------------------------------------------------------------


def test_below_threshold_returns_none(tmp_path):
    """A score below the threshold must return None."""
    _enroll(tmp_path, "bob", seed=10)

    # Completely different audio → cosine similarity will be low.
    audio = _make_audio(seed=999)
    # Force threshold to 1.0 (nothing can match) to guarantee failure.
    result = identify_speaker(audio, base_dir=tmp_path, threshold=1.0)
    assert result is None


# ---------------------------------------------------------------------------
# Multiple enrolled users — correct one is returned
# ---------------------------------------------------------------------------


def test_correct_user_selected_among_multiple(tmp_path):
    """With two enrolled users the closer embedding wins."""
    _enroll(tmp_path, "alice", seed=1)
    _enroll(tmp_path, "carol", seed=500)

    # Audio matching alice's seed → should match alice.
    audio = _make_audio(seed=1)
    result = identify_speaker(audio, base_dir=tmp_path, threshold=0.5)
    assert result == "alice"


# ---------------------------------------------------------------------------
# Fallback to default profile: non-matching audio does not crash
# ---------------------------------------------------------------------------


def test_unrecognized_does_not_crash(tmp_path):
    """identify_speaker must not raise when no match is found."""
    _enroll(tmp_path, "dave", seed=20)

    audio = _make_audio(seed=999)
    # threshold=1.0 forces a miss; should return None gracefully.
    result = identify_speaker(audio, base_dir=tmp_path, threshold=1.0)
    assert result is None
