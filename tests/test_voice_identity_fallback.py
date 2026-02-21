"""Tests for voice identity scaffolding (BL-009).

All tests use synthetic embedding vectors and ``tmp_path`` so they are
fully offline and do not write to tracked repository paths.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from rex.voice_identity.embeddings_store import EmbeddingsStore
from rex.voice_identity.fallback_flow import resolve_speaker_identity
from rex.voice_identity.recognizer import SpeakerRecognizer, _cosine_similarity
from rex.voice_identity.types import (
    RecognitionDecision,
    VoiceEmbedding,
    VoiceIdentityConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vector(dim: int, index: int) -> list[float]:
    """Return a unit vector with 1.0 at *index* and 0.0 elsewhere."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _make_embedding(vector: list[float], model_id: str = "synthetic") -> VoiceEmbedding:
    return VoiceEmbedding(
        vector=vector,
        model_id=model_id,
        sample_count=3,
        updated_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# EmbeddingsStore tests
# ---------------------------------------------------------------------------


class TestEmbeddingsStore:
    def test_save_and_load(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        emb = _make_embedding([0.1, 0.2, 0.3])
        store.save("alice", emb)

        loaded = store.load("alice")
        assert loaded is not None
        assert loaded.vector == [0.1, 0.2, 0.3]
        assert loaded.model_id == "synthetic"
        assert loaded.sample_count == 3

    def test_load_nonexistent_returns_none(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        assert store.load("nobody") is None

    def test_delete(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        emb = _make_embedding([1.0, 0.0])
        store.save("bob", emb)
        assert store.delete("bob") is True
        assert store.load("bob") is None
        assert store.delete("bob") is False

    def test_list_enrolled_users(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        store.save("alice", _make_embedding([1.0]))
        store.save("bob", _make_embedding([2.0]))
        assert store.list_enrolled_users() == ["alice", "bob"]

    def test_load_all(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        store.save("alice", _make_embedding([0.5, 0.5]))
        store.save("bob", _make_embedding([0.3, 0.7]))
        all_emb = store.load_all()
        assert set(all_emb.keys()) == {"alice", "bob"}
        assert all_emb["alice"].vector == [0.5, 0.5]

    def test_save_creates_directories(self, tmp_path):
        base = tmp_path / "deep" / "nested"
        store = EmbeddingsStore(base)
        store.save("user1", _make_embedding([1.0]))
        assert (base / "user1" / "voice_embeddings.json").exists()

    def test_corrupt_file_returns_none(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        user_dir = tmp_path / "bad"
        user_dir.mkdir()
        (user_dir / "voice_embeddings.json").write_text("not json", encoding="utf-8")
        assert store.load("bad") is None

    def test_save_overwrite(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        store.save("alice", _make_embedding([1.0, 0.0]))
        store.save("alice", _make_embedding([0.0, 1.0]))
        loaded = store.load("alice")
        assert loaded is not None
        assert loaded.vector == [0.0, 1.0]

    def test_json_structure(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        store.save("alice", _make_embedding([0.1, 0.2], model_id="test-model"))
        raw = json.loads((tmp_path / "alice" / "voice_embeddings.json").read_text(encoding="utf-8"))
        assert raw["model_id"] == "test-model"
        assert raw["sample_count"] == 3
        assert "updated_at" in raw
        assert raw["embedding"] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            _cosine_similarity([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# SpeakerRecognizer tests
# ---------------------------------------------------------------------------


class TestSpeakerRecognizer:
    def _config(self, accept=0.85, review=0.65):
        return VoiceIdentityConfig(
            enabled=True,
            accept_threshold=accept,
            review_threshold=review,
        )

    def test_recognized(self):
        cfg = self._config()
        rec = SpeakerRecognizer(cfg)
        enrolled = {"alice": _make_embedding([1.0, 0.0, 0.0])}
        result = rec.recognize([1.0, 0.0, 0.0], enrolled)
        assert result.decision == RecognitionDecision.RECOGNIZED
        assert result.best_user_id == "alice"
        assert result.score > 0.99

    def test_review(self):
        cfg = self._config(accept=0.95, review=0.5)
        rec = SpeakerRecognizer(cfg)
        # Vectors with moderate similarity
        enrolled = {"bob": _make_embedding([1.0, 0.5, 0.0])}
        result = rec.recognize([1.0, 0.0, 0.5], enrolled)
        assert result.decision == RecognitionDecision.REVIEW
        assert result.best_user_id == "bob"

    def test_unknown(self):
        cfg = self._config(accept=0.95, review=0.9)
        rec = SpeakerRecognizer(cfg)
        enrolled = {"charlie": _make_embedding([1.0, 0.0, 0.0])}
        result = rec.recognize([0.0, 1.0, 0.0], enrolled)
        assert result.decision == RecognitionDecision.UNKNOWN
        assert result.best_user_id is None

    def test_no_enrolled_users(self):
        cfg = self._config()
        rec = SpeakerRecognizer(cfg)
        result = rec.recognize([1.0, 0.0], {})
        assert result.decision == RecognitionDecision.UNKNOWN
        assert result.best_user_id is None

    def test_multiple_enrolled_picks_best(self):
        cfg = self._config()
        rec = SpeakerRecognizer(cfg)
        enrolled = {
            "alice": _make_embedding(_unit_vector(4, 0)),
            "bob": _make_embedding(_unit_vector(4, 1)),
        }
        result = rec.recognize(_unit_vector(4, 1), enrolled)
        assert result.decision == RecognitionDecision.RECOGNIZED
        assert result.best_user_id == "bob"

    def test_thresholds_in_result(self):
        cfg = self._config(accept=0.80, review=0.60)
        rec = SpeakerRecognizer(cfg)
        enrolled = {"a": _make_embedding([1.0])}
        result = rec.recognize([1.0], enrolled)
        assert result.accept_threshold == 0.80
        assert result.review_threshold == 0.60


# ---------------------------------------------------------------------------
# Fallback flow tests
# ---------------------------------------------------------------------------


class TestFallbackFlow:
    def test_recognized_sets_session(self):
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.RECOGNIZED,
            best_user_id="alice",
            score=0.95,
            accept_threshold=0.85,
            review_threshold=0.65,
        )
        with patch("rex.voice_identity.fallback_flow.set_session_user") as mock_set:
            user = resolve_speaker_identity(result)
        assert user == "alice"
        mock_set.assert_called_once_with("alice")

    def test_review_matches_session(self):
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.REVIEW,
            best_user_id="bob",
            score=0.70,
            accept_threshold=0.85,
            review_threshold=0.65,
        )
        with patch("rex.voice_identity.fallback_flow.resolve_active_user", return_value="bob"):
            user = resolve_speaker_identity(result)
        assert user == "bob"

    def test_review_mismatch_falls_back(self):
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.REVIEW,
            best_user_id="bob",
            score=0.70,
            accept_threshold=0.85,
            review_threshold=0.65,
        )
        with patch("rex.voice_identity.fallback_flow.resolve_active_user", return_value="alice"):
            user = resolve_speaker_identity(result)
        assert user == "alice"

    def test_unknown_uses_identity_chain(self):
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.UNKNOWN,
            score=0.20,
            accept_threshold=0.85,
            review_threshold=0.65,
        )
        with patch("rex.voice_identity.fallback_flow.resolve_active_user", return_value="james"):
            user = resolve_speaker_identity(result)
        assert user == "james"

    def test_unknown_no_user_returns_none(self):
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.UNKNOWN,
            score=0.10,
            accept_threshold=0.85,
            review_threshold=0.65,
        )
        with patch("rex.voice_identity.fallback_flow.resolve_active_user", return_value=None):
            user = resolve_speaker_identity(result)
        assert user is None


# ---------------------------------------------------------------------------
# Integration: store -> recognizer -> fallback
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    def test_enroll_recognize_fallback(self, tmp_path):
        """Full pipeline: enroll, recognize, resolve identity."""
        store = EmbeddingsStore(tmp_path)
        cfg = VoiceIdentityConfig(enabled=True, accept_threshold=0.85, review_threshold=0.65)
        recognizer = SpeakerRecognizer(cfg)

        # Enroll alice with a known vector
        alice_vec = [1.0, 0.0, 0.0, 0.0]
        store.save("alice", _make_embedding(alice_vec))

        # Recognise with the same vector
        enrolled = store.load_all()
        result = recognizer.recognize(alice_vec, enrolled)
        assert result.decision == RecognitionDecision.RECOGNIZED
        assert result.best_user_id == "alice"

        # Fallback sets session user
        with patch("rex.voice_identity.fallback_flow.set_session_user") as mock_set:
            user = resolve_speaker_identity(result)
        assert user == "alice"
        mock_set.assert_called_once_with("alice")

    def test_unknown_speaker_fallback(self, tmp_path):
        """Unknown speaker falls back to identity chain."""
        store = EmbeddingsStore(tmp_path)
        cfg = VoiceIdentityConfig(enabled=True, accept_threshold=0.85, review_threshold=0.65)
        recognizer = SpeakerRecognizer(cfg)

        # Enroll alice
        store.save("alice", _make_embedding([1.0, 0.0, 0.0]))

        # Try to recognize a completely different voice
        enrolled = store.load_all()
        result = recognizer.recognize([0.0, 0.0, 1.0], enrolled)
        assert result.decision == RecognitionDecision.UNKNOWN

        with patch(
            "rex.voice_identity.fallback_flow.resolve_active_user",
            return_value="james",
        ):
            user = resolve_speaker_identity(result)
        assert user == "james"
