"""Tests for Phase 7.1 Voice Identity MVP.

All tests are fully offline:
- No network access
- No numpy, speechbrain, sounddevice, or resemblyzer required
- All WAV files are created with stdlib ``wave`` module
- All embeddings use the synthetic backend
- Config writes go to ``tmp_path`` only (never ``config/rex_config.json``)
"""

from __future__ import annotations

import json
import struct
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.voice_identity.calibration import calibrate
from rex.voice_identity.embedding_backends import (
    SpeechBrainBackend,
    SyntheticEmbeddingBackend,
    _bytes_to_unit_vector,
)
from rex.voice_identity.embeddings_store import EmbeddingsStore
from rex.voice_identity.optional_deps import get_embedding_backend
from rex.voice_identity.recognizer import SpeakerRecognizer
from rex.voice_identity.types import (
    RecognitionDecision,
    VoiceEmbedding,
    VoiceIdentityConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(num_samples: int = 1600, sample_value: int = 1000) -> bytes:
    """Create a minimal WAV file (int16 mono, 16kHz) in memory.

    Args:
        num_samples: Number of PCM samples.
        sample_value: Value for each sample (determines hash result).

    Returns:
        WAV file as bytes.
    """
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        samples = struct.pack(f"<{num_samples}h", *([sample_value] * num_samples))
        wf.writeframes(samples)
    return buf.getvalue()


def _write_wav_file(path: Path, num_samples: int = 1600, sample_value: int = 1000) -> Path:
    """Write a WAV file to disk and return its path."""
    data = _make_wav_bytes(num_samples, sample_value)
    path.write_bytes(data)
    return path


def _read_wav_pcm(path: Path) -> bytes:
    """Read raw PCM bytes from a WAV file on disk."""
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _make_embedding(vector: list[float], model_id: str = "synthetic") -> VoiceEmbedding:
    return VoiceEmbedding(
        vector=vector,
        model_id=model_id,
        sample_count=1,
        updated_at="2026-01-01T00:00:00Z",
    )


def _unit_vector(dim: int, index: int) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


# ---------------------------------------------------------------------------
# A. Embedding backend tests
# ---------------------------------------------------------------------------


class TestBytesToUnitVector:
    def test_output_has_correct_dim(self):
        v = _bytes_to_unit_vector(b"test", 64)
        assert len(v) == 64

    def test_is_unit_normalised(self):
        import math

        v = _bytes_to_unit_vector(b"hello world", 32)
        mag = math.sqrt(sum(x * x for x in v))
        assert abs(mag - 1.0) < 1e-9, f"Expected unit vector, got magnitude {mag}"

    def test_deterministic(self):
        v1 = _bytes_to_unit_vector(b"same bytes", 16)
        v2 = _bytes_to_unit_vector(b"same bytes", 16)
        assert v1 == v2

    def test_different_inputs_differ(self):
        v1 = _bytes_to_unit_vector(b"input_a", 16)
        v2 = _bytes_to_unit_vector(b"input_b", 16)
        assert v1 != v2

    def test_zero_bytes_does_not_crash(self):
        # All-zero bytes produce a valid non-zero vector (hash is never all-zero)
        v = _bytes_to_unit_vector(b"\x00" * 32, 8)
        assert len(v) == 8


class TestSyntheticEmbeddingBackend:
    def test_model_id(self):
        backend = SyntheticEmbeddingBackend(dim=16)
        assert backend.model_id == "synthetic"

    def test_embed_returns_correct_dim(self):
        backend = SyntheticEmbeddingBackend(dim=32)
        vector = backend.embed(b"some audio bytes")
        assert len(vector) == 32

    def test_embed_is_deterministic(self):
        backend = SyntheticEmbeddingBackend(dim=16)
        v1 = backend.embed(b"hello")
        v2 = backend.embed(b"hello")
        assert v1 == v2

    def test_embed_differs_for_different_audio(self):
        backend = SyntheticEmbeddingBackend(dim=16)
        v1 = backend.embed(b"speaker_a_audio")
        v2 = backend.embed(b"speaker_b_audio")
        assert v1 != v2

    def test_embed_ignores_sample_rate(self):
        backend = SyntheticEmbeddingBackend(dim=16)
        v1 = backend.embed(b"data", sample_rate=8000)
        v2 = backend.embed(b"data", sample_rate=16000)
        assert v1 == v2

    def test_wav_pcm_produces_embedding(self, tmp_path):
        wav = _write_wav_file(tmp_path / "test.wav", sample_value=500)
        pcm = _read_wav_pcm(wav)
        backend = SyntheticEmbeddingBackend(dim=192)
        v = backend.embed(pcm)
        assert len(v) == 192


class TestSpeechBrainBackendGuard:
    """Ensure SpeechBrainBackend fails gracefully when deps are missing."""

    def test_model_id_property_without_load(self):
        backend = SpeechBrainBackend()
        assert backend.model_id == "speechbrain"

    def test_embed_raises_import_error_when_speechbrain_missing(self):
        backend = SpeechBrainBackend()
        # Patch find_spec to simulate speechbrain not installed
        with patch("rex.voice_identity.embedding_backends.SpeechBrainBackend._load_model") as mock:
            mock.side_effect = ImportError(
                "speechbrain not installed. Install: pip install '.[voice-id]'"
            )
            with pytest.raises(ImportError, match="voice-id"):
                backend.embed(b"audio")

    def test_init_has_no_side_effects(self):
        # Constructing the backend should not import speechbrain or load any model
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.return_value = None  # simulate nothing installed
            # Should not raise
            backend = SpeechBrainBackend()
            assert backend._model is None


class TestGetEmbeddingBackend:
    def test_synthetic_backend_returned(self):
        backend = get_embedding_backend("synthetic", dim=16)
        assert isinstance(backend, SyntheticEmbeddingBackend)
        assert backend.model_id == "synthetic"

    def test_unknown_model_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_embedding_backend("not_a_real_model")

    def test_speechbrain_raises_import_error_when_missing(self):
        from importlib.util import find_spec

        # Only test the error path if speechbrain is genuinely not installed
        if find_spec("speechbrain") is None:
            with pytest.raises(ImportError, match="voice-id"):
                get_embedding_backend("speechbrain")
        else:
            # If installed, it should return a SpeechBrainBackend without error
            backend = get_embedding_backend("speechbrain")
            assert backend.model_id == "speechbrain"


# ---------------------------------------------------------------------------
# B. Enrollment tests
# ---------------------------------------------------------------------------


class TestEnrollment:
    def test_enroll_stores_embedding(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        backend = SyntheticEmbeddingBackend(dim=8)

        wav = _write_wav_file(tmp_path / "alice.wav", sample_value=100)
        pcm = _read_wav_pcm(wav)
        vector = backend.embed(pcm)

        emb = VoiceEmbedding(vector=vector, model_id="synthetic", sample_count=1)
        store.save("alice", emb)

        loaded = store.load("alice")
        assert loaded is not None
        assert loaded.vector == vector
        assert loaded.model_id == "synthetic"

    def test_enroll_replace_mode_resets(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        backend = SyntheticEmbeddingBackend(dim=8)

        # Initial enrollment
        wav1 = _write_wav_file(tmp_path / "wav1.wav", sample_value=111)
        v1 = backend.embed(_read_wav_pcm(wav1))
        store.save("bob", VoiceEmbedding(vector=v1, model_id="synthetic", sample_count=1))
        assert store.load("bob").vector == v1

        # Replace enrollment
        wav2 = _write_wav_file(tmp_path / "wav2.wav", sample_value=222)
        v2 = backend.embed(_read_wav_pcm(wav2))
        store.save("bob", VoiceEmbedding(vector=v2, model_id="synthetic", sample_count=1))

        loaded = store.load("bob")
        assert loaded.vector == v2
        assert loaded.vector != v1

    def test_multiple_users_enrolled(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        backend = SyntheticEmbeddingBackend(dim=16)

        for name, val in [("alice", 1), ("bob", 2), ("carol", 3)]:
            wav = _write_wav_file(tmp_path / f"{name}.wav", sample_value=val * 100)
            v = backend.embed(_read_wav_pcm(wav))
            store.save(name, VoiceEmbedding(vector=v, model_id="synthetic", sample_count=1))

        enrolled = store.list_enrolled_users()
        assert enrolled == ["alice", "bob", "carol"]

    def test_enroll_creates_directory(self, tmp_path):
        base = tmp_path / "memory_root"
        # directory does not exist yet
        assert not base.exists()
        store = EmbeddingsStore(base)
        v = SyntheticEmbeddingBackend(dim=4).embed(b"x")
        store.save("dave", VoiceEmbedding(vector=v, model_id="synthetic", sample_count=1))
        assert (base / "dave" / "voice_embeddings.json").exists()

    def test_store_rejects_path_traversal_user_id(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        vector = SyntheticEmbeddingBackend(dim=4).embed(b"x")

        with pytest.raises(ValueError, match="simple name"):
            store.save(
                "../evil", VoiceEmbedding(vector=vector, model_id="synthetic", sample_count=1)
            )

        assert not (tmp_path / ".." / "evil" / "voice_embeddings.json").exists()


# ---------------------------------------------------------------------------
# C. Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def _cfg(self) -> VoiceIdentityConfig:
        return VoiceIdentityConfig(
            enabled=True,
            accept_threshold=0.85,
            review_threshold=0.65,
            embedding_dim=8,
            model_id="synthetic",
        )

    def test_no_enrolled_users(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        report = calibrate(store, self._cfg())
        assert report.enrolled_users == []
        assert "No users enrolled" in report.notes[0]
        # Fallback to config defaults
        assert report.recommended_accept_threshold == 0.85
        assert report.recommended_review_threshold == 0.65

    def test_single_enrolled_user(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        store.save("alice", _make_embedding(_unit_vector(8, 0)))
        report = calibrate(store, self._cfg())
        assert report.enrolled_users == ["alice"]
        assert "Only one user" in report.notes[0]
        # Conservative defaults
        assert report.recommended_accept_threshold == 0.85
        assert report.recommended_review_threshold == 0.65

    def test_two_users_threshold_ordering(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        # Orthogonal vectors → minimum inter-user similarity ≈ 0
        store.save("alice", _make_embedding(_unit_vector(4, 0)))
        store.save("bob", _make_embedding(_unit_vector(4, 1)))

        report = calibrate(store, self._cfg())
        assert report.enrolled_users == ["alice", "bob"]
        assert report.min_inter_user_similarity is not None
        # accept > review (key ordering requirement)
        assert report.recommended_accept_threshold > report.recommended_review_threshold
        # Both thresholds in sensible range
        assert 0.0 <= report.recommended_review_threshold <= 1.0
        assert 0.0 <= report.recommended_accept_threshold <= 1.0

    def test_three_users_threshold_ordering(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        store.save("alice", _make_embedding(_unit_vector(4, 0)))
        store.save("bob", _make_embedding(_unit_vector(4, 1)))
        store.save("carol", _make_embedding(_unit_vector(4, 2)))

        report = calibrate(store, self._cfg())
        assert len(report.enrolled_users) == 3
        assert report.recommended_accept_threshold > report.recommended_review_threshold

    def test_high_similarity_warning(self, tmp_path):
        """Users with very similar vectors should trigger a warning note."""
        import math

        store = EmbeddingsStore(tmp_path)
        # Vectors very close to each other → high inter-user similarity
        v1 = [1.0, 0.001, 0.0, 0.0]
        mag1 = math.sqrt(sum(x * x for x in v1))
        v1 = [x / mag1 for x in v1]

        v2 = [1.0, 0.002, 0.0, 0.0]
        mag2 = math.sqrt(sum(x * x for x in v2))
        v2 = [x / mag2 for x in v2]

        store.save("twin_a", _make_embedding(v1))
        store.save("twin_b", _make_embedding(v2))

        report = calibrate(store, self._cfg())
        assert any("WARNING" in note for note in report.notes)

    def test_write_config_updates_thresholds(self, tmp_path):
        """--write-config path writes only to temp config, not real config."""
        # Set up temp config
        temp_config = tmp_path / "rex_config.json"
        initial_config = {
            "voice_identity": {
                "enabled": True,
                "accept_threshold": 0.85,
                "review_threshold": 0.65,
                "embedding_dim": 8,
                "model_id": "synthetic",
            }
        }
        temp_config.write_text(json.dumps(initial_config, indent=2))

        # Set up store with two orthogonal users
        store = EmbeddingsStore(tmp_path / "memory")
        store.save("u1", _make_embedding(_unit_vector(4, 0)))
        store.save("u2", _make_embedding(_unit_vector(4, 1)))

        report = calibrate(store, self._cfg())

        # Simulate writing to temp config (as the CLI --write-config does)
        data = json.loads(temp_config.read_text())
        vi = data.get("voice_identity", {})
        vi["accept_threshold"] = report.recommended_accept_threshold
        vi["review_threshold"] = report.recommended_review_threshold
        data["voice_identity"] = vi
        temp_config.write_text(json.dumps(data, indent=2))

        # Read back and verify
        result = json.loads(temp_config.read_text())
        assert result["voice_identity"]["accept_threshold"] == report.recommended_accept_threshold
        assert result["voice_identity"]["review_threshold"] == report.recommended_review_threshold
        # accept > review after write
        assert (
            result["voice_identity"]["accept_threshold"]
            > result["voice_identity"]["review_threshold"]
        )

    def test_corrupted_store_data_falls_back_to_config_defaults(self, tmp_path):
        store = EmbeddingsStore(tmp_path)
        bad_file = tmp_path / "alice" / "voice_embeddings.json"
        bad_file.parent.mkdir(parents=True)
        bad_file.write_text("{not-valid-json", encoding="utf-8")

        report = calibrate(store, self._cfg())

        assert report.enrolled_users == []
        assert report.recommended_accept_threshold == 0.85
        assert report.recommended_review_threshold == 0.65
        assert any("No users enrolled" in note for note in report.notes)


# ---------------------------------------------------------------------------
# D. Runtime wiring tests
# ---------------------------------------------------------------------------


class TestRuntimeWiring:
    """Tests for voice identity wiring into the recognition/session flow."""

    def _cfg(self, accept: float = 0.85, review: float = 0.65) -> VoiceIdentityConfig:
        return VoiceIdentityConfig(
            enabled=True,
            accept_threshold=accept,
            review_threshold=review,
            embedding_dim=4,
            model_id="synthetic",
        )

    def test_recognized_sets_session_user(self, tmp_path):
        """High-confidence recognition should set the active session user."""
        from rex.voice_identity.fallback_flow import resolve_speaker_identity
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.RECOGNIZED,
            best_user_id="alice",
            score=0.95,
            accept_threshold=0.85,
            review_threshold=0.65,
        )

        with patch("rex.voice_identity.fallback_flow.set_session_user") as mock_set:
            resolved = resolve_speaker_identity(result)

        mock_set.assert_called_once_with("alice")
        assert resolved == "alice"

    def test_uncertain_does_not_auto_set_user(self):
        """REVIEW decision should NOT auto-set the session user."""
        from rex.voice_identity.fallback_flow import resolve_speaker_identity
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.REVIEW,
            best_user_id="bob",
            score=0.72,
            accept_threshold=0.85,
            review_threshold=0.65,
        )

        with (
            patch("rex.voice_identity.fallback_flow.set_session_user") as mock_set,
            patch("rex.voice_identity.fallback_flow.resolve_active_user") as mock_resolve,
        ):
            mock_resolve.return_value = None
            resolve_speaker_identity(result)

        mock_set.assert_not_called()

    def test_unknown_does_not_set_user(self):
        """UNKNOWN decision should not call set_session_user."""
        from rex.voice_identity.fallback_flow import resolve_speaker_identity
        from rex.voice_identity.types import RecognitionResult

        result = RecognitionResult(
            decision=RecognitionDecision.UNKNOWN,
            best_user_id=None,
            score=0.30,
            accept_threshold=0.85,
            review_threshold=0.65,
        )

        with (
            patch("rex.voice_identity.fallback_flow.set_session_user") as mock_set,
            patch("rex.voice_identity.fallback_flow.resolve_active_user") as mock_resolve,
        ):
            mock_resolve.return_value = None
            resolve_speaker_identity(result)

        mock_set.assert_not_called()

    def test_recognition_pipeline_with_synthetic_backend(self, tmp_path):
        """End-to-end: enroll → recognise → check decision."""
        store = EmbeddingsStore(tmp_path)
        backend = SyntheticEmbeddingBackend(dim=8)

        # Enroll alice with a known audio pattern
        alice_pcm = b"alice_voice_pattern_deterministic"
        alice_vec = backend.embed(alice_pcm)
        store.save("alice", VoiceEmbedding(vector=alice_vec, model_id="synthetic", sample_count=1))

        enrolled = store.load_all()
        cfg = self._cfg()
        recognizer = SpeakerRecognizer(cfg)

        # Recognise alice with the same audio → should be RECOGNIZED (score=1.0)
        result = recognizer.recognize(alice_vec, enrolled)
        assert result.decision == RecognitionDecision.RECOGNIZED
        assert result.best_user_id == "alice"
        assert result.score > 0.85

    def test_different_speaker_is_unknown(self, tmp_path):
        """A different speaker vector should not match the enrolled user."""
        store = EmbeddingsStore(tmp_path)
        backend = SyntheticEmbeddingBackend(dim=16)

        alice_vec = backend.embed(b"alice_audio")
        store.save("alice", VoiceEmbedding(vector=alice_vec, model_id="synthetic", sample_count=1))

        enrolled = store.load_all()
        cfg = VoiceIdentityConfig(
            enabled=True,
            accept_threshold=0.95,  # very strict
            review_threshold=0.85,
            embedding_dim=16,
            model_id="synthetic",
        )
        recognizer = SpeakerRecognizer(cfg)

        # Use a very different audio pattern
        bob_vec = backend.embed(b"completely_different_bob_audio_xyz")
        result = recognizer.recognize(bob_vec, enrolled)
        # With strict thresholds and different audio → should not be RECOGNIZED
        assert result.decision != RecognitionDecision.RECOGNIZED


# ---------------------------------------------------------------------------
# E. Optional deps guard tests
# ---------------------------------------------------------------------------


class TestOptionalDepsGuards:
    """Ensure voice identity modules import cleanly without optional deps."""

    def test_embedding_backends_imports_without_heavy_deps(self):
        """Importing embedding_backends should not require numpy/speechbrain."""
        # The module is already imported at the top of this test file,
        # so we just verify the classes are accessible.
        assert SyntheticEmbeddingBackend is not None
        assert SpeechBrainBackend is not None

    def test_calibration_imports_without_heavy_deps(self):
        from rex.voice_identity.calibration import CalibrationReport as CR
        from rex.voice_identity.calibration import calibrate as cal

        assert cal is not None
        assert CR is not None

    def test_optional_deps_module_imports(self):
        from rex.voice_identity.optional_deps import (
            check_voice_id_available,
            get_embedding_backend,
            import_resemblyzer,
            import_speechbrain,
        )

        assert callable(check_voice_id_available)
        assert callable(get_embedding_backend)
        assert callable(import_speechbrain)
        assert callable(import_resemblyzer)

    def test_all_voice_identity_submodules_import(self):
        """All submodules should be importable without heavy deps."""
        import rex.voice_identity.calibration  # noqa: F401
        import rex.voice_identity.embedding_backends  # noqa: F401
        import rex.voice_identity.embeddings_store  # noqa: F401
        import rex.voice_identity.fallback_flow  # noqa: F401
        import rex.voice_identity.optional_deps  # noqa: F401
        import rex.voice_identity.recognizer  # noqa: F401
        import rex.voice_identity.types  # noqa: F401

    def test_selecting_missing_real_backend_fails_with_clear_message(self):
        """Requesting a real backend when deps are missing gives a clear error."""
        from importlib.util import find_spec

        if find_spec("speechbrain") is not None:
            pytest.skip("speechbrain is installed — cannot test missing-dep path")

        with pytest.raises(ImportError) as exc_info:
            get_embedding_backend("speechbrain")

        # Error message should mention install hint
        assert "voice-id" in str(exc_info.value)

    def test_unknown_model_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown voice identity model_id"):
            get_embedding_backend("unicorn_model")


# ---------------------------------------------------------------------------
# F. CLI smoke tests (offline, no real config writes)
# ---------------------------------------------------------------------------


class TestVoiceIdCLI:
    """Smoke-tests for the voice-id CLI command handler."""

    def _make_config(self, enabled: bool = True, model_id: str = "synthetic") -> dict:
        return {
            "voice_identity": {
                "enabled": enabled,
                "accept_threshold": 0.85,
                "review_threshold": 0.65,
                "embedding_dim": 8,
                "model_id": model_id,
            }
        }

    def _run_cmd(self, argv: list[str]) -> int:
        """Run the CLI and return the exit code."""
        from rex.cli import main

        return main(argv)

    def test_help_does_not_crash(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            self._run_cmd(["voice-id", "--help"])
        assert exc_info.value.code == 0

    def test_status_with_voice_id_disabled(self, tmp_path, capsys):
        """Status command should print disabled status without error."""
        import argparse

        from rex.cli import cmd_voice_id

        args = argparse.Namespace(voice_id_command="status", user=None)

        with (
            patch("rex.config_manager.load_config") as mock_cfg,
            patch("rex.voice_identity.embeddings_store.EmbeddingsStore") as mock_store_cls,
        ):
            mock_cfg.return_value = {
                "voice_identity": {
                    "enabled": False,
                    "accept_threshold": 0.85,
                    "review_threshold": 0.65,
                    "embedding_dim": 192,
                    "model_id": "synthetic",
                }
            }
            mock_store = MagicMock()
            mock_store.list_enrolled_users.return_value = []
            mock_store_cls.return_value = mock_store

            result = cmd_voice_id(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "no" in captured.out.lower() or "disabled" in captured.out.lower()

    def test_enroll_refuses_when_disabled(self, tmp_path, capsys):
        """Enroll should refuse if voice_identity.enabled is false."""
        import argparse

        from rex.cli import cmd_voice_id

        # Create a dummy WAV file
        wav_path = tmp_path / "test.wav"
        _write_wav_file(wav_path)

        args = argparse.Namespace(
            voice_id_command="enroll",
            user="testuser",
            wav=str(wav_path),
            label=None,
            replace=False,
            yes=True,
        )

        with patch("rex.config_manager.load_config") as mock_cfg:
            mock_cfg.return_value = {"voice_identity": {"enabled": False}}
            result = cmd_voice_id(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()

    def test_enroll_success(self, tmp_path, capsys):
        """Enroll should succeed when voice identity is enabled."""
        import argparse

        from rex.cli import cmd_voice_id

        wav_path = tmp_path / "enroll.wav"
        _write_wav_file(wav_path, sample_value=42)

        args = argparse.Namespace(
            voice_id_command="enroll",
            user="testuser",
            wav=str(wav_path),
            label="test label",
            replace=False,
            yes=True,
        )

        # Patch config and store paths to use tmp_path
        memory_dir = tmp_path / "Memory"
        memory_dir.mkdir()

        with (
            patch("rex.config_manager.load_config") as mock_cfg,
            patch("rex.cli.Path") as mock_path_cls,
        ):
            mock_cfg.return_value = {
                "voice_identity": {
                    "enabled": True,
                    "accept_threshold": 0.85,
                    "review_threshold": 0.65,
                    "embedding_dim": 8,
                    "model_id": "synthetic",
                }
            }
            # Use the real Path class but patch __file__ resolution
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw)

            # Directly patch EmbeddingsStore to use tmp_path
            with patch("rex.voice_identity.embeddings_store.EmbeddingsStore") as mock_store_cls:
                mock_store = MagicMock()
                mock_store.load.return_value = None  # No existing enrollment
                mock_store_cls.return_value = mock_store

                result = cmd_voice_id(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Enrolled" in captured.out

    def test_list_no_enrolled_users(self, tmp_path, capsys):
        """List should report no enrolled users gracefully."""
        import argparse

        from rex.cli import cmd_voice_id

        args = argparse.Namespace(voice_id_command="list")

        with patch("rex.voice_identity.embeddings_store.EmbeddingsStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.list_enrolled_users.return_value = []
            mock_store_cls.return_value = mock_store

            result = cmd_voice_id(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No users enrolled" in captured.out

    def test_calibrate_no_users(self, tmp_path, capsys):
        """Calibrate with no enrolled users should produce a report."""
        import argparse

        from rex.cli import cmd_voice_id

        args = argparse.Namespace(
            voice_id_command="calibrate",
            user=None,
            yes=False,
            write_config=False,
        )

        with (
            patch("rex.config_manager.load_config") as mock_cfg,
            patch("rex.voice_identity.embeddings_store.EmbeddingsStore") as mock_store_cls,
        ):
            mock_cfg.return_value = {
                "voice_identity": {
                    "enabled": True,
                    "accept_threshold": 0.85,
                    "review_threshold": 0.65,
                    "embedding_dim": 8,
                    "model_id": "synthetic",
                }
            }
            mock_store = MagicMock()
            mock_store.load_all.return_value = {}
            mock_store_cls.return_value = mock_store

            result = cmd_voice_id(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Calibration Report" in captured.out

    def test_calibrate_write_config_requires_yes(self, tmp_path, capsys):
        """--write-config without --yes should refuse and print a message."""
        import argparse

        from rex.cli import cmd_voice_id

        args = argparse.Namespace(
            voice_id_command="calibrate",
            user=None,
            yes=False,
            write_config=True,
        )

        with (
            patch("rex.config_manager.load_config") as mock_cfg,
            patch("rex.voice_identity.embeddings_store.EmbeddingsStore") as mock_store_cls,
        ):
            mock_cfg.return_value = {
                "voice_identity": {
                    "enabled": True,
                    "accept_threshold": 0.85,
                    "review_threshold": 0.65,
                    "embedding_dim": 8,
                    "model_id": "synthetic",
                }
            }
            mock_store = MagicMock()
            mock_store.load_all.return_value = {}
            mock_store_cls.return_value = mock_store

            result = cmd_voice_id(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "--yes" in captured.out
