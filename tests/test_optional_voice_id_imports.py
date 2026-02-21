"""Tests for optional voice-id dependency guards (BL-010).

These tests verify that the base install can import and use the voice
identity scaffolding without optional heavy dependencies (speechbrain,
resemblyzer) being installed.
"""

from __future__ import annotations


class TestBaseImports:
    """Verify that all voice_identity modules import without optional deps."""

    def test_import_types(self):
        from rex.voice_identity.types import (  # noqa: F401
            RecognitionDecision,
            RecognitionResult,
            VoiceEmbedding,
            VoiceIdentityConfig,
        )

    def test_import_embeddings_store(self):
        from rex.voice_identity.embeddings_store import EmbeddingsStore  # noqa: F401

    def test_import_recognizer(self):
        from rex.voice_identity.recognizer import SpeakerRecognizer  # noqa: F401

    def test_import_fallback_flow(self):
        from rex.voice_identity.fallback_flow import resolve_speaker_identity  # noqa: F401

    def test_import_optional_deps(self):
        from rex.voice_identity.optional_deps import (  # noqa: F401
            check_voice_id_available,
            import_resemblyzer,
            import_speechbrain,
        )

    def test_import_package(self):
        import rex.voice_identity  # noqa: F401


class TestOptionalDepsGuards:
    """Verify that missing optional deps are handled gracefully."""

    def test_check_voice_id_available_returns_bool(self):
        from rex.voice_identity.optional_deps import check_voice_id_available

        result = check_voice_id_available()
        assert isinstance(result, bool)

    def test_import_speechbrain_returns_none_if_missing(self):
        from rex.voice_identity.optional_deps import import_speechbrain

        # On a base install, speechbrain is not available
        result = import_speechbrain()
        # Result is None if not installed, or the module if it is
        assert result is None or hasattr(result, "__version__")

    def test_import_resemblyzer_returns_none_if_missing(self):
        from rex.voice_identity.optional_deps import import_resemblyzer

        result = import_resemblyzer()
        assert result is None or hasattr(result, "__version__")


class TestScaffoldingWorksWithoutOptionalDeps:
    """Verify core scaffolding runs without optional deps."""

    def test_recognizer_works(self):
        from rex.voice_identity.recognizer import SpeakerRecognizer
        from rex.voice_identity.types import (
            RecognitionDecision,
            VoiceEmbedding,
            VoiceIdentityConfig,
        )

        cfg = VoiceIdentityConfig(enabled=True)
        rec = SpeakerRecognizer(cfg)
        enrolled = {
            "alice": VoiceEmbedding(vector=[1.0, 0.0, 0.0]),
        }
        result = rec.recognize([1.0, 0.0, 0.0], enrolled)
        assert result.decision == RecognitionDecision.RECOGNIZED

    def test_embeddings_store_works(self, tmp_path):
        from rex.voice_identity.embeddings_store import EmbeddingsStore
        from rex.voice_identity.types import VoiceEmbedding

        store = EmbeddingsStore(tmp_path)
        store.save("user1", VoiceEmbedding(vector=[0.5, 0.5]))
        loaded = store.load("user1")
        assert loaded is not None
        assert loaded.vector == [0.5, 0.5]
