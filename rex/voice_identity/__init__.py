"""Voice identity scaffolding for speaker recognition.

This package provides the architecture for future speaker recognition
without requiring heavy ML dependencies in the default install.  The
current implementation uses pure-Python cosine similarity on embedding
vectors and stores per-user embeddings as JSON under Memory/<user>/.

Heavy dependencies (speechbrain, resemblyzer, torch-audio) are only
needed when a real embedding model is plugged in.  The base install
operates with synthetic/precomputed embeddings for testing and
development.
"""

from rex.voice_identity.types import RecognitionDecision, RecognitionResult, VoiceEmbedding

__all__ = [
    "RecognitionDecision",
    "RecognitionResult",
    "VoiceEmbedding",
]
