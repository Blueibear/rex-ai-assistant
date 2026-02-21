# Voice Identity

## Implementation Status: Scaffolding (not production-ready)

The voice identity subsystem provides the architecture for speaker recognition without introducing heavy ML dependencies into the default install.

## What exists now

- **Pure-Python cosine similarity** for comparing speaker embedding vectors (no numpy/torch required).
- **Per-user embeddings store** at `Memory/<user>/voice_embeddings.json` with versioned metadata (model ID, sample count, timestamps).
- **Speaker recognizer** that returns a decision (`recognized`, `review`, `unknown`) with configurable thresholds.
- **Fallback identity flow** that integrates with the existing session-scoped identity system (`rex.identity`) when recognition is uncertain.
- **Voice loop hook** (`identify_speaker` callback) that optionally consults voice identity before processing speech.

## What is NOT implemented yet

- **Audio capture and real embedding extraction.** The scaffolding works with precomputed embedding vectors. A real embedding model (e.g., speechbrain, resemblyzer) must be plugged in to produce vectors from audio.
- **Enrollment via microphone.** Enrollment is currently "store precomputed vectors." A future PR will add `rex enroll --user <id>` with audio recording.
- **Voice PIN or interactive confirmation.** The `review` decision currently falls back to the existing identity chain. A future PR may add interactive confirmation prompts.
- **Speaker diarization.** Multi-speaker conversations are not handled.

## Architecture

```
voice_loop.py
    |
    v
identify_speaker() callback (no-op if voice identity disabled)
    |
    v
rex/voice_identity/recognizer.py
    |-- cosine similarity (pure Python)
    |-- thresholds: accept_threshold, review_threshold
    |
    v
rex/voice_identity/fallback_flow.py
    |-- recognized -> set_session_user()
    |-- review -> check existing identity, fall back
    |-- unknown -> use existing identity chain
    |
    v
rex/identity.py (existing)
    |-- --user flag > session state > config
```

## Module layout

| Module | Purpose |
|--------|---------|
| `rex/voice_identity/__init__.py` | Package exports |
| `rex/voice_identity/types.py` | Data types: `VoiceEmbedding`, `RecognitionResult`, `VoiceIdentityConfig` |
| `rex/voice_identity/embeddings_store.py` | Per-user JSON embedding storage under `Memory/<user>/` |
| `rex/voice_identity/recognizer.py` | `SpeakerRecognizer` with pure-Python cosine similarity |
| `rex/voice_identity/fallback_flow.py` | `resolve_speaker_identity()` bridging into `rex.identity` |

## Config keys

Add to `config/rex_config.json`:

```json
{
  "voice_identity": {
    "enabled": false,
    "accept_threshold": 0.85,
    "review_threshold": 0.65,
    "embedding_dim": 192,
    "model_id": "synthetic"
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice_identity.enabled` | bool | `false` | Enable voice identity in the voice loop |
| `voice_identity.accept_threshold` | float | `0.85` | Minimum cosine similarity for `recognized` |
| `voice_identity.review_threshold` | float | `0.65` | Minimum cosine similarity for `review`; below this is `unknown` |
| `voice_identity.embedding_dim` | int | `192` | Expected dimensionality of embedding vectors |
| `voice_identity.model_id` | string | `"synthetic"` | Identifier for the active embedding model |

## Embedding storage format

Each enrolled user has a file at `Memory/<user>/voice_embeddings.json`:

```json
{
  "model_id": "synthetic",
  "sample_count": 5,
  "updated_at": "2026-02-21T12:00:00+00:00",
  "embedding": [0.1, 0.2, 0.3, ...]
}
```

## How fallback identity is triggered

1. The voice loop records audio after wake word detection.
2. If `voice_identity.enabled` is `true` and an `identify_speaker` callback is provided, it runs before transcription.
3. The callback computes (or retrieves) an embedding for the audio and calls `SpeakerRecognizer.recognize()`.
4. Based on the decision:
   - **recognized**: `set_session_user(best_user_id)` is called, and all downstream commands see the recognized user.
   - **review**: If the existing session user matches the best guess, it is accepted silently. Otherwise, the existing identity chain (`--user` flag, session state, config) is used.
   - **unknown**: The existing identity chain is used without modification.

## Installing optional heavy dependencies

The base install does not include speaker recognition ML libraries. To install them:

```bash
pip install ".[voice-id]"
```

This installs speechbrain and resemblyzer. These are only needed when a real embedding model is configured (not the default `synthetic` model).

## Running tests

```bash
pytest -q tests/test_voice_identity_fallback.py
pytest -q tests/test_optional_voice_id_imports.py
```
