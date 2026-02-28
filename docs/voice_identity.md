# Voice Identity

## Implementation Status: MVP (Phase 7.1)

The voice identity subsystem provides speaker recognition without introducing
heavy ML dependencies into the default install.  Phase 7.1 delivers a usable
MVP on top of the existing scaffolding: enrollment via WAV file, threshold
calibration, and runtime wiring into the voice loop.

> **Biometric disclaimer**: Voice identity provides convenience-level speaker
> recognition, not biometric-grade security.  Do not use it as an
> authentication gate for sensitive operations.

---

## Quick start

### 1. Enable voice identity in config

Edit `config/rex_config.json`:

```json
{
  "voice_identity": {
    "enabled": true,
    "accept_threshold": 0.85,
    "review_threshold": 0.65,
    "embedding_dim": 192,
    "model_id": "synthetic"
  }
}
```

> **Note**: The `"synthetic"` backend uses stdlib-only hashing and is intended
> for development and testing.  For production speaker separation, install the
> real backend (see [Installing optional heavy dependencies](#installing-optional-heavy-dependencies))
> and set `"model_id": "speechbrain"`.

### 2. Enroll a voice sample

```bash
rex voice-id enroll --user alice --wav /path/to/alice_sample.wav --yes
```

Repeat with additional samples to improve coverage.  Use `--replace` to wipe
and re-enroll a user.

### 3. Calibrate thresholds (recommended)

```bash
# Print recommended thresholds only
rex voice-id calibrate

# Print + write to config
rex voice-id calibrate --write-config --yes
```

### 4. Check status

```bash
rex voice-id status
rex voice-id list
```

---

## CLI commands

| Command | Description |
|---------|-------------|
| `rex voice-id status [--user <uid>]` | Show config and enrollment status |
| `rex voice-id list` | List all enrolled users |
| `rex voice-id enroll --user <uid> --wav <path> [--label <text>] [--replace] [--yes]` | Enroll a voice sample |
| `rex voice-id calibrate [--yes] [--write-config]` | Compute and optionally write recommended thresholds |

### `rex voice-id status`

Prints the current voice identity configuration (enabled state, model, thresholds)
and enrolled user list.

```
Voice Identity Status
==================================================
  Enabled          : yes
  Backend/model_id : synthetic
  Accept threshold : 0.85
  Review threshold : 0.65
  Embedding dim    : 192
  Enrolled users   : 2
    - alice  (samples=1, updated=2026-02-01)
    - bob    (samples=1, updated=2026-02-01)
  Real backend     : not installed (synthetic only)
```

### `rex voice-id enroll`

Reads a WAV file, generates a speaker embedding, and stores it in
`Memory/<user>/voice_embeddings.json`.

Requirements:
- `voice_identity.enabled` must be `true` in config.
- The WAV file must exist and be readable.
- If the user is already enrolled, use `--replace` to wipe the existing record.
- `--yes` is required to confirm the write.

If a real backend (`speechbrain`) is configured but the package is not
installed, enrollment will refuse with a clear error message:

```
Error: Cannot load embedding backend: speechbrain is required for model_id='speechbrain'.
Install the voice-id extras: pip install '.[voice-id]'
```

### `rex voice-id calibrate`

Analyses enrolled embeddings to recommend `accept_threshold` and
`review_threshold` values.

- **No users enrolled**: returns an error indicating enrollment is needed.
- **Single user**: returns conservative defaults with a note to enroll more users.
- **Multiple users**: computes pairwise inter-user cosine similarity and
  recommends thresholds that keep false-accept rates low.

Without `--write-config`, the report is printed only.  With `--write-config
--yes`, the recommended thresholds are written to `config/rex_config.json`.

---

## How runtime recognition works

1. The voice loop records audio after wake word detection.
2. If `voice_identity.enabled=true` and at least one user is enrolled, an
   `identify_speaker` callback is built at startup and wired into the voice loop.
3. The callback converts the recorded numpy audio to PCM bytes and passes them
   to the configured embedding backend.
4. The embedding is compared against all enrolled users via cosine similarity.
5. Based on the decision:
   - **recognized** (score ≥ accept_threshold): `set_session_user(best_user_id)`
     is called.  All downstream commands see the recognized user.
   - **review** (review_threshold ≤ score < accept_threshold): A log message
     is emitted (`"Voice uncertain (review): best_match=... Run 'rex identify'
     to set user manually."`).  The session user is **not** auto-set.
   - **unknown** (score < review_threshold): Nothing changes; the existing
     identity chain (`--user` flag, session state, config default) is used.

---

## Embedding backends

### Synthetic (default, stdlib-only)

```json
"model_id": "synthetic"
```

- Uses SHA-256 hashing to derive a deterministic unit vector from PCM bytes.
- No heavy dependencies; works in the base install.
- Not suitable for production speaker separation — different speakers may have
  similar synthetic embeddings because the hashing is based on audio content
  bytes, not acoustic speaker features.
- Useful for testing the enrollment/calibration/wiring pipeline without
  installing ML packages.

### SpeechBrain ECAPA-TDNN (optional)

```json
"model_id": "speechbrain"
```

- Uses SpeechBrain's ECAPA-TDNN model for speaker embedding extraction.
- Requires the voice-id extras: `pip install '.[voice-id]'`
- The model is loaded **lazily** on first use — no download or network access
  occurs during import or application startup.
- Provides genuine acoustic speaker discrimination.

To switch backends after enrolling with `synthetic`, you must re-enroll all
users with the new backend (embeddings are not cross-compatible between
backends).

---

## Config keys

Add to `config/rex_config.json` (template is in `config/rex_config.example.json`):

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
| `voice_identity.review_threshold` | float | `0.65` | Minimum cosine similarity for `review`; below is `unknown` |
| `voice_identity.embedding_dim` | int | `192` | Expected dimensionality of embedding vectors |
| `voice_identity.model_id` | string | `"synthetic"` | Backend identifier: `"synthetic"` or `"speechbrain"` |

---

## Embedding storage format

Each enrolled user has a file at `Memory/<user>/voice_embeddings.json`:

```json
{
  "model_id": "synthetic",
  "sample_count": 1,
  "updated_at": "2026-02-28T12:00:00+00:00",
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

The `embedding` array is the averaged unit-normalised speaker vector.  The
`model_id` must match the currently configured `voice_identity.model_id`.

---

## Installing optional heavy dependencies

The base install does not include speaker recognition ML libraries.  To
install them:

```bash
pip install ".[voice-id]"
```

This installs `speechbrain>=1.0.0` and `resemblyzer>=0.1.3`.  These are only
needed when a real embedding model is configured (not the default `synthetic`).

---

## Module layout

| Module | Purpose |
|--------|---------|
| `rex/voice_identity/__init__.py` | Package exports |
| `rex/voice_identity/types.py` | Core types: `VoiceEmbedding`, `RecognitionResult`, `VoiceIdentityConfig` |
| `rex/voice_identity/embeddings_store.py` | Per-user JSON embedding storage under `Memory/<user>/` |
| `rex/voice_identity/recognizer.py` | `SpeakerRecognizer` with pure-Python cosine similarity |
| `rex/voice_identity/fallback_flow.py` | `resolve_speaker_identity()` bridging into `rex.identity` |
| `rex/voice_identity/optional_deps.py` | Import guards + `get_embedding_backend()` factory |
| `rex/voice_identity/embedding_backends.py` | `SyntheticEmbeddingBackend` + `SpeechBrainBackend` |
| `rex/voice_identity/calibration.py` | `calibrate()` computing thresholds from enrolled embeddings |

---

## Running tests

All tests are offline (no network, no model downloads, no numpy required):

```bash
# Existing scaffolding tests
pytest -q tests/test_voice_identity_fallback.py
pytest -q tests/test_optional_voice_id_imports.py

# Phase 7.1 MVP tests
pytest -q tests/test_voice_id_mvp.py
```

---

## Known limitations and Phase 7.2+ work

- **Synthetic backend only** provides deterministic hashing, not acoustic
  speaker discrimination.  Production use requires the `speechbrain` backend.
- **One embedding per user**: The store averages across samples via the
  `sample_count` field, but the current enrollment only stores one embedding.
  Multi-sample averaging is deferred to Phase 7.2+.
- **No microphone enrollment**: Enrollment reads a pre-recorded WAV file.
  Interactive microphone capture is deferred to Phase 7.2+.
- **No voice PIN or interactive confirmation**: The `review` decision falls
  back to the existing identity chain.  Interactive re-confirmation is deferred.
- **No speaker diarization**: Multi-speaker conversations are not handled.
