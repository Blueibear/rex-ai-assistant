# AGENTS.md — Reusable Codebase Patterns

This file documents patterns, conventions, and findings that autonomous agents (and developers)
should know before making changes. Add entries here whenever a non-obvious pattern is discovered
or when a diagnostic finding should guide future fix work.

---

## Voice Pipeline Break Point

**Investigation:** US-134 (Phase 48)

**Test procedure:**

```
python scripts/test_voice_pipeline.py
```

The script mocks all heavy dependencies (Coqui XTTS, Whisper, audio devices) and runs
`AsyncRexAssistant._process_conversation()` end-to-end with `LOG_LEVEL=DEBUG`. It then
scans the captured log for the six `[PIPELINE] stage=` entries added in US-133 and
reports which stage is the last to complete and which is the first missing.

**Result with mocks (all 6 stages complete):**

```
stage=llm_response_received       ✓  (voice_loop.py::_process_conversation ~line 555)
stage=tts_input_prepared          ✓  (voice_loop.py::_speak_response, after chunk_text_for_xtts)
stage=tts_engine_called           ✓  (voice_loop.py::_speak_response, before each tts.tts_to_file call)
stage=audio_data_returned         ✓  (voice_loop.py::_speak_response, after soundfile.read per chunk)
stage=audio_playback_initiated    ✓  (voice_loop.py::_speak_response, before winsound/simpleaudio)
stage=audio_playback_completed    ✓  (voice_loop.py::_speak_response, after playback)
```

**Confirmed break point in a real (non-mock) environment:**

When Coqui XTTS is not installed (the common case on a fresh install without the GPU/CPU extras):

- **Last successful stage:** `llm_response_received`
- **First missing stage:** `tts_input_prepared`
- **Root cause:** `_speak_response` calls `self._get_tts()` before chunking text. `_get_tts()`
  raises `TextToSpeechError("TTS is not installed")` when `TTS.api` cannot be imported.
  This exception is caught in `_process_conversation` and logged as an error, but the pipeline
  continues silently — no audio is produced and no further `[PIPELINE]` stages are logged.
- **Relevant code path:**
  - `voice_loop.py` — `AsyncRexAssistant._speak_response` (function starts ~line 630)
  - `voice_loop.py` — `AsyncRexAssistant._get_tts` (function starts ~line 375):
    calls `_lazy_import_tts()` → returns `None` when `TTS.api` not importable → raises
    `TextToSpeechError("TTS is not installed")`
  - `voice_loop.py` — `AsyncRexAssistant._process_conversation` (~line 570):
    `except TextToSpeechError as exc: logger.error("TTS failed: %s", exc)` — error is swallowed

**Secondary break point (TTS installed, no audio output library):**

On non-Windows without `simpleaudio` installed, all 6 stages appear in the log but no audio is
played. The pipeline logs `[PIPELINE] stage=audio_playback_completed` but `_speak_response`
takes the `else` branch and logs a warning instead of playing audio:

```python
else:
    logger.warning("No audio playback library available - audio saved but not played")
```

Fix stories US-135 and US-136 should address:
1. US-135: ensure TTS text is reliably delivered (guard the `TextToSpeechError` catch to re-raise
   or surface the error more visibly rather than silently swallowing it)
2. US-136: ensure the correct audio playback library is available and selected per platform

---

## This codebase uses asyncio.to_thread for blocking I/O in async handlers

All synchronous blocking operations (file I/O, LLM calls, TTS synthesis, audio recording) in
`voice_loop.py` are wrapped in `asyncio.to_thread(fn, *args)` so the event loop is not blocked.
When adding new operations to `_process_conversation` or `_handle_interaction`, follow this pattern.

## Config split: secrets in .env, runtime settings in config/rex_config.json

Never read secrets from `config/rex_config.json`. Secrets belong in `.env` only.
`AppConfig` (Pydantic v2) loads runtime settings; `python-dotenv` loads secrets at startup.

## Empty env var pattern

`os.getenv("REX_FOO", "default")` returns `""` when `.env` contains `REX_FOO=` (blank value).
Use `os.getenv("REX_FOO") or "default"` to treat empty-string as missing.

## Conventional Commits enforcement

A `commit-msg` hook lives at `.githooks/commit-msg` (version-controlled source).
Install it on a fresh clone with:

```bash
cp .githooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

The hook rejects any message that does not match:
`^(feat|fix|test|docs|refactor|chore|perf|ci)(\(.+\))?: .+`

See `CONTRIBUTING.md` for full details and examples.

---

## Optional dependency imports

Use `_import_optional(module_name)` (defined in `voice_loop.py`) or `find_spec` guards before
importing optional packages. Heavy optional deps (TTS, Whisper, simpleaudio, sounddevice) must
not be imported at module level - they are lazy-loaded to avoid import errors on minimal installs.

## Sync token streams need an async bridge

When a provider exposes a synchronous token iterator but the caller is async (for example the
voice loop), bridge it with `asyncio.to_thread(...)` plus an `asyncio.Queue` so token streaming
does not block the event loop.

## Provider token streams should degrade gracefully

For LLM provider `stream()` implementations in `rex/llm_client.py`, catch exceptions raised
during iteration, log a warning, and stop the generator so already-yielded tokens are preserved.

## Offline transport test harnesses live in tests/helpers

For IMAP/SMTP/Twilio integration tests, prefer `tests/helpers/fake_imap.py`,
`tests/helpers/fake_smtp.py`, and `tests/helpers/fake_twilio.py` over ad hoc `MagicMock`
transports. These helpers record calls in simple lists/counters (for example `login_calls`,
`send_message_calls`, `store_calls`) instead of mock assertion helpers like
`assert_called_once_with`.
