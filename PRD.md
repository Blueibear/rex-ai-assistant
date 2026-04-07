# PRD: AskRex Assistant — Full Roadmap

> **Codex/Ralph task selection rule**
> A "task" means one full User Story (US-###), not an individual checkbox line.
> Choose the first US-### that contains any unchecked acceptance criteria `[ ]`.

## Introduction

AskRex Assistant is a local-first, voice-activated AI companion supporting wake word detection, STT, LLM chat, TTS, and optional integrations for search, messaging, email, calendar, and Home Assistant. This PRD covers the complete roadmap across 9 phases, from critical system fixes through developer tooling. Each user story is sized for a single AI implementation context window (~10 min of work) and ordered by dependency.

**Tech stack:** Python 3.11, Flask, React (GUI), Pydantic v2, OpenAI Whisper, openWakeWord, Coqui XTTS, edge-tts, pyttsx3, Ollama/OpenAI LLM backends.

**Target platforms:** Windows 11, macOS, Linux (all AC must pass cross-platform).

---

## Goals

- Eliminate all blocking runtime failures (bridge execution, voice pipeline, dependency resolution)
- Establish a reliable, cross-platform voice loop (wake -> STT -> LLM -> TTS)
- Connect real Home Assistant control with context-aware, alias-friendly commands
- Build assistant intelligence (tool selection, speed perception, proactive suggestions)
- Add communication layer (Telegram, notifications, cloud usage tracking)
- Implement multi-user system with permissions and personality
- Enable safe desktop/computer control
- Overhaul UI/UX for discoverability and guided setup
- Bring documentation, installer, and CI to production quality
- Establish structured logging and debug infrastructure

---

## User Stories

---

### PHASE 1 -- CORE SYSTEM FIXES (BLOCKERS)

---

#### US-001: Add venv-aware Python resolver utility
**Description:** As a developer, I want a utility function that resolves the correct venv Python binary so that all bridge scripts use a consistent interpreter.

**Acceptance Criteria:**
- [x] New module `rex/bridge_utils.py` exports `resolve_python()` returning the absolute path to the active venv Python
- [x] On Windows, resolves `.venv\Scripts\python.exe`; on macOS/Linux, resolves `.venv/bin/python`
- [x] Falls back to `sys.executable` if no venv detected
- [x] Unit test covers Windows, macOS, Linux path resolution
- [x] Typecheck passes (`mypy rex/bridge_utils.py`)

---

#### US-002: Replace raw `python` calls in bridge scripts with venv resolver
**Description:** As a developer, I want all bridge scripts (`rex_*_bridge.py`) to use the venv-aware resolver so that subprocess calls never use the wrong interpreter.

**Acceptance Criteria:**
- [x] Every `rex_*_bridge.py` file at repo root imports and uses `resolve_python()` from `rex/bridge_utils.py`
- [x] No raw `"python"` or `"python3"` string remains in any bridge subprocess call
- [x] `grep -r "subprocess.*['\"]python" rex_*_bridge.py` returns zero matches
- [x] Existing bridge tests still pass
- [x] Typecheck passes

---

#### US-003: Replace raw `python` calls in GUI backend with venv resolver
**Description:** As a developer, I want `rex/gui_app.py` and any GUI subprocess calls to use the venv-aware resolver.

**Acceptance Criteria:**
- [x] `rex/gui_app.py` uses `resolve_python()` for all subprocess invocations
- [x] `grep -r "subprocess.*['\"]python" rex/gui_app.py` returns zero matches
- [x] GUI launch still works via `rex-gui` entry point
- [x] Typecheck passes

---

#### US-004: Fix working directory resolution for bridge scripts
**Description:** As a developer, I want bridge scripts to resolve the repo root correctly so that relative path failures are eliminated.

**Acceptance Criteria:**
- [x] `rex/bridge_utils.py` exports `repo_root()` returning the absolute path to the repo root (directory containing `pyproject.toml`)
- [x] All `rex_*_bridge.py` files use `repo_root()` to build absolute paths to scripts and config
- [x] No bridge script uses `os.getcwd()` or relative paths to locate other scripts
- [x] Unit test confirms `repo_root()` returns the correct directory
- [x] Typecheck passes

---

#### US-005: Verify and create missing bridge scripts (tasks, reminders, memory)
**Description:** As a developer, I want the `rex_tasks_bridge.py`, `rex_reminders_bridge.py`, and `rex_memories_bridge.py` scripts to exist and conform to the standard JSON I/O contract.

**Acceptance Criteria:**
- [x] `rex_tasks_bridge.py` exists, accepts JSON on stdin `{"action": "list"|"add"|"complete", ...}`, returns JSON on stdout
- [x] `rex_reminders_bridge.py` exists with the same JSON I/O pattern
- [x] `rex_memories_bridge.py` exists with the same JSON I/O pattern
- [x] Each script returns `{"error": "..."}` on invalid input (not a traceback)
- [x] Smoke test for each: `echo '{"action":"list"}' | python <bridge>.py` returns valid JSON
- [x] Typecheck passes

---

#### US-006: Verify and create missing bridge scripts (shopping, speakers)
**Description:** As a developer, I want the `rex_shopping_list_bridge.py` and `rex_speaker_bridge.py` scripts to exist and conform to the standard JSON I/O contract.

**Acceptance Criteria:**
- [x] `rex_shopping_list_bridge.py` exists, accepts JSON on stdin, returns JSON on stdout
- [x] `rex_speaker_bridge.py` exists with the same JSON I/O pattern
- [x] Each script returns `{"error": "..."}` on invalid input
- [x] Smoke test for each returns valid JSON
- [x] Typecheck passes

---

#### US-007: Trace and fix voice pipeline wake-to-capture stage
**Description:** As a developer, I want the wake word detection to reliably trigger audio capture so that the voice pipeline does not hang at the first stage.

**Acceptance Criteria:**
- [x] `rex/wakeword/listener.py` emits a structured log event on wake word detection
- [x] Audio capture begins within 200ms of wake word detection (log timestamps confirm)
- [x] If wake word config is empty or invalid, a clear error is raised at startup (not a silent hang)
- [x] Test covers the wake -> capture transition with a mock audio stream
- [x] Works on Windows, macOS, Linux
- [x] Typecheck passes

---

#### US-008: Trace and fix voice pipeline capture-to-STT stage
**Description:** As a developer, I want captured audio to be reliably passed to the STT engine so that transcription always occurs after capture.

**Acceptance Criteria:**
- [x] Audio capture completion emits a structured log event with audio duration
- [x] STT engine receives the audio buffer and begins transcription (log confirms handoff)
- [x] If STT fails, the error is logged and the pipeline resets (no hang)
- [x] Test covers capture -> STT handoff with a mock audio buffer
- [x] Typecheck passes

---

#### US-009: Trace and fix voice pipeline STT-to-LLM-to-TTS stage
**Description:** As a developer, I want the LLM response to be generated from the transcript and spoken back via TTS so that the full voice loop completes.

**Acceptance Criteria:**
- [x] STT result is passed to `Assistant.generate_reply()` (not raw `LanguageModel.generate()`)
- [x] LLM response is passed to the TTS engine and audio playback begins
- [x] If TTS fails, the text response is logged and the pipeline resets (no hang)
- [x] End-to-end test covers STT transcript -> LLM -> TTS with mocks
- [x] Typecheck passes

---

#### US-074: Diagnose and fix standalone rex_loop.py voice conversation
**Description:** As a developer, I want `rex_loop.py` (the standalone voice loop entry point) to complete a full voice conversation reliably, not just detect the wake word.

**Acceptance Criteria:**
- [x] Run `python rex_loop.py` and document which stages succeed: wake word detection, audio capture, STT transcription, LLM response generation, TTS spoken output
- [x] For each failing stage, add a structured log message identifying the failure point and cause
- [x] Fix all identified failures so that a full wake -> capture -> transcribe -> LLM -> speak cycle completes
- [x] `rex_loop.py` uses `build_voice_loop` from `rex.voice_loop` (the canonical implementation, per learned rules)
- [x] If a stage cannot be fixed in this story (e.g., missing hardware), the loop logs the blocker and exits cleanly instead of hanging
- [x] Integration test with mocked audio confirms full pipeline completion
- [x] Typecheck passes

---

#### US-075: Fix config type validation and coercion warnings
**Description:** As a developer, I want config values like `llm_temperature` to be stored and validated as their correct types so that runtime coercion warnings are eliminated.

**Acceptance Criteria:**
- [x] `AppConfig` field `llm_temperature` is typed as `float` (not `str`), with a Pydantic validator that coerces string input and logs a deprecation warning
- [x] All other config fields that are currently stored as strings but used as numeric types are similarly corrected
- [x] `config/rex_config.json` template uses correct JSON types (numbers, not quoted numbers)
- [x] Config validation runs on load and raises clear errors for invalid values (e.g., `"temperature": "abc"`)
- [x] `rex doctor` includes a config validation check that reports any type mismatches
- [x] No `UserWarning` or coercion warning on clean config load
- [x] Unit test confirms both correct-type and string-type inputs are handled
- [x] Typecheck passes

---

#### US-076: Remove mock calendar from voice loop runtime path
**Description:** As a developer, I want the voice loop to use real calendar integration (or report "not configured") instead of silently running in mock mode.

**Acceptance Criteria:**
- [x] Voice loop startup log does NOT show "Calendar service connected (mock mode)" when calendar is configured with real credentials
- [x] If calendar ICS URL is not configured, voice loop logs "Calendar: not configured" (not "mock mode")
- [x] Mock calendar data is only used in test fixtures, never in production runtime paths
- [x] `rex/calendar_service.py` raises `IntegrationNotConfiguredError` instead of returning mock data when unconfigured
- [x] Voice loop gracefully handles missing calendar (no crash, just skips calendar-related tool)
- [x] Test confirms mock is never loaded outside of test context
- [x] Typecheck passes

---

#### US-077: Pin XTTS/transformers compatible versions and patch BeamSearchScorer
**Description:** As a developer, I want XTTS to load successfully by pinning compatible transformers version or patching the missing `BeamSearchScorer` import.

**Acceptance Criteria:**
- [x] `requirements-gpu-cu124.txt` (and other GPU requirements files) pin a transformers version compatible with the installed XTTS version
- [x] If `BeamSearchScorer` is missing from the installed transformers, a compatibility shim provides it before XTTS loads
- [x] Shim is applied in `rex/compat/` and triggered by lazy import logic (using `find_spec()` per learned rules)
- [x] If pinned version is not available, TTS falls back to edge-tts with a clear log message
- [x] `rex doctor` reports XTTS + transformers version compatibility status
- [x] Test confirms XTTS loads (or falls back cleanly) with both compatible and incompatible transformers versions
- [x] Typecheck passes

---

#### US-078: Fix torio/FFmpeg runtime dependency for voice pipeline
**Description:** As a developer, I want FFmpeg extension loading failures in torio to be resolved or handled so the voice pipeline does not crash on audio operations.

**Acceptance Criteria:**
- [x] Determine whether FFmpeg is required for the voice pipeline audio path (capture, playback) or only for XTTS
- [x] If required: add FFmpeg to install docs and `rex doctor` prerequisites; `install.py` checks for FFmpeg
- [x] If not required: suppress torio FFmpeg warnings and ensure audio operations use an alternative backend
- [x] Voice pipeline audio capture and playback work without FFmpeg extensions loaded (or FFmpeg is present)
- [x] `rex doctor` reports FFmpeg status and whether it is required for the active TTS backend
- [x] No unhandled exception from torio/FFmpeg during voice loop operation
- [x] Test confirms voice pipeline startup succeeds with and without FFmpeg
- [x] Typecheck passes

---

#### US-010: Fix voice pipeline hang states
**Description:** As a developer, I want the voice loop to have timeouts at each stage so that it never hangs indefinitely.

**Acceptance Criteria:**
- [x] Configurable timeout (default 30s) for STT transcription
- [x] Configurable timeout (default 60s) for LLM generation
- [x] Configurable timeout (default 30s) for TTS synthesis
- [x] On timeout, pipeline logs the stage, resets, and re-enters listening state
- [x] Test simulates a timeout at each stage and confirms recovery
- [x] Typecheck passes

---

#### US-011: Add missing voice dependencies to requirements
**Description:** As a developer, I want `edge-tts` and `pyttsx3` and any other missing voice deps included in the install so that all TTS backends work out of the box.

**Acceptance Criteria:**
- [x] `edge-tts` is in `requirements.txt` (or `pyproject.toml` dependencies)
- [x] `pyttsx3` is in `requirements.txt` (or `pyproject.toml` dependencies)
- [x] `pip install .` in a fresh venv installs both without errors
- [x] `python -c "import edge_tts; import pyttsx3"` succeeds after install
- [x] No other runtime `ModuleNotFoundError` for voice-related imports
- [x] Works on Windows, macOS, Linux
- [x] Typecheck passes

---

#### US-012: Fix custom voice duration validation
**Description:** As a user, I want accurate validation messaging when uploading a custom voice sample so that I know exactly what is wrong.

**Acceptance Criteria:**
- [ ] Duration check in `rex/custom_voices.py` correctly calculates audio duration in seconds
- [ ] If sample is too short, message says "Sample is X.Xs, minimum is Ys"
- [ ] If sample is too long, message says "Sample is X.Xs, maximum is Ys"
- [ ] If format is unsupported, message names the format and lists accepted formats
- [ ] Unit test with known-duration audio files confirms correct validation
- [ ] Typecheck passes

---

#### US-013: Remove mock calendar data and connect real backend
**Description:** As a user, I want the calendar integration to return real data (or a clear "not configured" message) instead of fake events.

**Acceptance Criteria:**
- [ ] No hardcoded fake calendar events remain in `rex/calendar_service.py` or `rex/calendar_backends/`
- [ ] If ICS feed URL is not configured, API returns `{"status": "not_configured", "message": "..."}`
- [ ] If ICS feed URL is configured, API returns real parsed events
- [ ] Test covers both configured and not-configured paths
- [ ] Typecheck passes

---

#### US-014: Remove mock email data and connect real backend
**Description:** As a user, I want the email integration to return real data (or a clear "not configured" message) instead of fake messages.

**Acceptance Criteria:**
- [ ] No hardcoded fake email data remains in `rex/email_service.py` or `rex/email_backends/`
- [ ] If IMAP/SMTP credentials are absent, API returns `{"status": "not_configured", "message": "..."}`
- [ ] If credentials are present, API returns real inbox data
- [ ] Test covers both configured and not-configured paths
- [ ] Typecheck passes

---

#### US-015: Replace "exit code 2" with meaningful error reporting
**Description:** As a developer, I want bridge script failures to surface tracebacks, stderr, and meaningful messages instead of opaque exit codes.

**Acceptance Criteria:**
- [ ] All bridge scripts wrap execution in try/except and return `{"error": "<message>", "traceback": "<tb>"}` on failure
- [ ] GUI backend captures stderr from subprocess calls and includes it in error responses
- [ ] CLI mode prints the actual error message, not just "exit code 2"
- [ ] Test confirms a deliberately broken bridge returns a readable error
- [ ] Typecheck passes

---

#### US-016: Fix STT language handling for "auto" mode
**Description:** As a user, I want STT to accept `"auto"` as a language setting and fall back to `"en"` without crashing.

**Acceptance Criteria:**
- [ ] If `stt_language` config is `"auto"`, STT engine is called with `language=None` (auto-detect)
- [ ] If STT engine does not support auto-detect, falls back to `"en"`
- [ ] No crash or exception when `stt_language` is `"auto"`, empty string, or missing
- [ ] Unit test covers `"auto"`, `"en"`, `""`, and `None` inputs
- [ ] Typecheck passes

---

#### US-017: Fix Whisper/STT runtime failure and error exposure
**Description:** As a developer, I want Whisper STT failures to produce real error messages and for the correct backend (faster-whisper vs whisper) to be verified at startup.

**Acceptance Criteria:**
- [ ] `rex doctor` checks which STT backend is installed and reports it
- [ ] If neither whisper nor faster-whisper is installed, `rex doctor` reports the gap
- [ ] STT runtime errors are caught and logged with full traceback (not swallowed)
- [ ] If transcription fails, the voice loop logs the error and resets (no hang)
- [ ] Test simulates STT failure and confirms error is surfaced
- [ ] Typecheck passes

---

#### US-018: Fix wake word config mismatch and empty resolution
**Description:** As a developer, I want the wake word config to resolve consistently so that an empty or mismatched setting does not silently break detection.

**Acceptance Criteria:**
- [ ] If `wake_word` config is empty or `None`, a sensible default is used (e.g., `"hey_rex"`)
- [ ] If the configured wake word model file does not exist, startup raises a clear error
- [ ] `rex doctor` validates wake word config and reports status
- [ ] Test covers empty, None, valid, and invalid wake word configs
- [ ] Typecheck passes

---

#### US-019: Fix XTTS/transformers compatibility issues
**Description:** As a developer, I want XTTS and transformers to load without import errors or deprecation crashes.

**Acceptance Criteria:**
- [ ] Lazy import of XTTS uses `find_spec()` before `import_module()` (per learned rules)
- [ ] Compatibility shims for transformers version differences are applied before XTTS load
- [ ] If XTTS is not installed, TTS gracefully falls back to edge-tts or pyttsx3
- [ ] No `ImportError` or `AttributeError` on `import rex.tts_utils` with or without XTTS installed
- [ ] Test covers XTTS-present and XTTS-absent scenarios
- [ ] Typecheck passes

---

#### US-020: Fix FFmpeg/torio errors and config coercion warnings
**Description:** As a developer, I want FFmpeg and torio-related errors to be handled cleanly and config coercion warnings to be resolved.

**Acceptance Criteria:**
- [ ] If FFmpeg is not on PATH, a clear warning is logged at startup (not a crash)
- [ ] `rex doctor` checks for FFmpeg and reports its presence/version
- [ ] Config values that trigger coercion warnings are fixed to use correct types in `AppConfig`
- [ ] No `UserWarning` or `DeprecationWarning` from config loading
- [ ] Test confirms config loads without warnings
- [ ] Typecheck passes

---

### PHASE 2 -- HOME AUTOMATION CORE

---

#### US-021: Add Music Assistant HTTP client
**Description:** As a developer, I need an HTTP client for Music Assistant so that Rex can send playback commands.

**Acceptance Criteria:**
- [ ] New module `rex/integrations/music_assistant.py` with `MusicAssistantClient` class
- [ ] Client supports: `play(query, room=None)`, `pause(room=None)`, `resume(room=None)`, `skip(room=None)`, `set_volume(level, room=None)`
- [ ] Config fields: `music_assistant_url`, `music_assistant_token` in `AppConfig`
- [ ] If not configured, all methods raise `IntegrationNotConfiguredError`
- [ ] Unit test with mocked HTTP responses for each method
- [ ] Typecheck passes

---

#### US-022: Wire Music Assistant commands to assistant tool routing
**Description:** As a user, I want to say "play Shape of You" and have Rex send the command to Music Assistant.

**Acceptance Criteria:**
- [ ] `Assistant.generate_reply()` recognizes music intent and routes to `MusicAssistantClient`
- [ ] Tool catalog includes music commands (play, pause, resume, skip, volume)
- [ ] Room targeting works: "play jazz in the kitchen" targets the kitchen speaker
- [ ] If Music Assistant is not configured, assistant replies "Music Assistant is not set up"
- [ ] Integration test with mocked Music Assistant confirms routing
- [ ] Typecheck passes

---

#### US-023: Add room context system
**Description:** As a developer, I need a room context module so that commands can be scoped to the room the user is in.

**Acceptance Criteria:**
- [ ] New module `rex/context/room.py` with `RoomContext` class
- [ ] `RoomContext.current_room` is settable via: explicit parameter, speaker origin, last active UI context, config default
- [ ] Priority order: explicit > speaker origin > last active > config default
- [ ] Unit test confirms priority resolution
- [ ] Typecheck passes

---

#### US-024: Add speaker origin detection to room context
**Description:** As a user, I want Rex to know which room I am speaking from based on the input device or MQTT topic.

**Acceptance Criteria:**
- [ ] `RoomContext` can be populated from MQTT audio topic (e.g., `rex/audio/kitchen`)
- [ ] `RoomContext` can be populated from a configured device-to-room mapping in config
- [ ] If no mapping exists, `current_room` falls back to default
- [ ] Test covers MQTT topic, device mapping, and fallback paths
- [ ] Typecheck passes

---

#### US-025: Add device alias system with synonym and fuzzy matching
**Description:** As a user, I want to say "turn on the bedroom light" and have Rex resolve that to the actual Home Assistant entity ID.

**Acceptance Criteria:**
- [ ] New module `rex/ha/device_aliases.py` with `AliasResolver` class
- [ ] Aliases stored in `config/device_aliases.json` mapping natural names to HA entity IDs
- [ ] Fuzzy matching: "bedrom light" resolves to "bedroom light" (Levenshtein distance <= 2)
- [ ] Synonyms: "lamp" matches "light" if configured
- [ ] `resolve(query)` returns `(entity_id, confidence)` or `None`
- [ ] Unit test covers exact match, fuzzy match, synonym, and no-match cases
- [ ] Typecheck passes

---

#### US-026: Add device discovery via Home Assistant API
**Description:** As a user, I want Rex to scan Home Assistant for available devices so I can approve and name them.

**Acceptance Criteria:**
- [ ] `rex/ha/discovery.py` calls HA `/api/states` to list all entities
- [ ] Returns list of `{entity_id, friendly_name, domain, state}`
- [ ] Results cached for 5 minutes (configurable)
- [ ] If HA is not configured, returns empty list with a log warning
- [ ] Unit test with mocked HA API response
- [ ] Typecheck passes

---

#### US-027: Add device approval and rename workflow
**Description:** As a user, I want to approve discovered devices and give them custom names that Rex will recognize.

**Acceptance Criteria:**
- [ ] `rex/ha/discovery.py` exports `approve_device(entity_id, alias)` and `ignore_device(entity_id)`
- [ ] Approved devices are written to `config/device_aliases.json`
- [ ] Ignored devices are written to `config/device_ignore.json`
- [ ] CLI command `rex ha approve` lists pending devices and accepts approval
- [ ] Test covers approve, rename, and ignore workflows
- [ ] Typecheck passes

---

#### US-028: Add device state awareness
**Description:** As a developer, I need Rex to query real-time device state from HA so it can respond intelligently.

**Acceptance Criteria:**
- [ ] `rex/ha/device_state.py` queries HA `/api/states/<entity_id>` for current state
- [ ] Returns structured data: `{entity_id, state, attributes: {brightness, volume, media_title, ...}}`
- [ ] If entity not found, returns `None`
- [ ] `Assistant` can answer "is the kitchen light on?" using device state
- [ ] Unit test with mocked HA state responses
- [ ] Typecheck passes

---

#### US-029: Add command confirmation and undo support
**Description:** As a user, I want Rex to confirm actions ("Turned off the bedroom light") and offer undo ("Say undo to turn it back on").

**Acceptance Criteria:**
- [ ] After executing an HA command, Rex speaks a confirmation including device name and action
- [ ] Undo state is stored for the last 5 commands (FIFO)
- [ ] "Undo" or "undo that" within 30 seconds reverses the last command
- [ ] Undo sends the inverse HA command (on->off, off->on, volume up->volume down)
- [ ] Test covers confirmation message generation and undo reversal
- [ ] Typecheck passes

---

#### US-030: Add clarification system for ambiguous commands
**Description:** As a user, I want Rex to ask for clarification when a command is ambiguous instead of guessing wrong.

**Acceptance Criteria:**
- [ ] If `AliasResolver.resolve()` returns multiple matches with similar confidence, Rex asks "Did you mean X or Y?"
- [ ] If a command is missing required context (e.g., "turn it on" with no recent device reference), Rex asks "Which device?"
- [ ] Clarification question is spoken via TTS and the pipeline re-enters listening for the answer
- [ ] Test covers multi-match and missing-context scenarios
- [ ] Typecheck passes

---

#### US-031: Add error recovery with alternative suggestions
**Description:** As a user, I want Rex to suggest alternatives when a command fails instead of just saying "error."

**Acceptance Criteria:**
- [ ] If an HA command fails (device offline, unreachable), Rex says what went wrong and suggests an alternative
- [ ] Example: "The kitchen light is not responding. Would you like me to try the dining room light instead?"
- [ ] Alternatives sourced from same-room devices or recently used devices
- [ ] If no alternative exists, Rex says "I could not complete that. The device may be offline."
- [ ] Test covers device-offline and alternative-suggestion paths
- [ ] Typecheck passes

---

### PHASE 3 -- ASSISTANT INTELLIGENCE

---

#### US-032: Add tool auto-selection system
**Description:** As a user, I want Rex to automatically choose the right tool (search, HA, calendar, email) without me specifying which one to use.

**Acceptance Criteria:**
- [ ] `rex/tool_catalog.py` exposes a registry of available tools with intent patterns
- [ ] `Assistant.generate_reply()` uses LLM function-calling or pattern matching to select the right tool
- [ ] If multiple tools match, the highest-confidence one is chosen
- [ ] If no tool matches, Rex falls back to conversational LLM response
- [ ] Test covers weather (search), "turn on light" (HA), "what's on my calendar" (calendar) routing
- [ ] Typecheck passes

---

#### US-033: Add perceived speed system (instant acknowledgment)
**Description:** As a user, I want Rex to immediately acknowledge my command so I know it was heard, even if processing takes time.

**Acceptance Criteria:**
- [ ] After wake word + STT, Rex plays a short acknowledgment sound or speaks "On it" before LLM processing
- [ ] Acknowledgment happens within 500ms of STT completion
- [ ] Acknowledgment is configurable (sound, phrase, or disabled)
- [ ] Config field: `acknowledgment_mode` in `AppConfig` (values: `"sound"`, `"phrase"`, `"none"`)
- [ ] Test confirms acknowledgment fires before LLM call
- [ ] Typecheck passes

---

#### US-034: Add progressive response system
**Description:** As a user, I want Rex to speak partial responses as they stream in for long answers.

**Acceptance Criteria:**
- [ ] If LLM supports streaming, TTS begins on the first complete sentence
- [ ] Subsequent sentences are queued and spoken sequentially
- [ ] If LLM does not support streaming, behavior falls back to full-response TTS
- [ ] No audio overlap between sentence chunks
- [ ] Test confirms sentence-level streaming with a mock streaming LLM
- [ ] Typecheck passes

---

#### US-035: Add proactive suggestion engine (pattern detection)
**Description:** As a developer, I need a module that detects repeated user patterns and suggests automations.

**Acceptance Criteria:**
- [ ] New module `rex/suggestions/pattern_detector.py`
- [ ] Tracks command history and detects patterns (e.g., "user turns on kitchen light every day at 7am")
- [ ] Pattern requires at least 3 occurrences within a time window to be considered
- [ ] `detect_patterns()` returns a list of `{pattern, frequency, suggested_automation}`
- [ ] Suggestions are never acted on automatically; always presented as questions
- [ ] Unit test with synthetic command history confirms pattern detection
- [ ] Typecheck passes

---

#### US-036: Surface proactive suggestions to the user
**Description:** As a user, I want Rex to occasionally suggest automations based on my habits, and let me accept or dismiss them.

**Acceptance Criteria:**
- [ ] At most one suggestion per session (not spammy)
- [ ] Suggestion is spoken: "I noticed you turn on the kitchen light at 7am most days. Want me to automate that?"
- [ ] User can accept ("yes") or dismiss ("no thanks")
- [ ] Dismissed patterns are not suggested again for 30 days
- [ ] Accepted patterns create a scheduled automation entry
- [ ] Test covers suggest, accept, and dismiss flows
- [ ] Typecheck passes

---

#### US-037: Add capability registry
**Description:** As a developer, I need a structured registry of all Rex capabilities so the LLM, UI, and docs can query it.

**Acceptance Criteria:**
- [ ] New module `rex/capabilities/registry.py` with `CapabilityRegistry` class
- [ ] Each capability has: `name`, `description`, `inputs`, `outputs`, `triggers`, `enabled`
- [ ] Registry auto-populates from installed integrations at startup
- [ ] `registry.list()` returns all capabilities; `registry.search(query)` filters by keyword
- [ ] Unit test confirms registry populates and search works
- [ ] Typecheck passes

---

#### US-038: Add "What can you do?" dynamic response
**Description:** As a user, I want to ask "What can you do?" and get an accurate, context-aware list of current capabilities.

**Acceptance Criteria:**
- [ ] "What can you do?" intent is recognized by the assistant
- [ ] Response is generated from `CapabilityRegistry`, listing only enabled capabilities
- [ ] Response is grouped by category (Home, Communication, Productivity, etc.)
- [ ] If no capabilities are configured, Rex says "I can chat with you, but no integrations are set up yet"
- [ ] Test confirms response reflects actual enabled capabilities
- [ ] Typecheck passes

---

### PHASE 4 -- COMMUNICATION LAYER

---

#### US-039: Add Telegram bot integration (send messages)
**Description:** As a developer, I need a Telegram bot client so Rex can send messages to the user.

**Acceptance Criteria:**
- [ ] New module `rex/integrations/telegram/client.py` with `TelegramClient` class
- [ ] Config fields: `telegram_bot_token`, `telegram_chat_id` in `AppConfig`; token in `.env`
- [ ] `send_message(text)` sends a message to the configured chat
- [ ] If not configured, raises `IntegrationNotConfiguredError`
- [ ] Unit test with mocked Telegram API
- [ ] Typecheck passes

---

#### US-040: Add Telegram bot integration (receive commands)
**Description:** As a user, I want to send commands to Rex via Telegram and get responses back.

**Acceptance Criteria:**
- [ ] Telegram webhook or polling handler receives incoming messages
- [ ] Incoming text is routed through `Assistant.generate_reply()`
- [ ] Response is sent back to the Telegram chat
- [ ] Unrecognized commands get a conversational LLM response
- [ ] Test covers inbound message -> assistant -> outbound response flow
- [ ] Typecheck passes

---

#### US-041: Add local desktop notifications
**Description:** As a user, I want Rex to show desktop notifications for important events (reminders, alerts).

**Acceptance Criteria:**
- [ ] New module `rex/notifications/desktop.py`
- [ ] Uses `plyer` or platform-native API for cross-platform notifications
- [ ] `notify(title, message, urgency="normal")` shows a desktop notification
- [ ] Works on Windows (toast), macOS (notification center), Linux (libnotify)
- [ ] If notification system unavailable, logs a warning (no crash)
- [ ] Unit test confirms notification call is made (mocked)
- [ ] Typecheck passes

---

#### US-042: Add push notification support
**Description:** As a user, I want to receive push notifications on my phone when Rex has an alert.

**Acceptance Criteria:**
- [ ] New module `rex/notifications/push.py` supporting at least one provider (ntfy.sh or Pushover)
- [ ] Config fields: `push_provider`, `push_token`, `push_topic` in `AppConfig`
- [ ] `send_push(title, message, priority="normal")` sends a push notification
- [ ] If not configured, raises `IntegrationNotConfiguredError`
- [ ] Unit test with mocked HTTP
- [ ] Typecheck passes

---

#### US-043: Add Ollama cloud usage tracking
**Description:** As a developer, I need to track per-request token usage for Ollama so users know their consumption.

**Acceptance Criteria:**
- [ ] `rex/llm_client.py` Ollama backend logs `{model, prompt_tokens, completion_tokens, timestamp}` per request
- [ ] Usage records stored in `data/llm_usage.json` (append-only, rotated at 10MB)
- [ ] `rex usage` CLI command prints a summary (total requests, total tokens, by model)
- [ ] Unit test confirms usage is recorded on LLM call
- [ ] Typecheck passes

---

#### US-044: Add smart cloud routing (prefer local)
**Description:** As a user, I want Rex to prefer local Ollama for simple tasks and reserve cloud LLM for complex ones.

**Acceptance Criteria:**
- [ ] `rex/model_router.py` routes based on estimated complexity (message length, tool requirements)
- [ ] Simple queries (< 200 tokens, no tools) go to local Ollama if available
- [ ] Complex queries (tools required, long context) go to cloud provider if configured
- [ ] If local is unavailable, all queries go to cloud (with a log warning)
- [ ] Config: `llm_routing_mode` in `AppConfig` (values: `"local_preferred"`, `"cloud_only"`, `"local_only"`)
- [ ] Test covers routing decisions for simple and complex queries
- [ ] Typecheck passes

---

#### US-045: Add cloud fallback when usage limit hit
**Description:** As a user, I want Rex to automatically fall back to local when my cloud API limit is reached.

**Acceptance Criteria:**
- [ ] If cloud LLM returns 429 (rate limit) or 402 (quota exceeded), model router switches to local
- [ ] User is notified: "Cloud limit reached, switching to local model"
- [ ] Router retries cloud after a configurable cooldown (default 1 hour)
- [ ] Test simulates 429 response and confirms fallback
- [ ] Typecheck passes

---

#### US-046: Add cloud usage visibility to UI
**Description:** As a user, I want to see local vs cloud LLM usage in the dashboard.

**Acceptance Criteria:**
- [ ] Dashboard API endpoint `GET /api/usage` returns `{local: {requests, tokens}, cloud: {requests, tokens}}`
- [ ] React dashboard displays usage summary (today, this week, this month)
- [ ] Percentage bar shows local vs cloud split
- [ ] Data sourced from `data/llm_usage.json`
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### PHASE 5 -- USER SYSTEM

---

#### US-047: Add user authentication (login system)
**Description:** As a user, I want to log in with a username and password so my data is separate from other users.

**Acceptance Criteria:**
- [ ] New module `rex/auth.py` with `create_user(username, password)`, `authenticate(username, password)`, `get_current_user()`
- [ ] Passwords hashed with bcrypt
- [ ] Users stored in `data/users.db` (SQLite)
- [ ] Session tokens issued on login (JWT, 24h expiry)
- [ ] API endpoints: `POST /api/auth/login`, `POST /api/auth/register`, `POST /api/auth/logout`
- [ ] Unit test covers registration, login, bad password, and token validation
- [ ] Typecheck passes

---

#### US-048: Add per-user data isolation
**Description:** As a user, I want my memories, preferences, and history to be separate from other users.

**Acceptance Criteria:**
- [ ] Memory profiles keyed by user ID (not just default profile)
- [ ] Conversation history keyed by user ID
- [ ] Config preferences (TTS voice, wake word) stored per user
- [ ] API requests require valid session token; data scoped to authenticated user
- [ ] Test confirms User A cannot see User B's data
- [ ] Typecheck passes

---

#### US-049: Add profile picture support
**Description:** As a user, I want to upload a profile picture that appears in the dashboard.

**Acceptance Criteria:**
- [ ] API endpoint `POST /api/user/avatar` accepts image upload (JPEG/PNG, max 2MB)
- [ ] Image stored in `data/avatars/<user_id>.jpg` (resized to 256x256)
- [ ] API endpoint `GET /api/user/avatar` returns the image
- [ ] Default avatar used if none uploaded
- [ ] Dashboard displays the avatar in the header
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-050: Add personality system (backend)
**Description:** As a developer, I need a personality system that controls the assistant's tone and style.

**Acceptance Criteria:**
- [ ] New module `rex/personality.py` with `Personality` dataclass: `name`, `system_prompt`, `tone_keywords`, `greeting`
- [ ] Built-in personalities: "Professional", "Friendly", "Minimal"
- [ ] `get_personality(name)` returns the personality; `list_personalities()` returns all
- [ ] `Assistant` injects the active personality's system prompt into LLM calls
- [ ] Config field: `personality` in per-user config (default: "Friendly")
- [ ] Unit test confirms personality prompt injection
- [ ] Typecheck passes

---

#### US-051: Add personality preview and selection UI
**Description:** As a user, I want to preview and switch personalities in the dashboard.

**Acceptance Criteria:**
- [ ] Dashboard settings page shows available personalities with preview text
- [ ] Selecting a personality updates the user's config
- [ ] Preview shows a sample greeting in the selected personality's tone
- [ ] Change takes effect on next interaction (no restart required)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-052: Add permissions system
**Description:** As an admin, I want to restrict sensitive actions (computer control, email send) to specific users.

**Acceptance Criteria:**
- [ ] New module `rex/permissions.py` with `Permission` enum and `check_permission(user, action)` function
- [ ] Permissions: `computer_control`, `email_send`, `sms_send`, `ha_control`, `admin`
- [ ] Permissions stored per user in `data/users.db`
- [ ] First registered user gets `admin` permission by default
- [ ] API actions check permissions before execution; return 403 if denied
- [ ] Unit test covers grant, revoke, and denial
- [ ] Typecheck passes

---

### PHASE 6 -- DESKTOP / COMPUTER CONTROL

---

#### US-053: Add desktop file read/write capability
**Description:** As a user, I want Rex to read and write files on my computer when I ask.

**Acceptance Criteria:**
- [ ] `rex/computers/file_ops.py` exports `read_file(path)`, `write_file(path, content)`, `list_dir(path)`
- [ ] Operations restricted to an allowlisted set of directories (configurable)
- [ ] Attempts to access paths outside the allowlist return a permission error
- [ ] Works on Windows, macOS, Linux (path normalization handled)
- [ ] Unit test covers read, write, list, and blocked-path scenarios
- [ ] Typecheck passes

---

#### US-054: Add desktop program launch capability
**Description:** As a user, I want Rex to open applications when I ask ("open Notepad", "launch Chrome").

**Acceptance Criteria:**
- [ ] `rex/computers/app_launcher.py` exports `launch_app(name)`
- [ ] App name resolved via a configurable app registry (`config/app_registry.json`)
- [ ] On Windows, uses `os.startfile()` or `subprocess`; on macOS, uses `open`; on Linux, uses `xdg-open`
- [ ] If app not found in registry, Rex says "I don't know how to open that. You can add it in settings."
- [ ] Unit test with mocked subprocess calls
- [ ] Typecheck passes

---

#### US-055: Add safety layer for computer control
**Description:** As a user, I want Rex to ask for confirmation before executing potentially dangerous computer actions.

**Acceptance Criteria:**
- [ ] Actions classified as `safe` (read file, list dir) or `dangerous` (write file, delete, execute command)
- [ ] Dangerous actions require voice or UI confirmation before execution
- [ ] Configurable: `computer_control_confirmation` in `AppConfig` (values: `"always"`, `"dangerous_only"`, `"never"`)
- [ ] Default is `"dangerous_only"`
- [ ] Test covers confirmation flow for dangerous action and bypass for safe action
- [ ] Typecheck passes

---

#### US-056: Add file summarization and search
**Description:** As a user, I want Rex to summarize a document or search my files for content.

**Acceptance Criteria:**
- [ ] `rex/computers/file_ops.py` exports `summarize_file(path)` and `search_files(directory, query)`
- [ ] Summarize reads the file and passes content to LLM with a summarize prompt
- [ ] Search uses `grep`-like matching across files in the directory (text files only)
- [ ] Both respect the directory allowlist
- [ ] Test covers summarize and search with mock file content
- [ ] Typecheck passes

---

### PHASE 7 -- UI / UX

---

#### US-057: Audit and expose all features in the UI
**Description:** As a user, I want every Rex feature to be visible and accessible in the dashboard (no hidden capabilities).

**Acceptance Criteria:**
- [ ] Dashboard navigation includes sections for: Chat, Voice, Home, Integrations, Settings, About
- [ ] Each configured integration has a visible entry in the Integrations section
- [ ] Each tool in the capability registry has a visible entry
- [ ] No feature is only accessible via CLI without a corresponding UI element
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-058: Add guided first-run setup wizard
**Description:** As a new user, I want a step-by-step setup wizard on first launch so I can configure Rex without guesswork.

**Acceptance Criteria:**
- [ ] On first launch (no `data/users.db`), the dashboard shows a setup wizard
- [ ] Steps: Create account -> Choose LLM provider -> Configure TTS -> (Optional) Home Assistant -> Done
- [ ] Each step validates input before allowing next
- [ ] Wizard writes config to `config/rex_config.json` and `.env`
- [ ] After completion, wizard does not show again
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-059: Add Home Assistant setup screen in dashboard
**Description:** As a user, I want a dedicated HA setup screen where I can enter my HA URL, token, and test the connection.

**Acceptance Criteria:**
- [ ] New dashboard page: Settings -> Home Assistant
- [ ] Fields: HA URL, Long-lived access token
- [ ] "Test Connection" button that calls HA `/api/` and reports success or failure
- [ ] On success, saves to `config/rex_config.json`
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-060: Add device control panel in dashboard
**Description:** As a user, I want a device control panel with toggles and sliders for my HA devices.

**Acceptance Criteria:**
- [ ] New dashboard page: Home -> Devices
- [ ] Lists approved devices from `config/device_aliases.json`
- [ ] Lights: on/off toggle + brightness slider
- [ ] Switches: on/off toggle
- [ ] Media players: play/pause, volume slider
- [ ] Controls send commands to HA in real-time
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-061: Add command history panel
**Description:** As a user, I want to see a history of recent commands and their results.

**Acceptance Criteria:**
- [ ] New dashboard panel: History
- [ ] Shows last 50 commands with: timestamp, command text, result, success/failure indicator
- [ ] Commands stored in `data/command_history.db` (SQLite)
- [ ] API endpoint: `GET /api/history?limit=50`
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-062: Add status indicators to dashboard
**Description:** As a user, I want to see Rex's current state (listening, thinking, executing, done) in the dashboard.

**Acceptance Criteria:**
- [ ] Dashboard header shows a status indicator with icon and label
- [ ] States: Idle, Listening, Thinking, Executing, Done, Error
- [ ] Status updates pushed via SSE (`rex/dashboard/sse.py`)
- [ ] Voice loop emits status change events at each pipeline stage
- [ ] Test confirms status events are emitted at each stage
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-063: Add quick actions panel
**Description:** As a user, I want one-click buttons for common actions (e.g., "Lights off", "Play music", "Lock up").

**Acceptance Criteria:**
- [ ] New dashboard panel: Quick Actions
- [ ] User can add/remove quick actions via settings
- [ ] Each action maps to a Rex command (text input to `Assistant.generate_reply()`)
- [ ] Quick actions stored in per-user config
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-064: Overhaul settings UX
**Description:** As a user, I want the settings page to use dropdowns, tooltips, and inline API instructions instead of raw text fields.

**Acceptance Criteria:**
- [ ] LLM provider selection uses a dropdown (Ollama, OpenAI, Local)
- [ ] TTS engine selection uses a dropdown (XTTS, edge-tts, pyttsx3)
- [ ] API key fields have a tooltip explaining where to get the key
- [ ] Each integration section has a link to setup docs
- [ ] No raw JSON editing required for any standard setting
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

#### US-065: Implement branding in UI
**Description:** As a user, I want the AskRex brand (logo, icons) to be consistent across the dashboard, system tray, and taskbar.

**Acceptance Criteria:**
- [ ] Dashboard header displays the AskRex logo
- [ ] System tray icon uses the AskRex icon (Windows, macOS, Linux)
- [ ] Taskbar/dock icon uses the AskRex icon
- [ ] Favicon is the AskRex icon
- [ ] No "Rex AI" or other banned names appear in the UI (per `docs/BRANDING.md`)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### PHASE 8 -- REPO + DOCUMENTATION

---

#### US-066: Full repo capability audit
**Description:** As a developer, I want a verified list of what Rex can and cannot do, so docs and UI do not overclaim.

**Acceptance Criteria:**
- [ ] Every feature listed in README.md is verified against `docs/claude/INTEGRATIONS_STATUS.md`
- [ ] Any feature marked STUB or NOT STARTED is either removed from README or explicitly marked as "coming soon"
- [ ] No feature is claimed as working that is not at least PARTIAL status
- [ ] `docs/claude/INTEGRATIONS_STATUS.md` is updated to reflect current state
- [ ] Typecheck passes

---

#### US-067: Documentation overhaul (README + INSTALL)
**Description:** As a new user, I want simple, accurate, step-by-step docs so I can install and run Rex without confusion.

**Acceptance Criteria:**
- [ ] `README.md` has: one-paragraph description, quick start (5 steps max), feature list (only verified features), link to full docs
- [ ] `INSTALL.md` has: prerequisites, step-by-step install for Windows/macOS/Linux, troubleshooting section
- [ ] No outdated commands or references to removed features
- [ ] A new user can follow INSTALL.md on a fresh machine and reach a working `rex doctor` output
- [ ] Typecheck passes

---

#### US-068: Simplify installer to "click install, it works"
**Description:** As a user, I want the install script to handle everything (venv, deps, config) in one command.

**Acceptance Criteria:**
- [ ] `install.py` (or `install.ps1` on Windows, `install.sh` on Linux/macOS) creates venv, installs deps, creates default config
- [ ] Script is idempotent (safe to run twice)
- [ ] On failure, script prints the exact error and suggests a fix
- [ ] After install, `rex doctor` passes all checks
- [ ] Works on Windows 11, macOS, Linux
- [ ] Typecheck passes

---

#### US-069: Ensure CLI/UI feature parity
**Description:** As a user, I want every feature available in the CLI to also be accessible in the UI, and vice versa.

**Acceptance Criteria:**
- [ ] Audit of CLI commands vs dashboard pages; gaps documented
- [ ] Each CLI-only feature gets a corresponding dashboard UI element (or API endpoint)
- [ ] Each UI-only feature gets a corresponding CLI command
- [ ] Gap list is zero at completion
- [ ] Typecheck passes

---

### PHASE 9 -- DEV + DEBUG SYSTEMS

---

#### US-070: Add structured logging system
**Description:** As a developer, I want structured JSON logging so that logs are parseable and filterable.

**Acceptance Criteria:**
- [ ] `rex/logging_config.py` configures structured JSON logging (using `python-json-logger` or similar)
- [ ] Each log entry includes: `timestamp`, `level`, `module`, `message`, `extra` (dict)
- [ ] Console output remains human-readable; file output is JSON
- [ ] Log file: `logs/rex.log` (rotated at 10MB, keep 5)
- [ ] All existing `logging.info/warning/error` calls continue to work
- [ ] Unit test confirms JSON log format in file output
- [ ] Typecheck passes

---

#### US-071: Add debug mode toggle
**Description:** As a developer, I want a `--debug` flag that enables verbose output for troubleshooting.

**Acceptance Criteria:**
- [ ] `rex --debug` sets log level to DEBUG across all modules
- [ ] Debug mode prints: config values (redacted secrets), loaded integrations, model info
- [ ] `rex doctor --debug` includes additional diagnostic info
- [ ] Config field: `debug_mode` in `AppConfig` (can also be set via env var `REX_DEBUG=1`)
- [ ] Typecheck passes

---

#### US-072: Stabilize flaky tests
**Description:** As a developer, I want all tests to pass reliably so that CI is trustworthy.

**Acceptance Criteria:**
- [ ] Run `pytest -q` 5 times; all runs produce the same pass/fail result
- [ ] Any test that depends on timing uses mocked time or generous tolerances
- [ ] Any test that depends on network uses mocked HTTP
- [ ] Any test that depends on filesystem uses `tmp_path` fixture
- [ ] No test is marked `@pytest.mark.skip` without a linked issue
- [ ] Typecheck passes

---

#### US-073: Enforce CI (tests + lint must pass)
**Description:** As a developer, I want CI to block merges if tests or lint fail.

**Acceptance Criteria:**
- [ ] GitHub Actions workflow runs: `pytest -q`, `ruff check`, `black --check`
- [ ] Workflow triggers on: push to `master`, pull request to `master`
- [ ] Branch protection rule on `master` requires CI to pass
- [ ] Workflow runs on Python 3.11, Ubuntu latest
- [ ] `mypy` check included (non-blocking warning for now)
- [ ] Typecheck passes

---

## Non-Goals

- No mobile app (Telegram covers mobile interaction for now)
- No multi-language UI (English only for this cycle)
- No voice assistant marketplace or third-party skill system
- No cloud-hosted deployment (local-first only)
- No real-time video or camera integration
- No smart home protocols beyond Home Assistant (no direct Zigbee/Z-Wave)
- No billing or payment system for cloud LLM usage
- No automatic priority assignment based on ML models (pattern detection is rule-based)

---

## Technical Considerations

- **Existing components to reuse:** `rex/ha_bridge.py` (HA integration base), `rex/tool_catalog.py` (tool registry), `rex/dashboard_store.py` (SQLite persistence), `rex/dashboard/sse.py` (real-time push), `rex/notifications/` (notification infrastructure), `rex/computers/` (agent server base)
- **Config split:** Secrets in `.env`, runtime config in `config/rex_config.json` (per CLAUDE.md)
- **Lazy imports:** All heavy ML imports (whisper, XTTS, transformers) must use `find_spec()` before `import_module()` (per learned rules)
- **Windows compatibility:** All file paths must use `pathlib.Path`; no hardcoded `/` separators
- **Branding:** Product name is "AskRex Assistant"; CLI is `rex`; see `docs/BRANDING.md` for banned names

---

## Implementation Priority

The recommended implementation order based on the user's priority list:

1. **Bridge system** (US-001 through US-006, US-015) -- unblocks all GUI functionality
2. **Voice loop** (US-007 through US-010, US-016 through US-020, US-074 through US-078) -- unblocks core voice experience
3. **Dependencies** (US-011) -- unblocks TTS
4. **Home Assistant** (US-021 through US-031, US-059) -- primary feature focus
5. **Context + aliases** (US-023 through US-025) -- makes HA usable
6. **Feedback + status** (US-033, US-062) -- makes Rex feel responsive
7. **Everything else** in phase order

---

## Issue-to-Story Mapping

| Issue | Stories |
|-------|---------|
| ISSUE-001 | US-001, US-002, US-003 |
| ISSUE-002 | US-004 |
| ISSUE-003 | US-005, US-006 |
| ISSUE-004 | US-007, US-008, US-009, US-010 |
| ISSUE-005 | US-011 |
| ISSUE-006 | US-012 |
| ISSUE-007 | US-013, US-014 |
| ISSUE-008 | US-015 |
| ISSUE-009 | US-070 |
| ISSUE-013 | US-047, US-048 |
| ISSUE-015 | US-049 |
| ISSUE-016 | US-039, US-040 |
| ISSUE-017 | US-041 |
| ISSUE-018 | US-042 |
| ISSUE-019 | US-021, US-022 |
| ISSUE-023 | US-066 |
| ISSUE-024 | US-066 |
| ISSUE-025 | US-023, US-024 |
| ISSUE-026 | US-025 |
| ISSUE-027 | US-026, US-027 |
| ISSUE-028 | US-032 |
| ISSUE-029 | US-033, US-034 |
| ISSUE-030 | US-035, US-036 |
| ISSUE-032 | US-057 |
| ISSUE-033 | US-058 |
| ISSUE-034 | US-065 |
| ISSUE-035 | US-050 |
| ISSUE-036 | US-051 |
| ISSUE-037 | US-043 |
| ISSUE-038 | US-044 |
| ISSUE-039 | US-045 |
| ISSUE-040 | US-046 |
| ISSUE-041 | US-028 |
| ISSUE-042 | US-029 |
| ISSUE-043 | US-030 |
| ISSUE-044 | US-031 |
| ISSUE-045 | US-037 |
| ISSUE-046 | US-038 |
| ISSUE-047 | US-052 |
| ISSUE-048 | US-053, US-054 |
| ISSUE-049 | US-055 |
| ISSUE-050 | US-056 |
| ISSUE-051 | US-059 |
| ISSUE-052 | US-060 |
| ISSUE-053 | US-061 |
| ISSUE-054 | US-062 |
| ISSUE-055 | US-063 |
| ISSUE-056 | US-064 |
| ISSUE-057 | US-067 |
| ISSUE-058 | US-068 |
| ISSUE-059 | US-069 |
| ISSUE-060 | US-071 |
| ISSUE-061 | US-072 |
| ISSUE-062 | US-073 |
| ISSUE-063 | US-016 |
| ISSUE-064 | US-017 |
| ISSUE-065 | US-018 |
| ISSUE-066 | US-019, US-020 |
| ISSUE-067 | US-074 |
| ISSUE-068 | US-075 |
| ISSUE-069 | US-076 |
| ISSUE-070 | US-077 |
| ISSUE-071 | US-078 |
