# PRD: AskRex Stability and Completeness

> **Codex/Ralph task selection rule**
> A "task" means one full User Story (US-###), not an individual checkbox line.
> Choose the first US-### that contains any unchecked acceptance criteria `[ ]`.

## Introduction

AskRex Assistant has accumulated 44 tracked issues spanning critical runtime blockers, broken GUI features, incomplete setup flows, and missing control surfaces. This PRD converts those issues into dependency-ordered, single-context-window user stories so they can be executed sequentially by an AI implementation loop (Ralph) or a human developer.

The goal is to take AskRex from "demo with known breakage" to "reliably boots, voice works, GUI is a complete control surface for all configured integrations."

## Goals

- Every CLI entry point (`rex whoami`, `rex chat`, `rex identify`) runs without crash on a clean install
- XTTS voice output initializes successfully under PyTorch 2.6
- All Electron bridge scripts resolve to real, working Python scripts
- GUI text chat produces streaming responses end-to-end
- Voice config has a single source of truth with no duplicate/conflicting sections
- Every GUI settings page is wired to real backend state (no placeholder data, no dead links)
- Scaffolding features (memory, autonomy, planning) have minimal viable implementations

## User Stories

---

### Phase 1: Critical Blockers

---

### US-301: Graceful default profile creation on first run
**Description:** As a user running Rex for the first time, I want the app to create a usable default profile automatically so that CLI commands and GUI flows don't crash on missing `profiles/default.json`.

**Acceptance Criteria:**
- [x] `rex/profile_manager.py` (or equivalent loader) checks for `profiles/default.json` at startup
- [x] If missing, copies `profiles/default.example.json` to `profiles/default.json` (or generates a minimal valid profile from `profiles/profile.schema.json`)
- [x] `python -m rex whoami` succeeds on a fresh clone with no manual profile setup
- [x] `python -m rex chat` starts without profile-related crash
- [x] `python -m rex identify --user james` does not crash if `james.json` is absent (warns and falls back to default)
- [x] Settings > Users page loads without error when only the auto-generated default profile exists
- [x] Existing `profiles/default.json` is never overwritten if it already exists
- [x] Typecheck passes
- [x] Tests pass (`pytest tests/ -q -k profile`)

---

### US-302: XTTS PyTorch 2.6 safe-globals allowlist
**Description:** As a developer, I need all required XTTS classes allowlisted for `torch.load()` under PyTorch 2.6's `weights_only=True` default so that XTTS voice output initializes without crashing.

**Acceptance Criteria:**
- [x] Identify every class `torch.load()` encounters when loading an XTTS checkpoint (at minimum: `XttsConfig`, `XttsAudioConfig`, and any referenced dataclasses/namedtuples)
- [x] Add all identified classes to `torch.serialization.add_safe_globals()` in the XTTS init path (likely `rex/tts_utils.py` or `patch_tts_torch_load.py`)
- [x] The allowlist call happens BEFORE `torch.load()` is invoked (not after a failed attempt)
- [x] XTTS initializes successfully: `python -c "from rex.tts_utils import get_tts_engine; get_tts_engine('xtts')"` exits 0
- [x] If XTTS dependencies are not installed, the import fails gracefully with a clear message (no raw `ModuleNotFoundError` traceback)
- [x] Typecheck passes
- [x] Tests pass (`pytest tests/ -q -k tts`)

---

### US-303: Centralized bridge path resolver for Electron
**Description:** As a developer, I need a single bridge path resolver so that all Electron-to-Python bridge calls use correct, validated script paths instead of hardcoded or outdated ones.

**Acceptance Criteria:**
- [x] Create or update a resolver module (e.g., `gui/src/main/bridgeResolver.ts`) that maps bridge names to their Python script paths relative to repo root
- [x] The resolver validates that each target script exists at launch time and logs an error with the expected path if missing
- [x] All Electron `spawn`/`exec` calls for bridge scripts route through this resolver (no inline path strings remain)
- [x] The following bridges resolve correctly: `rex_tasks_bridge.py`, `rex_reminders_bridge.py`, `rex_shopping_list_bridge.py`, `rex_speaker_bridge.py`, `rex_chat_stream_bridge.py`, `rex_voices_bridge.py`, `rex_voice_enrollment_bridge.py`, `rex_voice_sample_bridge.py`, `rex_wakeword_list_bridge.py`, `rex_wakeword_train_bridge.py`, `rex_stt_bridge.py`, `rex_memories_bridge.py`
- [x] Typecheck passes (`npx tsc --noEmit` in `gui/`)
- [x] Verify changes work: launch the Electron app and confirm Tasks, Reminders, and Shopping List pages load without "bridge exited" errors

---

### US-304: Fix GUI text chat streaming bridge
**Description:** As a user, I want to type a message in the GUI chat and receive a streamed response so that text conversation works end-to-end.

**Acceptance Criteria:**
- [x] `rex_chat_stream_bridge.py` is importable and runs standalone: `python rex_chat_stream_bridge.py --help` exits 0
- [x] The bridge uses `Assistant.generate_reply()` (not a direct LLM call)
- [x] The Electron chat page spawns the bridge via the centralized resolver (US-303)
- [x] A typed message in the GUI produces a streaming response displayed token-by-token
- [x] If the backend is unreachable or config is invalid, the GUI shows a user-visible error (not just "exited with code 2")
- [x] Typecheck passes (both Python and TS)
- [ ] Verify changes work in Electron

---

### US-305: OpenClaw voice backend clean disable
**Description:** As a developer, I need `openclaw.use_voice_backend = false` in config to fully bypass `VoiceBridge` at runtime so the local voice loop is the only active path when OpenClaw is disabled.

**Acceptance Criteria:**
- [x] When `config.use_openclaw_voice_backend` is `false`, no code path imports or instantiates `VoiceBridge`
- [x] `rex/voice_loop.py` -> `build_voice_loop` uses `Assistant` directly when the flag is off
- [x] `python rex_loop.py` with the flag off does not log any OpenClaw-related connection attempts or async errors
- [x] When the flag is `true` and the gateway is unreachable, startup fails with a clear error message (not a hang or cryptic traceback)
- [x] Typecheck passes
- [x] Tests pass (`pytest tests/ -q -k "voice_loop or openclaw"`)

---

### Phase 2: High Priority Functionality Gaps

---

### US-306: Unify wake-word config into a single section
**Description:** As a developer, I want one canonical `wakeword` config section so that wake-word behavior is predictable and there are no conflicting keys.

**Acceptance Criteria:**
- [x] `config/rex_config.json` has exactly one wake-word section (canonical key: `wakeword`)
- [x] Any references to the old `wake_word` key in Python code are migrated to read from `wakeword`
- [x] Config loading detects the old `wake_word` key, copies values into `wakeword`, removes the old key, and logs a deprecation notice
- [x] `config/rex_config.schema.json` is updated to reflect the single key
- [x] Typecheck passes
- [x] Tests pass (`pytest tests/ -q -k "wakeword or wake"`)

---

### US-307: Remove legacy REX_WAKEWORD_THRESHOLD env var
**Description:** As a user, I don't want misleading deprecation warnings about `REX_WAKEWORD_THRESHOLD` on every startup.

**Acceptance Criteria:**
- [x] All references to `REX_WAKEWORD_THRESHOLD` in Python code are removed
- [x] The config schema and docs reference only the JSON config path for threshold
- [x] Startup produces no warning about `REX_WAKEWORD_THRESHOLD` even if the env var is still set
- [x] Typecheck passes
- [x] Tests pass

---

### US-308: TTS voice preview in Settings
**Description:** As a user, I want to click "Preview" next to a voice in Settings > Voice and hear a short sample so I can choose the right voice.

**Acceptance Criteria:**
- [x] The Settings > Voice page has a working "Preview" button for each listed voice
- [x] Clicking Preview calls `rex_voice_sample_bridge.py` (via the centralized resolver) with the selected voice ID
- [x] The bridge generates a short TTS clip ("Hello, I'm your Rex assistant") and plays it through system audio
- [x] If TTS is not configured or fails, the GUI shows an inline error (not a silent failure)
- [x] Typecheck passes (Python + TS)
- [x] Verify changes work in Electron

---

### US-309: Voice enrollment guided UX
**Description:** As a user enrolling my voice, I want to see a phrase to read, progress indication, and validation feedback so the process is usable.

**Acceptance Criteria:**
- [x] The voice enrollment page displays a specific prompt phrase for the user to read aloud
- [x] During recording, a visual indicator confirms audio is being captured
- [x] After recording, the UI shows pass/fail feedback: sufficient audio length, acceptable volume level
- [x] If the sample is too short or too quiet, the user is prompted to re-record with a specific reason
- [x] The enrollment bridge (`rex_voice_enrollment_bridge.py`) stores the sample in the correct voice identity directory
- [x] Typecheck passes (Python + TS)
- [x] Verify changes work in Electron

---

### US-310: Fix wake word "Play sample" to play actual wake word audio
**Description:** As a user, I want the "Play sample" button to play a representative clip of the selected wake word, not unrelated speech.

**Acceptance Criteria:**
- [x] The Play Sample button plays a short audio clip demonstrating the selected wake word pronunciation
- [x] If no sample audio exists for a custom wake word, the button is disabled with a tooltip explaining why
- [x] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-311: Fix Settings > Audio Output page
**Description:** As a user, I want to select my audio output device and test it from Settings > Audio Output.

**Acceptance Criteria:**
- [x] The Audio Output page loads without bridge-path errors
- [x] `rex_speaker_bridge.py` is called via the centralized resolver and returns available output devices
- [x] Selecting a device updates config
- [x] The "Test" button plays a short test tone through the selected device
- [x] Typecheck passes (Python + TS)
- [ ] Verify changes work in Electron

---

### US-312: Surface all existing Rex settings in the GUI
**Description:** As a user, I want the GUI Settings to expose every Rex setting, including Telegram setup, so the GUI is a complete control surface.

**Acceptance Criteria:**
- [x] Audit `config/rex_config.schema.json` and `config/rex_config.json` for all user-facing keys
- [x] Each key has a corresponding input in the appropriate Settings tab
- [x] Telegram bot token and chat ID fields are present in Settings > Integrations (or a Telegram sub-page)
- [x] Saving any new field writes to `config/rex_config.json` correctly
- [x] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-313: Home Assistant device status page
**Description:** As a user with Home Assistant configured, I want a dashboard page showing HA device states.

**Acceptance Criteria:**
- [x] A "Home Assistant" page exists in the GUI sidebar
- [x] If HA is not configured, the page shows a message with a link to the correct Settings page (not Settings > General)
- [x] If HA is configured, the page fetches and displays device states (entity name, state, last updated)
- [x] The page has a manual refresh button
- [x] Typecheck passes (Python + TS)
- [ ] Verify changes work in Electron

---

### US-314: Wire up the Integrations page
**Description:** As a user, I want the Integrations page to show real status and link to the correct config pages.

**Acceptance Criteria:**
- [x] The page queries the backend for configured integrations (email, calendar, SMS, MQTT, HA, Telegram, search)
- [x] Each integration shows: name, status (configured/not configured), and a "Configure" link to the correct Settings sub-page
- [x] "No integrations found" only appears when genuinely none are configured
- [x] "No capabilities found" section is removed or populated from `rex/capabilities/`
- [x] The "Configure" link for HA goes to the HA settings page (not Settings > General)
- [x] Typecheck passes
- [ ] Verify changes work in Electron

---

### Phase 3: Medium Priority Product and UX

---

### US-315: Replace placeholder data in Calendar, Email, and SMS pages
**Description:** As a user, I want these pages to show real data or a clear "not configured" state, not fake content.

**Acceptance Criteria:**
- [ ] Calendar page calls `rex/calendar_service.py` and displays real events (or "No calendar configured")
- [ ] Email page calls `rex/email_service.py` and displays real inbox items (or empty state)
- [ ] SMS page calls the messaging backend and displays real threads (or empty state)
- [ ] No hardcoded fake names, dates, or messages remain in GUI source for these pages
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-316: Review and update Beta labels on Email and SMS
**Description:** As a product owner, I want Beta labels to accurately reflect feature maturity.

**Acceptance Criteria:**
- [ ] If Email and SMS are still beta-quality, keep the label but add a tooltip explaining what "Beta" means
- [ ] If stable, remove the Beta label
- [ ] Decision documented in a code comment or changelog entry
- [ ] Typecheck passes

---

### US-317: Fix "Configure Home Assistant" link routing
**Description:** As a user, I want the Home page HA link to go to the HA configuration page, not Settings > General.

**Acceptance Criteria:**
- [ ] The link navigates to the HA configuration page (per US-314)
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-318: Populate timezone dropdown with full IANA list
**Description:** As a user, I want to select my timezone from a complete list, not just `America/Chicago`.

**Acceptance Criteria:**
- [ ] The timezone dropdown is populated from a standard IANA timezone list
- [ ] The dropdown supports type-ahead filtering
- [ ] The currently configured timezone is pre-selected
- [ ] Saving updates `config/rex_config.json`
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-319: Add folder picker for Allowed File Roots
**Description:** As a user, I want a folder picker instead of raw text input for allowed file roots.

**Acceptance Criteria:**
- [ ] An "Add Folder" button opens an Electron native folder picker dialog
- [ ] Selected folders are added to the list and persisted to config
- [ ] Existing raw text input is preserved as fallback
- [ ] Each listed folder has a "Remove" button
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-320: Actionable notifications with instructions and links
**Description:** As a user, when Rex shows a notification requiring action, I want it to include what to do and where to go.

**Acceptance Criteria:**
- [ ] `rex/notification.py` supports `action_url` and `action_label` fields on notifications
- [ ] The GUI notification component renders action links when present
- [ ] At least three existing notification types include actionable links (e.g., "TTS not configured", "Profile missing", "Integration error")
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-321: Add "Reset to defaults" option in Settings
**Description:** As a user, I want a way to reset Rex to factory defaults when troubleshooting.

**Acceptance Criteria:**
- [ ] Settings > System has a "Reset to Defaults" button
- [ ] Clicking shows a confirmation dialog explaining what will be reset
- [ ] On confirm, replaces `config/rex_config.json` with `config/rex_config.example.json`
- [ ] Does NOT delete user profiles, voice samples, or `.env` secrets
- [ ] After reset, the app reloads cleanly
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-322: Add AskRex branding assets to GUI and desktop surfaces
**Description:** As a user, I want consistent AskRex branding across the GUI, system tray, and window title.

**Acceptance Criteria:**
- [ ] GUI window title uses the canonical product name from `docs/BRANDING.md`
- [ ] System tray icon uses the official AskRex icon asset
- [ ] The GUI sidebar or header shows the AskRex logo
- [ ] If branding assets don't exist yet, placeholder assets are created at the correct paths with TODO comments
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

### US-323: Log rotation and session separation
**Description:** As a developer, I want logs to separate sessions and rotate old entries so stale timestamps don't cause confusion.

**Acceptance Criteria:**
- [ ] Logging config uses `RotatingFileHandler` (5 MB max, 3 backups)
- [ ] Each startup writes a session-start marker: `=== Rex session started at <ISO timestamp> ===`
- [ ] Old entries are preserved in rotated files, not mixed with current session
- [ ] Typecheck passes
- [ ] Tests pass

---

### Phase 4: Scaffolding Completion

---

### US-324: Per-user memory -- minimal viable read/write
**Description:** As a user, I want Rex to remember facts about me across sessions so conversations feel personalized.

**Acceptance Criteria:**
- [ ] Memory system supports `store(user, key, value)` and `recall(user, key) -> value`
- [ ] Stored facts persist to disk (JSON file per user in `Memory/`)
- [ ] `Assistant.generate_reply()` injects recalled facts into the system prompt when relevant
- [ ] CLI test: `python -m rex remember "My dog is named Max"` then asking "What's my dog's name?" in chat returns "Max"
- [ ] Typecheck passes
- [ ] Tests pass (`pytest tests/ -q -k memory`)

---

### US-325: Autonomous workflows -- minimal scheduled task runner
**Description:** As a user, I want Rex to execute simple scheduled tasks so autonomous workflows have a foundation.

**Acceptance Criteria:**
- [ ] `rex/workflow_runner.py` reads task definitions from config
- [ ] Each task has: name, schedule (cron or interval), action (Rex command string)
- [ ] The runner executes due tasks when the voice loop or daemon is running
- [ ] At least one example task is included (daily weather briefing)
- [ ] Tasks that fail log the error and do not block subsequent tasks
- [ ] Typecheck passes
- [ ] Tests pass (`pytest tests/ -q -k workflow`)

---

### US-326: Smart planning -- minimal plan-and-execute skeleton
**Description:** As a developer, I want a basic plan-and-execute framework so multi-step user requests can be decomposed.

**Acceptance Criteria:**
- [ ] `rex/planner.py` exposes `create_plan(goal: str) -> list[Step]` and `execute_plan(steps: list[Step]) -> Result`
- [ ] `Step` is a dataclass with `description`, `tool` (optional), and `status`
- [ ] `create_plan` calls the LLM to decompose a goal into steps
- [ ] `execute_plan` iterates steps, calling tools where specified, updating status
- [ ] At least one integration test demonstrates plan creation and execution
- [ ] Typecheck passes
- [ ] Tests pass (`pytest tests/ -q -k planner`)

---

### Phase 5: Verification

---

### US-327: End-to-end smoke test -- CLI boot and chat
**Description:** As a developer, I want a smoke test verifying Rex boots and handles a chat round-trip.

**Acceptance Criteria:**
- [ ] A pytest fixture runs `python -m rex doctor` -> exit 0
- [ ] Then runs `echo "hello" | python -m rex chat --no-tts` -> non-empty output, exit 0
- [ ] The test is runnable in CI (no GPU, no mic required)
- [ ] Typecheck passes
- [ ] Tests pass

---

### US-328: End-to-end verification -- GUI launch and backend connection
**Description:** As a developer, I want to verify the Electron GUI launches, connects to Flask, and renders without crash or login loop.

**Acceptance Criteria:**
- [ ] A manual test script documents exact steps: launch command, expected first screen, backend connection verification
- [ ] The home page renders within 10 seconds of launch
- [ ] No JavaScript console errors related to missing bridges or failed API calls on the home page
- [ ] If auth is required, the login flow completes without looping
- [ ] Typecheck passes
- [ ] Verify changes work in Electron

---

## Non-Goals

- New feature development beyond completing existing scaffolding
- Mobile app or cloud deployment
- Redesign of the GUI framework (Electron + React stays)
- Migration away from Flask
- New LLM provider integrations
- Full-featured memory, autonomy, or planning systems (minimal viable only in this PRD)

## Technical Considerations

- **US-301 (profile loading) is a dependency for nearly everything.** Must be the first story executed.
- **US-302 (XTTS allowlist) can run in parallel with US-301** since it touches different code paths.
- **US-303 (bridge resolver) is a dependency for US-304, US-308, US-309, US-310, US-311.** Must complete before those.
- **US-306 and US-307 (config cleanup) should be batched** to avoid multiple schema migrations.
- **GUI stories (US-312 through US-322) can largely run in parallel** once the bridge resolver is in place.
- The Electron GUI uses Vite + React + TypeScript (`gui/`). Python bridges are standalone scripts at repo root (`rex_*_bridge.py`).
- Config lives in `config/rex_config.json` with schema at `config/rex_config.schema.json`.
- Profiles live in `profiles/` with schema at `profiles/profile.schema.json`.
- Story IDs start at US-301 to avoid collision with the existing PRD (US-001 through US-220).
