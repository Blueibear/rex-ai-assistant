# PRD: Rex AI Assistant — Master Next-Cycle

> **Codex/Ralph task selection rule**
> A "task" means one full User Story (US-###), not an individual checkbox line.
> Choose the first US-### that contains any unchecked acceptance criteria `[ ]`.
> Complete the full story in one iteration. If it cannot be completed in one iteration, split it first.
> Only mark acceptance criteria `[x]` when the full story is done and tests pass.

---

## Introduction

This PRD covers the next full development cycle for Rex AI Assistant. It consolidates:

1. **CI fix work** — resolve all remaining failing checks inherited from the OpenClaw migration and stale dashboard module retirement.
2. **Repo hygiene** — normalize branding, archive stale housekeeping files, eliminate root-directory clutter, establish a single documentation hierarchy, and scrub tracked personal/sample data.
3. **Major feature development** — multi-model routing, multi-user voice recognition, skill training, automatic tool dispatch, local file and Windows system access, a fully realized UI, latency reduction, shopping list, multi-email per user, smart speaker integration, voice/wake-word configuration via UI, log viewer, and inbound/outbound phone call handling.

Stories are ordered by dependency. Each story is sized to complete in one Ralph loop context window (~10 min of AI work). Earlier stories must not depend on later ones.

> **Note on uploaded `index.js`:** The files attached to this PRD request are from the Kapture MCP browser-automation server — they are not Rex source files and do not contribute failing checks. Rex's CI failures are Python-stack only and are fully catalogued in `PRD-ci-fix-pr216.md`. The stories below supersede that document.

---

## Goals

- Green CI on all checks (ruff, black, mypy, pytest) with zero regressions
- Root directory reduced to ≤30 files; all stale housekeeping docs archived
- Single authoritative docs tree under `docs/`
- No committed personal data in default repo state
- Multi-model routing that auto-selects the right model per task type
- Multi-user voice recognition with automatic profile matching (no verbal identification required)
- Natural-language and script-based skill training pipeline
- Automatic tool dispatch — Rex selects and invokes tools without user prompting
- Local file read/write and Windows diagnostics/settings capability
- Complete, polished GUI exposing every Rex capability
- Measurably reduced perceived latency
- Shared shopping list accessible from UI and mobile
- Per-user multi-email account isolation
- Smart speaker (Sonos, Bose, etc.) integration for TTS and wake word
- Voice selection via UI dropdown with sample playback
- Personalized voice creation via UI file upload
- Wake word selection and custom wake word training via UI
- Full log viewer in UI
- Phone number linking with inbound handling, message taking, and outbound calling

---

## Non-Goals

- No multi-tenant cloud hosting or SaaS billing
- No OAuth for calendar in this cycle (ICS only)
- No GPU-specific CI runners
- No mobile native apps (PWA is sufficient for mobile shopping list)
- No support for Python < 3.11
- No third-party plugin marketplace

---

---

# PHASE 0 — CI Fixes (Unblock All Checks)

---

### US-CI-001: Fix ruff lint — suppress phantom deleted-file errors

**Description:** As a developer, I want ruff to run only against files that actually exist so that CI does not fail on E902 "No such file or directory" errors for files deleted within the PR.

**Acceptance Criteria:**
- [x] CI lint script updated to filter `git diff --name-only` output through `[ -f "$f" ]` before passing to ruff
- [x] `ruff check --fix` and `ruff check` both exit 0 on files that exist in the working tree
- [x] No new ruff ignore entries added for real code violations
- [x] Typecheck passes

---

### US-CI-002: Fix ruff lint — sort imports in two test files

**Description:** As a developer, I want import blocks sorted in the two files that trigger I001 so CI is fully green on lint.

**Acceptance Criteria:**
- [x] `tests/test_us025_planner_task_execution.py` import block sorted per ruff I001
- [x] `tests/test_us029_event_triggers.py` import block sorted per ruff I001
- [x] `ruff check` exits 0 on both files
- [x] `black --check` exits 0 on both files
- [x] Typecheck passes

---

### US-CI-003: Fix mypy — remove stale type: ignore comments

**Description:** As a developer, I want unnecessary `# type: ignore` comments removed from `rex/wakeword/embedding.py` and `rex/compat/transformers_shims.py` so mypy reports no unused-ignore errors.

**Acceptance Criteria:**
- [x] `# type: ignore[assignment]` removed from `rex/wakeword/embedding.py` line 20
- [x] `# type: ignore[attr-defined]` removed from `rex/compat/transformers_shims.py` line 76
- [x] `mypy rex --ignore-missing-imports` reports 0 errors on these two files
- [x] Existing tests still pass
- [x] Typecheck passes

---

### US-CI-004: Fix mypy — correct overloaded subscribe/publish signatures in event_bus.py

**Description:** As a developer, I want the `subscribe` and `publish` overload implementations in `rex/openclaw/event_bus.py` to match their declared signatures so mypy reports no misc errors.

**Acceptance Criteria:**
- [x] `subscribe` overload implementation signature at line 86 (approx) matches all `@overload` variants
- [x] `publish` overload implementation signature at line 151 (approx) matches all `@overload` variants
- [x] `mypy rex --ignore-missing-imports` reports 0 errors on `rex/openclaw/event_bus.py`
- [x] Existing event_bus tests pass
- [x] Typecheck passes

---

### US-CI-005: Fix mypy — no-any-return in event_bridge.py and attr-defined in browser_core.py

**Description:** As a developer, I want `rex/openclaw/event_bridge.py` and `rex/openclaw/browser_core.py` to carry correct type annotations so mypy reports zero errors in both files.

**Acceptance Criteria:**
- [x] `event_bridge.py` line 95: return type narrowed or cast so it does not return `Any` where `Callable[[], None] | None` is expected
- [x] `browser_core.py` lines 92, 97, 100-102, 248-252: untyped instance variables annotated in `__init__` so attr-defined errors are gone
- [x] `mypy rex --ignore-missing-imports` reports 0 errors on both files
- [x] Existing tests pass
- [x] Typecheck passes

---

### US-CI-006: Fix mypy — no-redef and assignment error in tool_executor.py

**Description:** As a developer, I want `rex/openclaw/tool_executor.py` to have no variable redefinition or incompatible assignment errors so mypy is fully clean.

**Acceptance Criteria:**
- [x] Line 274 `result` variable renamed or conditional branch restructured to eliminate no-redef error
- [x] Line 948 assignment type narrowed or cast to eliminate incompatible-types error
- [x] `mypy rex --ignore-missing-imports` exits 0 on `rex/openclaw/tool_executor.py`
- [x] Existing tool_executor tests pass
- [x] Typecheck passes

---

### US-CI-007: Retire dashboard test files that reference deleted static assets

**Description:** As a developer, I want the 11 dashboard test files that read from the deleted `rex/dashboard/static/` and `rex/dashboard/templates/` directories to be skipped so pytest reports no failures from them.

**Acceptance Criteria:**
- [x] The following files each receive a module-level `pytest.mark.skip(reason="rex/dashboard retired")` decorator: `test_us149_gui_shell.py`, `test_us150_design_system.py`, `test_us151_nav_state.py`, `test_us152_chat_message_list.py`, `test_us153_chat_input.py`, `test_us157_voice_waveform.py`, `test_us161_schedule_coming_up.py`, `test_us163_overview_quick_actions.py`, `test_us164_hover_focus_states.py`, `test_us165_loading_error_states.py`, `test_us166_accessibility.py`
- [x] `pytest -q` reports 0 failed tests (was 261 failed)
- [x] Test count of previously passing tests does not decrease
- [x] Typecheck passes

---

### US-CI-008: Fix voice_loop patch target in three test files

**Description:** As a developer, I want the three test files that monkeypatch `voice_loop.load_plugins` to target the correct name `_load_plugins_impl` so the patches actually apply and tests pass.

**Acceptance Criteria:**
- [x] All `mock.patch("…voice_loop.load_plugins")` calls in affected test files updated to `mock.patch("…voice_loop._load_plugins_impl")`
- [x] Previously failing tests in those 3 files now pass
- [x] `pytest -q` exits 0 on those files
- [x] Typecheck passes

---

### US-CI-009: Fix tomllib import fallback in test_us140_full_extra.py

**Description:** As a developer, I want `test_us140_full_extra.py` to import `tomllib` with a Python 3.10 fallback so the test collection never errors on older interpreters.

**Acceptance Criteria:**
- [x] File updated to: `try: import tomllib except ImportError: import tomli as tomllib`
- [x] `pyproject.toml` or test requirements includes `tomli` as a test dependency
- [x] `pytest --collect-only tests/test_us140_full_extra.py` exits 0
- [x] Typecheck passes

---

### US-CI-010: Fix missing rex.contracts.browser import in browser bridge test

**Description:** As a developer, I want `tests/test_openclaw_browser_bridge.py::test_satisfies_protocol` to either import from the correct module path or be skipped with a clear reason so pytest collects cleanly.

**Acceptance Criteria:**
- [x] Import updated to the correct current module path, OR test marked `pytest.mark.skip(reason="rex.contracts.browser retired in Phase 7")`
- [x] `pytest -q tests/test_openclaw_browser_bridge.py` exits 0
- [x] Typecheck passes

---

### US-CI-011: Fix retired rex/dashboard/routes.py reference in test_us174

**Description:** As a developer, I want `tests/test_us174.py::test_chat_mode_not_affected` to not read from the deleted `rex/dashboard/routes.py` so pytest does not fail on file-not-found.

**Acceptance Criteria:**
- [x] Test updated to either target the correct current routes module or marked `pytest.mark.skip(reason="dashboard routes retired")`
- [x] `pytest -q tests/test_us174.py` exits 0
- [x] Typecheck passes

---

---

# PHASE 1 — Repo Hygiene

---

### US-RH-001: Archive all stale VERIFICATION_REPORT files

**Description:** As a developer, I want all `VERIFICATION_REPORT_*.md` and `VERIFICATION_REPORT_*.txt` files in the repo root moved to `docs/archive/verification/` so the root is uncluttered.

**Acceptance Criteria:**
- [x] Directory `docs/archive/verification/` created
- [x] All 18+ `VERIFICATION_REPORT_*` files in repo root moved there via `git mv`
- [x] `docs/archive/verification/INDEX.md` created listing all moved files with a one-line description each
- [x] Root directory file count reduced by at least 18
- [x] `pytest -q` still passes (no test imports these paths)
- [x] Typecheck passes

---

### US-RH-002: Archive stale PRD files from root

**Description:** As a developer, I want all completed or superseded PRD files moved to `docs/archive/prd/` so only active planning documents remain visible at the root.

**Acceptance Criteria:**
- [x] Directory `docs/archive/prd/` created
- [x] The following moved via `git mv`: `PRD-ci-fix-pr216.md`, `PRD-complete.md`, `PRD-full-repo-audit.md`, `PRD-full-test-and-fix.md`, `PRD-gui-autonomy-integrations.md`, `PRD-openclaw-http-integration.md`, `PRD-openclaw-pivot-for-rex.md`, `PRD-repo-quality.md`, `PRD-voice-selector-and-fixes.md`, `PRD-complete-3_31.md`
- [x] `PRD-master-next-cycle.md` (this file) remains at root as the active PRD
- [x] `docs/archive/prd/INDEX.md` created with one-line description per archived PRD
- [x] Typecheck passes

---

### US-RH-003: Archive stale housekeeping and summary files from root

**Description:** As a developer, I want one-off housekeeping files that served a single session moved to `docs/archive/housekeeping/` so the root contains only files with ongoing operational value.

**Acceptance Criteria:**
- [x] Directory `docs/archive/housekeeping/` created
- [x] The following moved via `git mv`: `AGENTS.md`, `APP_ROADMAP.md`, `BACKLOG.md`, `BATCH_001_PROMPT.md`, `CHANGELOG_IMPROVEMENTS.md`, `CODEX_REPO_AUDIT.md`, `CODEX_REPO_AUDIT_ISSUES.json`, `COMPLETED_WORK_SUMMARY.md`, `FINAL_SUMMARY.txt`, `ROADMAP_BIBLE.md`, `STABILIZATION_REPORT.md`, `STABILIZATION_REPORT.txt`, `TEST_FIXES.md`
- [x] `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, `README.md`, `SECURITY_ADVISORY.md` remain at root
- [x] Root file count ≤ 35 after this story
- [x] Typecheck passes

---

### US-RH-004: Consolidate security audit files into docs/security/

**Description:** As a developer, I want all security-related audit and fix documents moved under `docs/security/` alongside the existing `docs/security/` contents so security posture history is in one place.

**Acceptance Criteria:**
- [x] `SECURITY_AUDIT_2026-01-08.md` moved to `docs/security/` via `git mv`
- [x] `SECURITY_FIX_SUMMARY.md` moved to `docs/security/` via `git mv`
- [x] `SECURITY_ADVISORY.md` at root retained (it is user-facing)
- [x] `docs/security/INDEX.md` updated or created to list all files
- [x] Typecheck passes

---

### US-RH-005: Normalize branding — Rex AI Assistant is the canonical product name

**Description:** As a developer, I want every reference to "REX", "rex-assistant", "rex_assistant", or informal variants in user-facing strings, README, and package metadata to use the canonical name "Rex AI Assistant" (display) / `rex-ai-assistant` (slug) / `rex` (CLI command) consistently.

**Acceptance Criteria:**
- [x] `pyproject.toml` `name` field is `rex-ai-assistant`
- [x] `pyproject.toml` `description` field uses "Rex AI Assistant"
- [x] `README.md` H1 heading is "Rex AI Assistant"
- [x] `rex/cli.py` version/help string references "Rex AI Assistant"
- [x] No inconsistent capitalizations (e.g. "REX AI", "rex assistant") in any top-level `.md` file
- [x] `grep -r "REX AI\b" docs/ README.md` returns 0 matches (except legitimate acronym uses)
- [x] Typecheck passes

---

### US-RH-006: Establish single documentation hierarchy under docs/

**Description:** As a developer, I want a top-level `docs/INDEX.md` that maps every documentation file in the repo so contributors can find any doc without guessing.

**Acceptance Criteria:**
- [x] `docs/INDEX.md` created listing all files under `docs/` organized by category (Architecture, Configuration, Integrations, API, Development, Archive)
- [x] Each entry has a one-line description
- [x] `README.md` links to `docs/INDEX.md` as the "Full Documentation" entry point
- [x] All file paths in `docs/INDEX.md` are validated (no 404 links to moved files)
- [x] Typecheck passes

---

### US-RH-007: Remove or replace committed personal/sample data

**Description:** As a developer, I want the `Memory/` directory to contain only template or clearly synthetic sample profiles so no real personal data is committed to the default repo state.

**Acceptance Criteria:**
- [x] All files under `Memory/alice/`, `Memory/cole/`, `Memory/james/`, `Memory/voice-user/` inspected for real PII (names, emails, phone numbers, addresses)
- [x] Any file containing real personal data is either deleted or replaced with a clearly synthetic placeholder (e.g. `alice@example.com`, fictional phone numbers)
- [x] `Memory/README.md` created explaining these are example profiles and that real user data should never be committed
- [x] `Rex_Settings_Reference.xlsx` checked for personal data; sanitized or moved to `docs/archive/`
- [x] `.gitignore` updated to include `Memory/*/` so real runtime memory is not accidentally committed
- [x] `pytest -q` still passes
- [x] Typecheck passes

---

---

# PHASE 2 — Multi-Model Routing

---

### US-MM-001: Add model routing config schema

**Description:** As a developer, I want `AppConfig` to support a `model_routing` block that maps task categories to model identifiers so routing rules are declared in config rather than hardcoded.

**Acceptance Criteria:**
- [x] `AppConfig` gains a `model_routing: ModelRoutingConfig` field (Pydantic model)
- [x] `ModelRoutingConfig` has fields: `default`, `coding`, `reasoning`, `search`, `vision`, `fast`, each accepting a string model identifier
- [x] `config/rex_config.example.json` updated with a `model_routing` block showing sensible defaults
- [x] `config/rex_config.json` schema docs updated
- [x] Existing config loads without error when `model_routing` is absent (all fields optional with defaults)
- [x] Typecheck passes

---

### US-MM-002: Implement model router — intent classification

**Description:** As a developer, I want a `ModelRouter` class that classifies an incoming user message into a task category so the correct model can be selected.

**Acceptance Criteria:**
- [x] `rex/model_router.py` created with `ModelRouter` class
- [x] `ModelRouter.classify(message: str) -> TaskCategory` method implemented
- [x] `TaskCategory` is a `StrEnum` with values: `coding`, `reasoning`, `search`, `vision`, `fast`, `default`
- [x] Classification uses keyword/pattern heuristics (no LLM call) for deterministic, zero-latency routing
- [x] Unit tests in `tests/test_model_router.py` cover: code-related messages → `coding`, complex multi-step → `reasoning`, web search intent → `search`, image description request → `vision`, simple factual → `fast`
- [x] Typecheck passes

---

### US-MM-003: Wire model router into Assistant.generate_reply()

**Description:** As a developer, I want `Assistant.generate_reply()` to automatically select the model from the routing config based on the user's message so different task types use the best available model.

**Acceptance Criteria:**
- [x] `Assistant.generate_reply()` calls `ModelRouter.classify(message)` before invoking the LLM
- [x] The classified category is used to look up the model identifier in `AppConfig.model_routing`
- [x] If the resolved model is not available (e.g. Ollama not running), falls back to `model_routing.default` with a logged warning
- [x] Model selection is logged at DEBUG level: `"model_router: classified as {category}, using {model}"`
- [x] Existing tests that mock `LanguageModel.generate` still pass
- [x] New tests in `tests/test_model_routing_integration.py` verify: coding message → coding model used, fallback on unavailable model
- [x] Typecheck passes

---

### US-MM-004: Add Ollama model availability check to router fallback

**Description:** As a developer, I want the model router to check Ollama availability at startup and cache the result so fallback decisions are fast and do not block each request.

**Acceptance Criteria:**
- [x] `ModelRouter` checks Ollama `/api/tags` at init time if any routing target is an Ollama model
- [x] Available Ollama model list cached; refreshed every 60 seconds via background thread
- [x] If a routed model is not in the available list, `ModelRouter.classify()` still returns the category but `resolve_model()` returns the fallback
- [x] No external network call made if all routing targets are OpenAI or local Transformers models
- [x] Typecheck passes

---

### US-MM-005: UI settings panel — model routing configuration

**Description:** As a user, I want to configure model routing rules through the UI so I can change which model handles each task type without editing JSON files.

**Acceptance Criteria:**
- [x] Settings panel (see US-UI-001 through US-UI-003) includes a "Model Routing" section
- [x] Each task category (`coding`, `reasoning`, `search`, `vision`, `fast`, `default`) has a dropdown or text input
- [x] Save button writes changes to `config/rex_config.json` via the config API
- [x] Changes take effect on next Assistant call without restarting Rex
- [x] Typecheck passes
- [x] Verify changes work in browser

---

---

# PHASE 3 — Multi-User Voice Recognition

---

### US-VID-001: Voice enrollment — capture and store speaker embeddings

**Description:** As a user, I want to enroll my voice with Rex so it can recognize me automatically without me having to say who I am.

**Acceptance Criteria:**
- [x] `rex/voice_identity/enrollment.py` (or equivalent) provides `enroll_user(user_id: str, audio_samples: list[np.ndarray]) -> None`
- [x] Speaker embeddings computed and stored in `Memory/{user_id}/voice_embedding.npy`
- [x] Enrollment requires at least 3 audio samples; raises `ValueError` if fewer provided
- [x] Existing `rex/voice_identity/` structure respected; no duplicate implementations
- [x] Unit tests cover: successful enrollment, insufficient samples error, embedding file created
- [x] Typecheck passes

---

### US-VID-002: Real-time speaker identification during STT pipeline

**Description:** As a developer, I want the STT pipeline to identify which enrolled user is speaking before passing the transcript to the Assistant so the correct user profile is active.

**Acceptance Criteria:**
- [x] `rex/voice_identity/identifier.py` provides `identify_speaker(audio: np.ndarray) -> str | None` returning the matched `user_id` or `None` if no match
- [x] Cosine similarity threshold configurable via `AppConfig` (default 0.75)
- [x] Identification result injected into `Assistant.generate_reply()` context as `active_user_id`
- [x] If speaker unrecognized, Rex responds with a default profile and does not crash
- [x] Unit tests cover: enrolled user recognized, unrecognized speaker returns None, below-threshold returns None
- [x] Typecheck passes

---

### US-VID-003: Auto-switch user profile on speaker identification

**Description:** As a user, I want Rex to automatically load my memory, preferences, and connected accounts when it recognizes my voice so I get a personalized response without any manual switching.

**Acceptance Criteria:**
- [x] `Assistant.generate_reply()` loads the identified user's memory profile when `active_user_id` is set
- [x] System prompt injected with user-specific context (name, preferences) from `Memory/{user_id}/`
- [x] Per-user email, calendar, and messaging credentials loaded for the identified user (not another user's)
- [x] Fallback to `default` profile if `active_user_id` is None
- [x] Integration test: two enrolled users, each asking a question — verify correct profile loaded for each
- [x] Typecheck passes

---

### US-VID-004: Voice enrollment UI

**Description:** As a user, I want to enroll or re-enroll my voice through the Rex UI so I do not need to use the CLI.

**Acceptance Criteria:**
- [x] UI has an "Enroll Voice" flow: user clicks "Start Enrollment", Rex records 3 samples with visual countdown, stores embedding
- [x] Progress indicator shows how many samples have been captured
- [x] Success/failure feedback displayed after enrollment
- [x] Enrolled users listed with a "Delete Enrollment" option
- [x] Typecheck passes
- [x] Verify changes work in browser

---

---

# PHASE 4 — Skill Training

---

### US-SK-001: Skill registry — data model and storage

**Description:** As a developer, I want a `SkillRegistry` that stores custom skills as structured records so Rex can load, list, and execute them.

**Acceptance Criteria:**
- [x] `rex/skills/registry.py` created with `SkillRegistry` class
- [x] Each skill record: `id`, `name`, `description`, `trigger_patterns: list[str]`, `handler: str` (module path or script path), `created_at`, `enabled: bool`
- [x] Registry persisted to `config/skills.json`
- [x] `SkillRegistry.register()`, `list_skills()`, `enable()`, `disable()`, `delete()` methods implemented
- [x] Unit tests cover CRUD operations and persistence
- [x] Typecheck passes

---

### US-SK-002: Script-based skill registration

**Description:** As a developer, I want to register a new skill by placing a Python script in `plugins/skills/` with a standard header so Rex automatically discovers and loads it.

**Acceptance Criteria:**
- [x] Rex scans `plugins/skills/` at startup for `*.py` files with a `SKILL_METADATA` dict at the top
- [x] `SKILL_METADATA` must contain: `name`, `description`, `triggers: list[str]`
- [x] Valid scripts auto-registered in `SkillRegistry`; invalid scripts logged and skipped gracefully
- [x] Example skill `plugins/skills/example_weather_skill.py` added showing the pattern
- [x] Unit tests cover: valid script discovered, invalid script skipped, metadata extracted correctly
- [x] Typecheck passes

---

### US-SK-003: Natural language skill creation via chat

**Description:** As a user, I want to describe a new skill to Rex in plain language so Rex creates and registers the skill automatically without me needing to write Python.

**Acceptance Criteria:**
- [x] `Assistant` detects skill-creation intent (e.g. "teach yourself to…", "learn how to…", "add a skill that…")
- [x] On detection, Rex prompts for: skill name, what it should do, example trigger phrases
- [x] Rex generates a Python skill script using the `SKILL_METADATA` pattern and saves it to `plugins/skills/`
- [x] New skill immediately registered in `SkillRegistry` and available for invocation
- [x] Rex confirms: "I've learned how to [X]. You can trigger it by saying [example phrase]."
- [x] Integration test: user message "teach yourself to tell me the current battery level", skill script created, registered, and callable
- [x] Typecheck passes

---

### US-SK-004: Skill invocation — auto-dispatch from user message

**Description:** As a developer, I want the `Assistant` to check registered skill triggers before routing to the LLM so custom skills can intercept and handle matching messages.

**Acceptance Criteria:**
- [x] `SkillRouter.match(message: str) -> Skill | None` checks all enabled skills' `trigger_patterns` against the user message
- [x] On match, skill handler executed; result returned to user
- [x] On no match, normal LLM routing proceeds
- [x] Skill execution errors caught and reported to user gracefully; Rex does not crash
- [x] Skill invocations logged at INFO level
- [x] Unit tests cover: match found and executed, no match falls through, execution error handled
- [x] Typecheck passes

---

---

# PHASE 5 — Automatic Tool Dispatch

---

### US-TD-001: Tool registry — catalog all available Rex tools

**Description:** As a developer, I want a `ToolRegistry` that catalogs every tool Rex can invoke (search, email, calendar, Home Assistant, etc.) with metadata for auto-dispatch.

**Acceptance Criteria:**
- [x] `rex/tools/registry.py` created with `ToolRegistry` class
- [x] Each tool entry: `name`, `description`, `capability_tags: list[str]`, `requires_config: list[str]`, `handler: Callable`
- [x] All existing tools registered: web search, email read/send, calendar, Home Assistant, messaging, weather, file ops
- [x] `ToolRegistry.available_tools()` returns only tools whose `requires_config` fields are satisfied by current `AppConfig`
- [x] Unit tests cover: tool registration, availability filter, missing config excludes tool
- [x] Typecheck passes

---

### US-TD-002: Auto tool selection — map user intent to tools

**Description:** As a developer, I want the `Assistant` to automatically select and invoke the right tool(s) for a user request without the user having to specify a tool name.

**Acceptance Criteria:**
- [x] `ToolDispatcher.select_tools(message: str) -> list[Tool]` implemented using keyword/intent matching
- [x] Rules cover: email intent → email tool, weather intent → weather tool, search intent → search tool, calendar intent → calendar tool, smart home intent → Home Assistant tool
- [x] Multiple tools selected when intent spans multiple domains
- [x] Results from all selected tools aggregated and passed to LLM as context
- [x] No tool invoked if no intent match (normal LLM path)
- [x] Integration tests cover: email question → email tool invoked, weather question → weather tool invoked, compound question → both tools invoked
- [x] Typecheck passes

---

### US-TD-003: Tool execution pipeline with timeout and retry

**Description:** As a developer, I want each tool invocation wrapped in a timeout and one-retry policy so a slow tool does not hang Rex indefinitely.

**Acceptance Criteria:**
- [x] Each tool call wrapped with configurable timeout (default 10s, configurable via `AppConfig.tool_timeout_seconds`)
- [x] On timeout, Rex reports "I couldn't reach [tool name] in time" and continues with available results
- [x] One automatic retry on transient errors (network timeout, HTTP 5xx); no retry on auth errors
- [x] All tool invocations logged: tool name, duration, success/failure
- [x] Unit tests cover: successful invocation, timeout, retry on transient error, no retry on auth error
- [x] Typecheck passes

---

---

# PHASE 6 — Local File and Windows System Access

---

### US-WIN-001: Local file read/write capability

**Description:** As a user, I want Rex to read and write files on my local machine when I ask it to so I can manage documents through voice or chat.

**Acceptance Criteria:**
- [x] `rex/tools/file_ops.py` created with tools: `read_file(path)`, `write_file(path, content)`, `list_directory(path)`, `move_file(src, dst)`, `delete_file(path)`
- [x] All paths validated against an allowlist root (configurable `AppConfig.allowed_file_roots`, default: user home directory)
- [x] Path traversal attacks blocked (no `../` escapes outside allowlist)
- [x] Operations log at INFO level with path and operation type
- [x] Tools registered in `ToolRegistry`
- [x] Unit tests cover: read existing file, write new file, directory listing, path traversal blocked
- [x] Typecheck passes

---

### US-WIN-002: Windows diagnostics — system info and hardware status

**Description:** As a user, I want Rex to report on Windows system health (CPU, RAM, disk, battery, running processes) so I can diagnose issues through conversation.

**Acceptance Criteria:**
- [x] `rex/tools/windows_diagnostics.py` created (Windows-only; gracefully no-ops on non-Windows)
- [x] Implements: `get_system_info()`, `get_cpu_usage()`, `get_memory_usage()`, `get_disk_usage()`, `get_battery_status()`, `list_processes()`
- [x] Uses `psutil` (already in dependency tree); no new heavy dependencies
- [x] Each function returns a structured dict suitable for LLM context injection
- [x] Tools registered in `ToolRegistry` with `capability_tags: ["windows", "diagnostics"]`
- [x] Unit tests mock `psutil` and verify output structure
- [x] Typecheck passes

---

### US-WIN-003: Windows settings read/write via PowerShell bridge

**Description:** As a user, I want Rex to adjust common Windows settings (volume, display brightness, default apps, power plan) through voice commands so I can manage my PC hands-free.

**Acceptance Criteria:**
- [x] `rex/tools/windows_settings.py` created with: `set_volume(level: int)`, `get_volume()`, `set_brightness(level: int)`, `get_power_plan()`, `set_power_plan(name: str)`
- [x] Each function executes via `subprocess` calling PowerShell cmdlets or `nircmd`; non-Windows raises `NotImplementedError` gracefully
- [x] Operations confirmed by re-reading the setting after writing and logging the result
- [x] Settings changes require user confirmation if `AppConfig.require_confirm_system_changes` is True (default True)
- [x] Tools registered in `ToolRegistry`
- [x] Unit tests mock `subprocess.run` and verify PowerShell command strings
- [x] Typecheck passes

---

### US-WIN-004: Windows issue diagnosis and repair suggestions

**Description:** As a user, I want Rex to diagnose common Windows issues and suggest or apply fixes so I can resolve problems through conversation.

**Acceptance Criteria:**
- [x] `rex/tools/windows_repair.py` implements: `check_disk_health()` (SMART via PowerShell), `check_windows_update_status()`, `flush_dns_cache()`, `run_sfc_scan()` (System File Checker, requires elevation prompt)
- [x] Each tool returns a structured result with `status`, `findings`, `recommended_actions`
- [x] Elevation-required operations inform user and request confirmation before executing
- [x] Tools that cannot run on non-Windows return a clear `platform_not_supported` status
- [x] Tools registered in `ToolRegistry`
- [x] Unit tests mock subprocess calls
- [x] Typecheck passes

---

---

# PHASE 7 — Full UI

---

### US-UI-001: UI framework scaffolding — single-page app with nav

**Description:** As a developer, I want a single-page React application scaffolded under `rex/ui/` that provides the navigation skeleton for all UI sections.

**Acceptance Criteria:**
- [x] `rex/ui/` directory created with: `index.html`, `App.jsx`, `main.jsx`, build config (Vite or equivalent)
- [x] Top-level nav includes: Dashboard, Chat, Voice, Settings, Logs, Shopping List, About
- [x] Each nav item renders a placeholder component
- [x] App served by Flask at `/ui/` when `AppConfig.ui_enabled` is True (default True)
- [x] `npm run build` produces `rex/ui/dist/` which Flask serves as static files
- [x] Typecheck passes (Python side)
- [x] Verify changes work in browser

---

### US-UI-002: Chat interface — text input and message history

**Description:** As a user, I want a chat window in the UI where I can type messages to Rex and see the conversation history so I can interact without using voice.

**Acceptance Criteria:**
- [x] Chat panel shows message history with user/Rex bubbles, timestamps
- [x] Text input with send button (also submits on Enter)
- [x] Messages streamed to UI via Server-Sent Events or WebSocket as Rex generates them
- [x] Conversation history persists across page reloads (stored in `rex/dashboard_store.py` or equivalent)
- [x] File upload button allows attaching an image or document to a message
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-003: Voice interface — push-to-talk and wake word status

**Description:** As a user, I want to see the voice loop status in the UI and trigger voice input manually so I can use Rex hands-free or with a button.

**Acceptance Criteria:**
- [x] UI shows live wake word status indicator (listening / detected / processing)
- [x] "Push to Talk" button triggers one-shot STT without wake word
- [x] Audio waveform visualization displays during recording
- [x] Transcription shown in real time as Rex processes it
- [x] Microphone device selector dropdown (populated from available input devices)
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-004: Settings — LLM and model routing

**Description:** As a user, I want a Settings → LLM section that exposes all model and routing configuration options so I can tune Rex's intelligence without editing config files.

**Acceptance Criteria:**
- [x] LLM provider dropdown: OpenAI, Ollama, Local Transformers
- [x] Model name/ID field per provider
- [x] Model routing table: one row per task category with model dropdown (implements US-MM-005)
- [x] API key fields masked (show/hide toggle); values saved to `.env`, not `rex_config.json`
- [x] Save button + success/error toast
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-005: Settings — STT, TTS, wake word

**Description:** As a user, I want a Settings → Voice section for speech-to-text, text-to-speech, and wake word configuration so all audio settings are in one place.

**Acceptance Criteria:**
- [x] STT model dropdown (Whisper model sizes), language selector, device selector (CPU/GPU/auto)
- [x] TTS engine dropdown (XTTS, edge-tts, pyttsx3), speaker/voice dropdown (implements US-VC-001)
- [x] Wake word selector (implements US-WW-001)
- [x] All changes written to `rex_config.json` via config API
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-006: Settings — integrations (email, calendar, messaging)

**Description:** As a user, I want a Settings → Integrations section where I can add, remove, and test connected accounts (email, calendar, SMS) without editing config files.

**Acceptance Criteria:**
- [x] Email accounts list with Add/Remove/Test per account (implements US-ME-001 and US-ME-002)
- [x] Calendar backend selector with connection test
- [x] SMS/messaging provider config with connection test
- [x] Home Assistant URL and token fields
- [x] Each integration shows a colored status badge: Connected / Not Configured / Error
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-007: Settings — users and voice profiles

**Description:** As a user, I want a Settings → Users section to manage enrolled users, their voice profiles, and their memory so multiple household members can be managed from one screen.

**Acceptance Criteria:**
- [x] Users list showing all enrolled users with their avatar/name
- [x] Per-user: Edit Name, Re-enroll Voice, Delete User actions
- [x] Voice enrollment flow embedded (implements US-VID-004)
- [x] Memory viewer: shows stored facts for selected user (read-only)
- [x] Add User button creates a new profile and starts enrollment flow
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-008: Settings — speakers and audio output devices

**Description:** As a user, I want a Settings → Audio Output section to configure which speakers Rex uses for TTS so I can route audio to Sonos, Bose, or any connected device.

**Acceptance Criteria:**
- [x] Available output devices listed (implements US-SP-001)
- [x] Smart speakers (Sonos, Bose) shown separately if discovered on the network
- [x] Per-device: select as default TTS output, test with "Hello" sample
- [x] Volume slider per output device
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-009: Settings — system and advanced

**Description:** As a user, I want a Settings → System section for advanced Rex options so power users can tune behavior without CLI access.

**Acceptance Criteria:**
- [x] Autonomy mode selector (read-only / confirm / auto-execute)
- [x] Tool timeout slider
- [x] Require-confirm for system changes toggle (implements US-WIN-003)
- [x] Allowed file roots configuration (comma-separated paths)
- [x] Debug logging toggle
- [x] Restart Rex button (gracefully restarts the Flask + voice loop)
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-UI-010: File and image upload for LLM context

**Description:** As a user, I want to upload files and images through the chat UI so Rex can analyze them and answer questions about their content.

**Acceptance Criteria:**
- [x] Drag-and-drop or browse file picker in chat input area
- [x] Supported types: `.txt`, `.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.csv`
- [x] Files sent to backend, extracted text / image injected into LLM context
- [x] Images displayed inline in chat bubble; documents shown as attachment chip
- [x] File size limit enforced: 10MB per file, 50MB per session (configurable)
- [x] Typecheck passes
- [x] Verify changes work in browser

---

---

# PHASE 8 — Latency Reduction

---

### US-LAT-001: LLM response streaming to chat UI

**Description:** As a user, I want Rex's text responses to stream into the chat window as they are generated so I see output immediately rather than waiting for the full response.

**Acceptance Criteria:**
- [x] `LanguageModel.generate()` supports a streaming mode returning tokens as they arrive (SSE or WebSocket)
- [x] Chat UI renders tokens progressively with a blinking cursor
- [x] TTS begins speaking the first completed sentence while the rest is still generating
- [x] Non-streaming fallback remains for API clients that do not support streaming
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-LAT-002: Acoustic "thinking" feedback during processing

**Description:** As a user, I want Rex to immediately acknowledge my voice input with a short audio cue so I know it heard me while it processes the full response.

**Acceptance Criteria:**
- [x] Within 200ms of wake word + end-of-utterance detection, Rex plays a short audio chime or says "mm-hmm" / "one moment"
- [x] Acknowledgment sound is configurable: chime file path or spoken filler phrase
- [x] Acknowledgment does not interrupt TTS if Rex is already speaking
- [x] Configurable via `AppConfig.acknowledgment_sound` (default: chime)
- [x] Typecheck passes

---

### US-LAT-003: STT pipeline warm-up — pre-load Whisper model at startup

**Description:** As a developer, I want the Whisper model loaded during Rex startup rather than on first use so the first voice request has no model-load latency.

**Acceptance Criteria:**
- [x] STT model loaded in a background thread immediately after startup (non-blocking)
- [x] `rex doctor` reports "STT model: loaded" once warm-up completes
- [x] First voice recognition request does not trigger model load (model already resident)
- [x] Memory footprint increase documented in `docs/performance-baseline.md`
- [x] Typecheck passes

---

### US-LAT-004: Response caching for repeated factual queries

**Description:** As a developer, I want frequently asked identical or near-identical questions to return cached answers so Rex responds instantly for common queries.

**Acceptance Criteria:**
- [x] `ResponseCache` implemented with TTL (default 5 minutes, configurable)
- [x] Cache keyed on normalized message text (lowercased, stripped punctuation)
- [x] Cache bypassed when message references time-sensitive intents ("right now", "current", "today")
- [x] Cache bypassed for tool-invoking queries (email, calendar, Home Assistant)
- [x] Cache hit rate logged at DEBUG level
- [x] Unit tests cover: cache hit, cache miss, TTL expiry, bypass conditions
- [x] Typecheck passes

---

---

# PHASE 9 — Shopping List

---

### US-SL-001: Shopping list data model and storage

**Description:** As a developer, I want a persistent shopping list data model so items added through any interface survive restarts.

**Acceptance Criteria:**
- [x] `rex/shopping_list.py` created with `ShoppingList` class
- [x] Each item: `id`, `name`, `quantity`, `unit`, `added_by: str`, `checked: bool`, `added_at`, `checked_at`
- [x] Persisted to `data/shopping_list.json` (created if absent)
- [x] Methods: `add_item()`, `check_item()`, `uncheck_item()`, `remove_item()`, `list_items()`, `clear_checked()`
- [x] Per-user shopping list scoped by `added_by` but all users can view and check items
- [x] Unit tests cover all CRUD operations and persistence
- [x] Typecheck passes

---

### US-SL-002: Voice commands for shopping list

**Description:** As a user, I want to add items to the shopping list by voice so I can capture grocery needs hands-free while cooking or moving around.

**Acceptance Criteria:**
- [x] `Assistant` detects shopping list intent: "add [item] to the shopping list", "I need [item]", "put [item] on the list"
- [x] Item extracted and added to `ShoppingList` for the identified user
- [x] Rex confirms verbally: "Added [item] to your shopping list"
- [x] Multiple items in one utterance handled: "add milk, eggs, and butter"
- [x] "What's on my shopping list?" reads back unchecked items
- [x] Integration test: voice utterance → item added → item appears in list
- [x] Typecheck passes

---

### US-SL-003: Shopping list UI — view, add, check off, and clear

**Description:** As a user, I want a Shopping List page in the UI where I can manually add, check off, and manage items so the list is useful even without voice.

**Acceptance Criteria:**
- [ ] Shopping List nav item in UI (see US-UI-001)
- [ ] Items displayed in two sections: "To Buy" and "Got It" (checked)
- [ ] Check box per item; clicking checks/unchecks it
- [ ] Text input to manually add items with quantity field
- [ ] "Clear checked" button removes all checked items
- [ ] Items update in real time without page refresh (SSE or polling every 5s)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-SL-004: Shopping list — mobile-accessible PWA endpoint

**Description:** As a user, I want to access the shopping list from my phone browser so I can check items off at the store without the Rex desktop app.

**Acceptance Criteria:**
- [ ] Shopping list page at `/shopping` is a standalone minimal HTML page (no full app shell)
- [ ] Page is mobile-responsive (works at 375px width)
- [ ] Includes a `<link rel="manifest">` PWA manifest so it can be added to home screen
- [ ] PIN or shared-secret protected (configurable, default: off) to prevent public access
- [ ] Check/uncheck syncs to the main `ShoppingList` storage in real time
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

---

# PHASE 10 — Multi-Email Per User

---

### US-ME-001: Multi-email config schema — per-user account list ✅ DONE

**Description:** As a developer, I want `AppConfig` to support multiple email accounts per user rather than a single global account so each user's emails are isolated.

**Acceptance Criteria:**
- [x] `AppConfig.users` gains a `email_accounts: list[EmailAccountConfig]` field per user
- [x] `EmailAccountConfig` fields: `account_id`, `display_name`, `backend` (imap/gmail/outlook), `credentials_key` (key into `.env`)
- [x] `config/rex_config.example.json` shows two email accounts for one user
- [x] Existing single-account config loads without error (migration shim converts old format)
- [x] Typecheck passes

---

### US-ME-002: Email service — route operations to requesting user's accounts only ✅ DONE

**Description:** As a developer, I want all email operations to be scoped to the requesting user's accounts so User A can never accidentally see User B's email.

**Acceptance Criteria:**
- [x] `EmailService` modified to accept `user_id: str` parameter on all operations
- [x] `EmailService.get_accounts(user_id)` returns only accounts where the user is the owner
- [x] Attempt to access another user's account raises `PermissionError` (not a silent empty result)
- [x] `Assistant` passes `active_user_id` from speaker identification to all email tool calls
- [x] Integration test: two users with separate accounts; verify each only sees their own emails
- [x] Typecheck passes

---

### US-ME-003: Email UI — add and test multiple accounts per user

**Description:** As a user, I want to add multiple email accounts for my profile through the Settings UI so I can check all my inboxes through Rex.

**Acceptance Criteria:**
- [ ] Settings → Integrations → Email section lists all accounts for the current UI user
- [ ] "Add Account" flow: choose backend (IMAP / Gmail OAuth / Outlook OAuth), enter credentials, test connection
- [ ] Test connection button sends a test fetch and reports success/failure
- [ ] Delete account removes credentials from `.env` and config
- [ ] Each account shows last-synced timestamp
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

---

# PHASE 11 — Smart Speaker Integration

---

### US-SP-001: Smart speaker discovery — Sonos and Bose on local network

**Description:** As a developer, I want Rex to discover Sonos and Bose speakers on the local network at startup so they can be used as TTS output targets.

**Acceptance Criteria:**
- [x] `rex/audio/speaker_discovery.py` created
- [x] Sonos discovery via `soco` library (optional; gracefully skipped if not installed)
- [x] Bose SoundTouch discovery via HTTP broadcast on `8090`
- [x] Discovered speakers cached with name, IP, model
- [x] Discovery runs in background thread; does not block startup
- [x] `rex doctor` includes a "Smart Speakers" section listing discovered devices
- [x] Unit tests mock network calls
- [x] Typecheck passes

---

### US-SP-002: Route TTS output to selected smart speaker

**Description:** As a user, I want Rex's voice responses to play through my Sonos or Bose speaker so I can hear Rex in any room.

**Acceptance Criteria:**
- [ ] TTS pipeline checks `AppConfig.tts_output_device`; if set to a discovered smart speaker, streams audio to that device
- [ ] Sonos: audio served as HTTP from Rex and played via `soco.play_uri()`
- [ ] Bose SoundTouch: audio played via SoundTouch REST API
- [ ] Fallback to local audio output if smart speaker unreachable
- [ ] Speaker selection persisted across restarts
- [ ] Unit tests mock speaker APIs
- [ ] Typecheck passes

---

### US-SP-003: Wake word detection from smart speaker microphone

**Description:** As a user, I want Rex to listen for its wake word through a Sonos or compatible microphone-enabled speaker so I can summon Rex from across the room.

**Acceptance Criteria:**
- [ ] `AppConfig.wake_word_input_device` accepts a discovered smart speaker name or `auto`
- [ ] For microphone-enabled Sonos devices, audio stream captured via HTTP or local Alexa-compatible endpoint (where available)
- [ ] Wake word detection pipeline accepts the remote audio stream with no changes to the core openWakeWord integration
- [ ] Graceful fallback to local microphone if remote audio unavailable
- [ ] Typecheck passes

---

---

# PHASE 12 — Voice Selection

---

### US-VC-001: Voice selection UI — dropdown with sample playback

**Description:** As a user, I want to browse available TTS voices in the UI and hear a sample before selecting so I can choose a voice I like.

**Acceptance Criteria:**
- [ ] Settings → Voice → TTS Voice section shows a dropdown listing all available voices for the selected TTS engine
- [ ] Each voice entry includes: name, language, engine label
- [ ] "Play Sample" button plays a 3-second preview using that voice ("Hi, I'm Rex. How can I help?")
- [ ] "Apply" saves the selected voice to `AppConfig.tts_voice`
- [ ] Change takes effect on the next Rex utterance without restart
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-VC-002: Personalized voice creation via audio file upload

**Description:** As a user, I want to upload audio samples through the UI to clone a custom voice for Rex's TTS so Rex sounds exactly how I want.

**Acceptance Criteria:**
- [ ] Settings → Voice → Custom Voice section has a file upload zone accepting `.wav` / `.mp3` files
- [ ] Minimum 10 seconds of audio required; UI shows remaining time needed
- [ ] On submit, XTTS voice cloning pipeline triggered; new voice saved as `config/custom_voices/{name}.pt`
- [ ] New voice appears in voice selection dropdown (US-VC-001) within 30 seconds of successful training
- [ ] Training progress shown in UI (polling or SSE)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

---

# PHASE 13 — Wake Word Configuration

---

### US-WW-001: Wake word selection via UI

**Description:** As a user, I want to choose Rex's wake word from a predefined list through the UI so I can use a word that feels natural and doesn't conflict with my other devices.

**Acceptance Criteria:**
- [ ] Settings → Voice → Wake Word section shows a dropdown with all bundled openWakeWord models
- [ ] Each entry shows the wake word text and engine name
- [ ] "Play Sample" button plays the reference audio for the wake word
- [ ] Selection saved to `AppConfig.wake_word_model`
- [ ] Voice loop reloads the new wake word model without full Rex restart (hot-swap)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-WW-002: Custom wake word training via UI

**Description:** As a user, I want to record a custom wake word through the UI so Rex responds to my own chosen phrase instead of the presets.

**Acceptance Criteria:**
- [ ] "Train Custom Wake Word" button in Settings → Voice → Wake Word
- [ ] UI guides user through recording 10 positive samples of the chosen phrase and 5 negative samples
- [ ] openWakeWord (or compatible) training script invoked with recorded samples
- [ ] Trained model saved to `config/wake_words/{phrase}/model.onnx`
- [ ] New wake word appears in dropdown and can be selected (US-WW-001)
- [ ] Training progress shown in UI; estimated time displayed
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

---

# PHASE 14 — Log Viewer UI

---

### US-LOG-001: Structured logging to rotating file

**Description:** As a developer, I want Rex to write structured JSON logs to a rotating log file so the UI log viewer has a machine-readable source.

**Acceptance Criteria:**
- [ ] `rex/logging_config.py` updated to add a `RotatingFileHandler` writing to `logs/rex.log`
- [ ] Each log line is valid JSON: `{timestamp, level, logger, message, extra}`
- [ ] Log rotation: 5MB max per file, keep last 5 files
- [ ] Log directory created automatically if absent
- [ ] `logs/` added to `.gitignore`
- [ ] Typecheck passes

---

### US-LOG-002: Log viewer UI — live tail with filter and search

**Description:** As a user, I want to view Rex's logs in real time from the UI so I can diagnose issues without SSH access.

**Acceptance Criteria:**
- [ ] Logs nav item in UI (see US-UI-001) opens the log viewer
- [ ] Logs tail in real time via SSE from `/api/logs/stream`
- [ ] Filter by log level (DEBUG / INFO / WARNING / ERROR / CRITICAL)
- [ ] Full-text search box filters visible entries
- [ ] "Pause" button freezes the stream; "Resume" re-tails from current position
- [ ] "Download" button exports the current log file as a `.log` file
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

---

# PHASE 15 — Phone Number Integration

---

### US-PH-001: Phone number linking — Twilio inbound webhook

**Description:** As a developer, I want Rex to receive inbound calls and SMS messages via a Twilio phone number so Rex can act as a voice assistant and message handler on a real phone line.

**Acceptance Criteria:**
- [ ] `rex/telephony/twilio_handler.py` created implementing Twilio webhook routes: `/telephony/inbound/call` and `/telephony/inbound/sms`
- [ ] Twilio credentials (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`) loaded from `.env`
- [ ] Inbound call answered with a TwiML greeting: "Hi, you've reached Rex. How can I help?"
- [ ] Inbound SMS text passed to `Assistant.generate_reply()` and response sent back via SMS
- [ ] Webhook signature validated (prevents spoofed requests)
- [ ] Integration gracefully disabled if Twilio credentials absent
- [ ] Typecheck passes

---

### US-PH-002: Inbound call — STT and LLM conversation loop

**Description:** As a user, I want callers to be able to speak with Rex on the phone so Rex can answer questions, take messages, and route calls.

**Acceptance Criteria:**
- [ ] TwiML `<Gather>` collects caller speech; transcription forwarded to `Assistant.generate_reply()`
- [ ] Rex's response synthesized via TTS and played back as TwiML `<Say>` or `<Play>`
- [ ] Conversation continues for up to 5 turns before ending gracefully
- [ ] "Leave a message" intent detected; caller's message saved to `data/voicemail/{caller_number}_{timestamp}.txt`
- [ ] Rex can transfer the call (TwiML `<Dial>`) if user has configured a transfer number
- [ ] Typecheck passes

---

### US-PH-003: Outbound calling — Rex initiates calls on user request

**Description:** As a user, I want to ask Rex to call a contact on my behalf so I can make hands-free calls without picking up my phone.

**Acceptance Criteria:**
- [ ] `rex/telephony/outbound.py` implements `make_call(to_number: str, message: str | None)` using Twilio REST API
- [ ] User says: "Call [contact name]" or "Call [phone number]"
- [ ] If contact name given, Rex looks up number in user's contact list (configurable JSON file or vCard)
- [ ] Rex calls the number and plays `message` if provided, or connects to Rex's conversation loop
- [ ] Outbound calls logged with timestamp, number, outcome
- [ ] Requires explicit user confirmation before dialing
- [ ] Typecheck passes

---

### US-PH-004: Phone integration settings UI

**Description:** As a user, I want to configure phone integration through the Settings UI so I can link my Twilio number and set call routing preferences without editing config files.

**Acceptance Criteria:**
- [ ] Settings → Integrations → Phone section with fields: Twilio Account SID, Auth Token (masked), Phone Number
- [ ] "Verify Connection" button calls Twilio API and reports status
- [ ] Transfer-to number field (where to forward calls Rex cannot handle)
- [ ] Voicemail notification toggle (notify user via chat when a voicemail is saved)
- [ ] Contact list file upload (`.vcf` or `.json`)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

---

## Technical Considerations

- All new Flask routes must be authenticated (existing `api_key_auth.py`) and rate-limited
- All new integrations must degrade gracefully when not configured — no startup crashes
- Windows-specific tools must guard with `sys.platform == "win32"` and return `NotImplementedError` on other platforms
- React UI build output (`rex/ui/dist/`) must be `.gitignore`d; build step documented in `INSTALL.md`
- Smart speaker libraries (`soco`, SoundTouch) are optional — install-time not required for core Rex
- Twilio is optional — install-time not required; activate only when `TWILIO_*` env vars present
- `psutil` is likely already present; confirm in `pyproject.toml` before adding
- All new config fields use Pydantic v2 validators; no raw dict access
- Story ordering: schema/config → backend logic → Flask routes → UI — never reversed
