# PRD: AskRex Architectural Simplification

**Version:** 1.0  
**Date:** 2026-05-24  
**Status:** Ready for implementation

---

## Introduction

The AskRex codebase has grown into a capable system, but five architectural problems are now slowing development and threatening production-readiness: too many active UI surfaces, a god-object `Assistant` class, overlapping tool layers, a monolithic config object, and a cluttered root directory. None of these are conceptual mistakes — they are growth scars. This PRD defines an incremental plan to resolve them without breaking the system.

The objective is not fewer features. It is fewer duplicate, overlapping, or legacy layers. The layered architecture the build spec calls for (input, identity, intent, context, model, planning, tool, verification, response, memory, monitoring, security, evaluation) is still the target. This work clears the path to get there cleanly.

**Scope of decisions already made:**
- Non-first-class surfaces → move to `/archived/`, remove entry points (not deleted outright)
- `Assistant` refactor → extract one subsystem per story, keeping `Assistant` functional throughout
- Tool canon → `rex/tools/` is authoritative; OpenClaw adapts into it
- Structure → one PRD, five epics, ordered by dependency

---

## Goals

- Reduce cognitive overhead for new contributors by eliminating "which layer owns this?" ambiguity
- Make `rex/assistant.py` safe to edit without understanding the entire system
- Establish a single canonical tool interface that all dispatch paths flow through
- Decompose `AppConfig` (128 fields, 1045 lines) into navigable sub-configs without breaking existing behavior
- Reduce root-level `.py` files from 46 to a legible handful
- All first-class entry points (`rex`, `rex-gui`, voice loop, backend services) remain fully functional after each story

---

## Epics Overview

| Epic | Area | Dependency |
|------|------|-----------|
| 1 | Config Decomposition | None — foundational |
| 2 | Surface Consolidation | None — can run in parallel with Epic 1 |
| 3 | Tool Layer Unification | Epic 1 preferred first |
| 4 | Assistant Refactor | Epics 1 and 3 first |
| 5 | Root Directory Cleanup | Epics 1 and 2 first |

---

## User Stories

---

### EPIC 1 — Config Decomposition

> **Goal:** Break `AppConfig`'s 128 fields into typed sub-config groups without changing existing behavior. New nested paths (`config.audio.sample_rate`) are the target; flat paths remain valid during the transition via backward-compat properties.

---

### US-001: Define sub-config Pydantic models

**Description:** As a developer, I want domain-specific config classes so I can understand and modify one subsystem's config without reading 1045 lines.

**Context:**  
`rex/config.py` contains `class AppConfig` at line 135 with ~128 fields spanning every subsystem. This story adds new sub-config classes above `AppConfig` — no behavior changes yet.

**Acceptance Criteria:**
- [x] Add the following Pydantic v2 model classes to `rex/config.py` above `AppConfig`:
  - `AudioConfig` — fields: `sample_rate`, `channels`, `chunk_size`, `input_device`, `output_device`, `vad_sensitivity`, and any other audio-hardware fields
  - `VoiceConfig` — fields: `tts_engine`, `tts_voice`, `stt_model`, `whisper_device`, `wakeword_model`, `wakeword_sensitivity`, `wakeword_fallback_keyword`, and related voice-pipeline fields
  - `LLMConfig` — fields: `llm_provider`, `model_name`, `openai_api_key_env`, `ollama_url`, `context_length`, `temperature`, model-routing fields
  - `ToolsConfig` — fields: `tool_timeout`, `tool_max_retries`, `enabled_tools`, `tool_permissions`
  - `IntegrationsConfig` — fields: `home_assistant_base_url`, `ha_token_env`, `email_*`, `calendar_*`, `music_*`, `shopping_*`, `openclaw_gateway_url`, `openclaw_gateway_timeout`, `openclaw_gateway_max_retries`
  - `UIConfig` — fields: `gui_port`, `gui_host`, `dashboard_*`, `theme`
  - `SecurityConfig` — fields: `api_key_env`, `rate_limit_*`, `allowed_origins`, `auth_*`
- [x] Each class uses `model_config = ConfigDict(extra="ignore")` so unknown fields do not raise
- [x] Each class has sensible defaults matching current `AppConfig` defaults
- [x] `AppConfig` is unchanged in this story
- [x] `pytest -q` passes

---

### US-002: Add nested sub-config fields to AppConfig

**Description:** As a developer, I want `AppConfig` to expose sub-config objects so I can access settings via `config.audio.sample_rate` rather than flat attribute lookup.

**Context:**  
Adds nested fields to `AppConfig` populated from existing flat fields. Flat fields remain on `AppConfig` unchanged so all existing code continues to work.

**Acceptance Criteria:**
- [x] `AppConfig` gains seven new fields: `audio: AudioConfig`, `voice: VoiceConfig`, `llm: LLMConfig`, `tools: ToolsConfig`, `integrations: IntegrationsConfig`, `ui: UIConfig`, `security: SecurityConfig`
- [x] Each sub-config is instantiated in `AppConfig.model_post_init()` or a `@model_validator(mode="after")` by reading from the flat fields already on `AppConfig`
- [x] Instantiating `AppConfig` from the existing `config/rex_config.json` format still works with no JSON changes required
- [x] `config.audio.sample_rate` and `config.sample_rate` both resolve to the same value
- [x] `rex doctor` still passes
- [x] `pytest -q` passes

---

### US-003: Add deprecation warnings to high-traffic flat config fields

**Description:** As a developer, I want to be notified when I use a flat config field that now has a nested equivalent, so I know to migrate my call site.

**Context:**  
Identify the 15–20 most-imported flat `AppConfig` fields (grep `config\.<field>` across `rex/`). Convert them to `@property` on `AppConfig` that emit `DeprecationWarning` and return the value from the nested sub-config. This story does not change any call sites — it only installs the warning mechanism.

**Acceptance Criteria:**
- [x] At least these fields are converted to deprecated properties: `llm_provider`, `model_name`, `tts_engine`, `tts_voice`, `whisper_device`, `wakeword_model`, `home_assistant_base_url`, `openclaw_gateway_url`, `tool_timeout`, `gui_port`, `api_key_env`, `rate_limit_per_minute`
- [x] Accessing a deprecated flat field emits `DeprecationWarning: Use config.<group>.<field> instead`
- [x] The value returned is identical to the nested path value
- [x] Existing code that reads flat fields still works (no `AttributeError`)
- [x] `pytest -q` passes (warnings are acceptable; failures are not)

---

### US-004: Update CLAUDE.md and docs for new config structure

**Description:** As a developer, I want the canonical reference docs to document the nested config structure so new contributors learn the right pattern from day one.

**Acceptance Criteria:**
- [x] `CLAUDE.md` config section updated to show nested structure (`config.audio.*`, `config.voice.*`, etc.) as the preferred access pattern
- [x] `CONFIGURATION.md` updated with the new grouped field reference
- [x] A short migration note explains that flat fields still work but emit deprecation warnings
- [x] No code changes in this story
- [x] `pytest -q` passes

---

### EPIC 2 — Surface Consolidation

> **Goal:** Reduce active UI surfaces to Electron GUI, voice loop, and CLI. Archive legacy surfaces with entry points removed. Backend services (API, tool server, TTS API) remain as services, not user-facing surfaces.

**First-class surfaces after this epic:**

| Surface | Entry point | Status |
|---------|------------|--------|
| Electron GUI | `rex-gui` | First-class |
| Voice loop | `python rex_loop.py` | First-class |
| CLI | `rex` | First-class |
| Flask API (backend) | internal | Backend service |
| Tool server | `rex-tool-server` | Backend service |
| TTS API | `rex-speak-api` | Backend service |

---

### US-005: Create /archived directory with policy document

**Description:** As a developer, I want a clearly marked holding area for deprecated surfaces so their provenance and future are obvious to anyone who finds them.

**Acceptance Criteria:**
- [x] `/archived/` directory created at repo root
- [x] `/archived/ARCHIVED.md` created explaining: what "archived" means (not deleted, not maintained, entry points removed), why each item was archived, and that items may be deleted in a future major version
- [x] `pytest -q` passes

---

### US-006: Archive Tkinter legacy GUI

**Description:** As a developer, I want the Tkinter GUI files removed from the active codebase so the repo's front door does not imply Tkinter is a supported interface.

**Context:**  
Files to archive: `gui.py`, `gui_settings_tab.py`, `run_gui.py` (all root-level). CLAUDE.md already calls `gui.py` deprecated.

**Acceptance Criteria:**
- [x] `gui.py`, `gui_settings_tab.py`, `run_gui.py` moved to `/archived/tkinter_gui/`
- [x] `setup.py` `py_modules` list (if present) no longer includes these files
- [x] No `pyproject.toml` entry points reference these files
- [x] `/archived/ARCHIVED.md` updated with a Tkinter GUI section
- [x] `rex-gui` still launches the React/Flask GUI (`rex.gui_app:main`) correctly
- [x] `pytest -q` passes

---

### US-007: Archive shopping PWA

**Description:** As a developer, I want the shopping PWA surface archived so it is not mistaken for a maintained UI path.

**Context:**  
`rex/shopping_pwa.py` is the standalone shopping PWA surface. The shopping list logic (`rex/shopping_list.py`, `rex/shopping_list_handler.py`) is used by the assistant and must stay. Only the PWA surface layer moves.

**Acceptance Criteria:**
- [x] `rex/shopping_pwa.py` moved to `/archived/shopping_pwa/shopping_pwa.py`
- [x] Any Flask route or entry point that serves the shopping PWA as a standalone app is removed or guarded with a deprecation log at startup
- [x] `rex/shopping_list.py` and `rex/shopping_list_handler.py` are untouched
- [x] `/archived/ARCHIVED.md` updated with shopping PWA section
- [x] `pytest -q` passes

---

### US-008: Mark Flask GUI as backend-only

**Description:** As a developer, I want `rex/gui_app.py` to be clearly scoped as a backend API for Electron, not a standalone browser UI, so no one adds new standalone Flask UI features.

**Context:**  
`rex/gui_app.py` is already the correct backend for the Electron GUI. The issue is that it can be run directly as a browser app, creating two "GUI" paths. This story adds a startup warning when accessed without the Electron shell, and updates the docs.

**Acceptance Criteria:**
- [x] `rex/gui_app.py` startup logs `WARNING: Rex GUI is designed to run inside the Electron shell. Running standalone may produce an incomplete experience.` when `ELECTRON_RUN_AS_NODE` env var is not set
- [x] `CLAUDE.md` updated: `rex-gui` entry point description changed to "Electron-backed GUI server; not a standalone browser app"
- [x] `README.md` GUI section updated to show only the Electron path as the supported interface
- [x] `pytest -q` passes

---

### US-009: Audit and finalize pyproject.toml entry points

**Description:** As a developer, I want `pyproject.toml` to only list entry points that correspond to first-class or intentional backend-service surfaces.

**Context:**  
Current entry points: `rex`, `rex-config`, `rex-speak-api`, `rex-agent`, `rex-gui`, `rex-tool-server`. Review each; document its role; remove any that point to archived files.

**Acceptance Criteria:**
- [x] Each entry point in `[project.scripts]` has a one-line role comment in `CLAUDE.md` (first-class / backend service / utility)
- [x] Any entry point pointing to an archived file is removed from `pyproject.toml`
- [x] `pip install -e .` succeeds cleanly
- [x] All remaining entry points (`rex`, `rex-gui`, `rex-speak-api`, `rex-agent`, `rex-tool-server`) launch without import errors
- [x] `pytest -q` passes

---

### EPIC 3 — Tool Layer Unification

> **Goal:** `rex/tools/` (`dispatcher.py` + `registry.py`) is the single canonical tool interface. OpenClaw's tool layers (`tool_executor.py`, `tool_registry.py`) become thin adapters that delegate to it. Post-processing tool handling in `assistant.py` routes through the dispatcher.

**Current state:**
- `rex/tools/registry.py` — local tool registry
- `rex/tools/dispatcher.py` — local tool dispatch
- `rex/openclaw/tool_executor.py` — OpenClaw execution (partially overlaps with above)
- `rex/openclaw/tool_registry.py` — OpenClaw registry (partially overlaps with above)
- `rex/openclaw/tool_bridge.py` — bridge between assistant and OpenClaw tools
- Tool post-processing logic also lives inline in `rex/assistant.py`

---

### US-010: Define canonical ToolInterface Protocol in rex/tools/

**Description:** As a developer, I want a typed Protocol that defines what a tool registry and dispatcher must implement, so I can depend on the interface rather than the implementation.

**Acceptance Criteria:**
- [x] Create `rex/tools/protocol.py` with:
  - `class ToolRegistryProtocol(Protocol)` — methods: `register(name, fn, schema)`, `lookup(name) -> Callable | None`, `list_tools() -> list[ToolDescriptor]`
  - `class ToolDispatcherProtocol(Protocol)` — methods: `dispatch(name, args, context) -> ToolResult`
  - `ToolDescriptor` dataclass: `name: str`, `description: str`, `schema: dict`, `source: str` (e.g. `"local"` or `"openclaw"`)
  - `ToolResult` dataclass: `success: bool`, `output: Any`, `error: str | None`
- [x] `rex/tools/registry.py` and `rex/tools/dispatcher.py` include a comment confirming they satisfy these protocols (no structural changes yet)
- [x] `pytest -q` passes

---

### US-011: Refactor rex/openclaw/tool_registry.py to delegate to rex/tools/registry.py

**Description:** As a developer, I want OpenClaw's tool registry to be a thin adapter over the canonical registry rather than a parallel implementation.

**Context:**  
`rex/openclaw/tool_registry.py` currently maintains its own registration logic. After this story it calls through to `rex/tools/registry.py` for registration and lookup, adding only OpenClaw-specific metadata (remote endpoint, auth, channel) on top.

**Acceptance Criteria:**
- [x] `rex/openclaw/tool_registry.py` imports and uses `rex.tools.registry` as its backing store for `register()` and `lookup()`
- [x] OpenClaw-specific metadata (remote endpoint URL, auth) is stored in a separate `_openclaw_meta: dict[str, OpenClawToolMeta]` dict — not mixed into the canonical registry
- [x] Tools registered via OpenClaw are visible in `rex/tools/registry.py`'s `list_tools()` with `source="openclaw"`
- [x] Existing OpenClaw tool invocation paths still work end-to-end
- [x] `pytest -q` passes

---

### US-012: Refactor rex/openclaw/tool_executor.py to delegate to rex/tools/dispatcher.py

**Description:** As a developer, I want OpenClaw's tool executor to route local tool calls through the canonical dispatcher so there is one code path for local execution.

**Context:**  
For tools that execute locally (not over the OpenClaw gateway), `tool_executor.py` should call `rex/tools/dispatcher.py` rather than reimplementing dispatch. Gateway HTTP execution stays in `tool_executor.py` — that is its genuine responsibility.

**Acceptance Criteria:**
- [x] `rex/openclaw/tool_executor.py` checks if a tool is local via `rex.tools.registry.lookup(name)`; if found locally, delegates to `rex.tools.dispatcher.dispatch()`
- [x] Remote OpenClaw execution (gateway HTTP call) remains in `tool_executor.py`
- [x] The `use_openclaw_tools` feature flag behavior is preserved: gateway tried first, 404 falls back to local dispatch
- [x] Existing tests in `tests/test_retirement_check_*.py` still pass
- [x] `pytest -q` passes

---

### US-013: Remove duplicate tool post-processing from assistant.py

**Description:** As a developer, I want all tool result handling to flow through `rex/tools/dispatcher.py` so there is one place to add logging, retries, or result transformation.

**Context:**  
`rex/assistant.py` currently contains inline tool post-processing (result formatting, error wrapping, retry logic) inside `generate_reply()`. This story moves that logic into a `ToolResultHandler` in `rex/tools/` and replaces the inline code with a single call.

**Acceptance Criteria:**
- [x] Tool result post-processing extracted to `rex/tools/result_handler.py`
- [x] `rex/assistant.py` inline tool result handling replaced with a call to `ToolResultHandler`
- [x] Voice loop tool responses (e.g., Home Assistant confirmations) produce identical output before and after the change
- [x] `pytest -q` passes

---

### EPIC 4 — Assistant Refactor

> **Goal:** `rex/assistant.py` (1795 lines) becomes a thin orchestrator. Responsibilities are extracted one subsystem at a time into dedicated modules. `Assistant.generate_reply()` calls through four components: `ContextBuilder` → `IntentRouter` → `ActionDispatcher` → `ResponseBuilder`. `Assistant` remains functional after every individual story.

**Extraction order:** ContextBuilder first (most self-contained), then IntentRouter, then ActionDispatcher (depends on Epic 3 tool canon), then ResponseBuilder, then final slim-down.

---

### US-014: Extract ContextBuilder

**Description:** As a developer, I want history retrieval, system prompt construction, and persona injection in a dedicated class so I can modify context assembly without touching LLM routing or tool dispatch.

**Acceptance Criteria:**
- [ ] Create `rex/context/builder.py` with `class ContextBuilder`
- [ ] `ContextBuilder.__init__` accepts `config: AppConfig`, `history_store: HistoryStore`, `identity` (or equivalent)
- [ ] `ContextBuilder.build(user_message: str) -> ContextPackage` returns a dataclass with: `messages: list[dict]` (LLM-formatted), `system_prompt: str`, `session_id: str`, `user_facts: dict`
- [ ] Logic moved from `assistant.py`: history retrieval, system prompt template rendering, user-facts injection, persona/personality injection
- [ ] `assistant.py` replaces that logic with `context = self._context_builder.build(user_message)`
- [ ] `Assistant.generate_reply()` produces identical output for a text input before and after the change
- [ ] `pytest -q` passes

---

### US-015: Extract IntentRouter

**Description:** As a developer, I want shortcut handling (time, date, greetings, recipe queries) in a dedicated router so I can add or remove shortcuts without reading the full assistant orchestration.

**Context:**  
`assistant.py` currently short-circuits `generate_reply()` for several intent types before hitting the LLM. These shortcuts move to a router that returns early with a `DirectResponse` when the intent is recognized.

**Acceptance Criteria:**
- [ ] Create `rex/intent/router.py` with `class IntentRouter`
- [ ] `IntentRouter.route(user_message: str, context: ContextPackage) -> IntentResult`
- [ ] `IntentResult` dataclass: `handled: bool`, `response: str | None`, `intent_type: str | None`
- [ ] The following shortcuts moved from `assistant.py` to `IntentRouter`: time/date queries, greeting detection, recipe shortcut, and any other direct-return patterns currently in `generate_reply()`
- [ ] `assistant.py` replaces inline shortcuts with `intent = self._intent_router.route(message, context); if intent.handled: return intent.response`
- [ ] Shortcut responses (e.g., "What time is it?") produce identical output
- [ ] `pytest -q` passes

---

### US-016: Extract ActionDispatcher

**Description:** As a developer, I want tool dispatch, skill dispatch, and Home Assistant command routing in a single `ActionDispatcher` so the path from intent to action is traceable in one file.

**Context:**  
`assistant.py` currently routes tool calls, skill invocations, HA commands, and OpenClaw actions inline. This story extracts that routing. Should follow Epic 3 since `ActionDispatcher` delegates to `rex/tools/dispatcher.py`.

**Acceptance Criteria:**
- [ ] Create `rex/actions/dispatcher.py` with `class ActionDispatcher`
- [ ] `ActionDispatcher.dispatch(intent: IntentResult, context: ContextPackage, llm_response: str) -> ActionResult`
- [ ] `ActionResult` dataclass: `success: bool`, `response: str`, `actions_taken: list[str]`, `error: str | None`
- [ ] Routing moved from `assistant.py`: tool invocation (via `rex.tools.dispatcher`), skill invocation, HA command routing (via `rex.ha_bridge`), OpenClaw tool bridge calls
- [ ] `assistant.py` replaces inline dispatch with `result = self._action_dispatcher.dispatch(intent, context, llm_response)`
- [ ] HA commands, tool calls, and skill invocations produce identical outputs
- [ ] `pytest -q` passes

---

### US-017: Extract ResponseBuilder

**Description:** As a developer, I want response cache checking, suggestion generation, follow-up injection, and response post-processing in a single `ResponseBuilder` so I can tune response shaping without touching dispatch logic.

**Acceptance Criteria:**
- [ ] Create `rex/response/builder.py` with `class ResponseBuilder`
- [ ] `ResponseBuilder.build(action_result: ActionResult, context: ContextPackage) -> FinalResponse`
- [ ] `FinalResponse` dataclass: `text: str`, `tts_text: str`, `suggestions: list[str]`, `followups: list[str]`, `cache_hit: bool`
- [ ] Logic moved from `assistant.py`: response cache lookup/write (via `rex.response_cache`), suggestion generation (via `rex.suggestions`), follow-up injection (via `rex.followup_engine`), TTS text cleaning/normalization
- [ ] `assistant.py` replaces inline post-processing with `final = self._response_builder.build(action_result, context)`
- [ ] Response suggestions and follow-ups appear correctly in GUI and voice output after the change
- [ ] `pytest -q` passes

---

### US-018: Slim Assistant.generate_reply() and update docs

**Description:** As a developer, I want `assistant.py` to read like an orchestration spec rather than an implementation, so the flow of a request through the system is obvious at a glance.

**Context:**  
After US-014–017, `generate_reply()` should already be much smaller. This story verifies the final shape, removes any remaining inline logic that belongs in a component, and documents the new architecture.

**Acceptance Criteria:**
- [ ] `rex/assistant.py` is under 400 lines
- [ ] `Assistant.generate_reply()` method body reads as: build context → route intent → dispatch action → build response → return; no inline business logic
- [ ] `CLAUDE.md` updated with the new `Assistant` architecture section documenting `Assistant -> ContextBuilder -> IntentRouter -> ActionDispatcher -> ResponseBuilder`
- [ ] `docs/claude/` reference docs updated to reflect new module locations
- [ ] `pytest -q` passes

---

### EPIC 5 — Root Directory Cleanup

> **Goal:** Reduce root-level `.py` files from 46 to ~10. The 21 `rex_*_bridge.py` files move to `/bridge/`. Legacy wrappers, patch scripts, and compatibility shims move to `/archived/` or `/scripts/`.

---

### US-019: Move root-level bridge files to /bridge/

**Description:** As a developer, I want the 21 `rex_*_bridge.py` files in a named directory so the root directory communicates "these are Electron IPC bridges" rather than "these are 21 unrelated scripts."

**Context:**  
Files: `rex_calendar_bridge.py`, `rex_chat_bridge.py`, `rex_chat_stream_bridge.py`, `rex_email_bridge.py`, `rex_file_extract_bridge.py`, `rex_memories_bridge.py`, `rex_reminders_bridge.py`, `rex_shopping_list_bridge.py`, `rex_sms_bridge.py`, `rex_speaker_bridge.py`, `rex_stt_bridge.py`, `rex_tasks_bridge.py`, `rex_tts_bridge.py`, `rex_voice_bridge.py`, `rex_voice_enrollment_bridge.py`, `rex_voice_sample_bridge.py`, `rex_voice_upload_bridge.py`, `rex_voices_bridge.py`, `rex_wakeword_list_bridge.py`, `rex_wakeword_sample_bridge.py`, `rex_wakeword_train_bridge.py`.

The Electron main process spawns these as child processes by path. Electron `main.js` (or equivalent) spawn paths must be updated together with the file moves.

**Acceptance Criteria:**
- [ ] All 21 `rex_*_bridge.py` files moved to `/bridge/`
- [ ] `/bridge/README.md` added explaining: "These are Electron IPC bridge processes. Each is spawned by the Electron main process and communicates over stdin/stdout JSON."
- [ ] Electron `main.js` (or equivalent) updated so all `spawn()` calls reference the new `/bridge/` paths
- [ ] Voice bridge, chat bridge, and STT bridge verified working end-to-end from the Electron GUI
- [ ] `pytest -q` passes

---

### US-020: Move root-level shims and patch files to /archived/

**Description:** As a developer, I want root-level compatibility shims and patch files in `/archived/` so they do not imply they are part of the normal startup path.

**Context:**  
Files to evaluate: `patch_tts_torch_load.py`, `patch_tts_transformers.py`, `python_compat.py`, `sitecustomize.py`, `placeholder_voice.py`, `flask_proxy.py`. Check each for live imports before moving. Files with live imports move to `rex/compat/` instead of `/archived/`.

**Acceptance Criteria:**
- [ ] Each file in scope is grepped for live imports across the codebase before any move is made
- [ ] Files with no live imports moved to `/archived/compat_shims/`
- [ ] Files with live imports moved to `rex/compat/` with import paths updated everywhere
- [ ] `/archived/ARCHIVED.md` updated with compat shims section
- [ ] `rex doctor` still passes
- [ ] `pytest -q` passes

---

### US-021: Move root-level legacy wrappers to /archived/

**Description:** As a developer, I want root-level re-export shims that are no longer primary entry points replaced with one-liners that emit deprecation warnings, and eventually moved to `/archived/`.

**Context:**  
Files to evaluate: `voice_loop.py` (root — per CLAUDE.md, kept only for `AsyncRexAssistant` re-exports), `rex_assistant.py`, `conversation_memory.py`, `memory_utils.py`, `audio_config.py`, `logging_utils.py`, `llm_client.py`, `config.py` (root-level duplicate), `assistant_errors.py`, `plugin_loader.py`.

For each: if it is a re-export shim, replace it with a one-liner that re-exports with `DeprecationWarning`. `rex_loop.py` (voice loop entry point) is NOT moved — it is first-class.

**Acceptance Criteria:**
- [ ] `voice_loop.py` (root) replaced with a one-liner re-export + `DeprecationWarning` pointing to `rex.voice_loop`
- [ ] Other identified root-level re-export shims treated the same way or moved to `/archived/`
- [ ] `rex_loop.py` is NOT touched
- [ ] Root directory contains 12 or fewer `.py` files after this story (verify with `ls *.py | wc -l`)
- [ ] `pytest -q` passes

---

### US-022: Consolidate install scripts and update docs

**Description:** As a developer, I want install-related scripts in `/scripts/` rather than the root so the installation surface is obvious to a first-time contributor.

**Context:**  
Root install scripts: `install.py`, `install.ps1`, `install.sh`, `install_full.sh`, `install_lean.sh`, `setup.sh`. Keep the one(s) the README and INSTALL.md point to as primary; move or archive the rest.

**Acceptance Criteria:**
- [ ] Primary install path per OS identified from `README.md` and `INSTALL.md` and preserved (or moved to `/scripts/` with a root-level note)
- [ ] Non-primary install scripts moved to `/scripts/install/`
- [ ] `README.md` install section updated to point to one install path per OS (Windows, macOS/Linux)
- [ ] `INSTALL.md` updated consistently
- [ ] `pytest -q` passes

---

### US-023: Final root audit and CLAUDE.md structure update

**Description:** As a developer, I want the CLAUDE.md repository structure section to accurately reflect the cleaned-up root so onboarding documents stay truthful.

**Acceptance Criteria:**
- [ ] Root `.py` file count is 10 or fewer (verified with `ls *.py | wc -l`)
- [ ] `CLAUDE.md` repository structure section updated to list what each root-level item is and why it is there
- [ ] `/bridge/`, `/archived/`, and any other new directories added to the repo structure section
- [ ] `rex doctor` passes
- [ ] `pytest -q` passes

---

## Non-Goals

- This PRD does not add any new features, integrations, or UI components.
- This PRD does not remove the OpenClaw integration. OpenClaw remains a first-class external tool ecosystem — it becomes a consumer of `rex/tools/` rather than a parallel implementation.
- This PRD does not delete archived files. Moving to `/archived/` is the end state for this release cycle. Deletion is a separate decision.
- This PRD does not change the voice pipeline architecture. Wake word, STT, and TTS subsystems are not restructured here.
- This PRD does not migrate all `AppConfig` flat-field call sites. The deprecation warning mechanism handles that incrementally over time.
- This PRD does not change the Electron application structure beyond updating bridge spawn paths in US-019.
- This PRD does not convert the Flask GUI to a different framework.

---

## Technical Considerations

- **Backward compatibility during transition:** The deprecation-warning approach (US-003, US-021) means no code breaks — it just becomes noisy. Treat warnings as a migration queue, not an emergency.
- **Story independence:** Epics 1 and 2 are independent and can run in parallel. Epic 3 stories can start after Epic 1. Epic 4 stories should follow Epics 1 and 3. Epic 5 stories can start after Epics 1 and 2.
- **Test gate:** `pytest -q` must pass after every story. Any story touching import paths must verify the full suite, not just the targeted module.
- **Lint gate:** Per CLAUDE.md, run `ruff check --fix` and `black` on all changed `.py` files before committing each story. Both must pass.
- **Commit messages:** Per CLAUDE.md, use Conventional Commits. Use `refactor:` for structural extraction stories and `chore:` for file-move and doc-update stories.
- **`rex doctor`:** Must pass after every Epic 2 and Epic 5 story, since those touch entry points and file paths.
- **Existing PRD:** The repo's existing `outputs/PRD.md` addresses production-readiness milestones. This document is complementary — it addresses the KISS structural issues that block sustainable development toward those milestones.
