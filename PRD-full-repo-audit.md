# PRD: Full Repository Audit & Quality Remediation

## Introduction

This PRD directs a systematic, exhaustive review of every file in the Rex AI Assistant repository. Every Python file, TypeScript file, shell script, and configuration file must be read, audited for issues, and remediated. The audit covers truncated/incomplete code, logic errors, syntax errors, missing implementations, undefined references, security vulnerabilities, dead code, and test integrity. All identified issues must be fixed — no issue is left as a known defect or deferred to follow-up. The repo must exit this process with clean linting (Ruff + Black), a passing TypeScript typecheck, and all tests passing.

**This PRD is exhaustive by design.** Every story must produce a fully read, fully fixed scope before it is marked complete.

---

## Goals

- Every `.py`, `.ts`, `.tsx`, `.js` file in the repo has been manually read and reviewed
- All truncated functions (bodies that are `pass`, `...`, `# TODO`, placeholder strings, or cut off mid-logic) are completed with correct implementations
- All logic errors, undefined references, bad imports, and security issues are corrected
- Ruff and Black pass with zero violations on all Python files
- `tsc --noEmit` passes on the `gui/` TypeScript codebase
- All existing tests pass with zero failures
- Test files themselves are free of incomplete assertions, missing fixtures, and placeholder bodies

---

## Definitions

**Issue types to detect and fix in every file:**
- Truncated function/class bodies (`pass`, `...`, `raise NotImplementedError` without context, empty return where a value is expected)
- Placeholder comments (`# TODO`, `# FIXME`, `# stub`, `# implement me`) left without implementation
- Imports of modules that do not exist or are circular
- Undefined variables or names used before assignment
- Unreachable code / dead branches
- Security anti-patterns: shell injection, hardcoded secrets, unvalidated external input used in dangerous contexts
- Type annotation mismatches obvious from context
- Missing `__all__` where a public API is implied
- Test files with `assert True`, `pass`, empty test bodies, or missing assertions
- TypeScript: `any` casts hiding obvious type errors, non-null assertions (`!`) on values that can be null

---

## User Stories

### US-001: Root-Level Voice Loop and Wake Word Files
**Description:** As a code reviewer, I want to audit every root-level file that drives the voice pipeline so that the live voice experience is free of bugs, truncated logic, and security issues.

**Files in scope:**
- `rex_loop.py`
- `voice_loop.py`
- `wakeword_listener.py`
- `wakeword_utils.py`
- `wake_acknowledgment.py`
- `rex_speak_api.py`
- `rex_assistant.py`

**Acceptance Criteria:**
- [x] Every file listed above is fully read before any edits begin
- [x] All truncated function bodies are completed with correct implementations
- [x] All `# TODO` / `# FIXME` / placeholder comments are resolved or removed
- [x] No undefined variable references remain
- [x] `rex_speak_api.py` binds only to localhost by default and has auth + rate limiting active
- [x] `rex_loop.py` uses `Assistant.generate_reply()` (not bare LLM calls) per CLAUDE.md rule
- [x] `voice_loop.py` (root) is distinct from `rex/voice_loop.py` — no unintended cross-references
- [x] All files pass `ruff check` and `black --check`
- [x] Typecheck passes

---

### US-002: Root-Level Bridge Files
**Description:** As a code reviewer, I want to audit all IPC bridge scripts at the repo root so that the Electron GUI communicates reliably with the Python backend.

**Files in scope:**
- `rex_chat_bridge.py`
- `rex_chat_stream_bridge.py`
- `rex_memories_bridge.py`
- `rex_reminders_bridge.py`
- `rex_stt_bridge.py`
- `rex_tasks_bridge.py`
- `rex_voice_bridge.py`
- `rex_voice_sample_bridge.py`
- `rex_voices_bridge.py`

**Acceptance Criteria:**
- [x] Every file listed above is fully read before any edits begin
- [x] All bridge entry points are complete — no missing `main()` bodies or stubs
- [x] All JSON serialization/deserialization handles errors gracefully (no bare `json.loads` on unchecked input without try/except)
- [x] No hardcoded secrets or absolute paths remain
- [x] All files pass `ruff check` and `black --check`
- [x] Typecheck passes

---

### US-003: Root-Level Config, Memory, LLM, and Logging
**Description:** As a code reviewer, I want to audit the root-level configuration, memory, LLM client, and logging modules so that these foundational systems are correct and complete.

**Files in scope:**
- `config.py`
- `llm_client.py`
- `conversation_memory.py`
- `memory_utils.py`
- `logging_utils.py`
- `assistant_errors.py`
- `plugin_loader.py`
- `audio_config.py`

**Acceptance Criteria:**
- [x] Every file listed above is fully read before any edits begin
- [x] No secrets (API keys, tokens) are hardcoded — all come from env or config
- [x] `llm_client.py` has complete implementations for all LLM provider branches (no stubs)
- [x] `conversation_memory.py` and `memory_utils.py` have no truncated persistence logic
- [x] All logging setup is complete — no half-initialized handlers
- [x] `plugin_loader.py` gracefully handles missing plugins without crashing
- [x] All files pass `ruff check` and `black --check`
- [x] Typecheck passes

---

### US-004: Root-Level Setup, GUI Entry Points, and Diagnostic Scripts
**Description:** As a code reviewer, I want to audit all remaining root-level Python files so that setup, GUI launch, and diagnostic tooling are complete and correct.

**Files in scope:**
- `setup.py`, `install.py`, `wsgi.py`, `flask_proxy.py`
- `run_gui.py`, `gui.py`, `gui_settings_tab.py`
- `conftest.py`
- `check_gpu_status.py`, `check_imports.py`, `check_patch_status.py`, `check_tts_imports.py`
- `test_imports.py`, `test_transformers_patch.py`
- `patch_tts_torch_load.py`, `patch_tts_transformers.py`
- `placeholder_voice.py`, `generate_wake_sound.py`, `find_gpt2_model.py`
- `list_voices.py`, `manual_search_demo.py`, `manual_whisper_demo.py`
- `play_test.py`, `record_wakeword.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] Patch scripts use `find_spec()` to check module availability before importing (per CLAUDE.md rule)
- [ ] `conftest.py` has all fixtures complete and no placeholder bodies
- [ ] GUI entry points (`run_gui.py`, `gui.py`) have no missing initialization logic
- [ ] Diagnostic scripts produce meaningful output — no stubs returning `None` silently
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-005: `rex/` Core — App, CLI, Config, and Assistant
**Description:** As a code reviewer, I want to audit the core `rex/` package entry points so that the CLI, application bootstrap, config loading, and assistant orchestration are fully implemented.

**Files in scope:**
- `rex/__init__.py`
- `rex/__main__.py`
- `rex/app.py`
- `rex/cli.py`
- `rex/config.py`
- `rex/config_manager.py`
- `rex/assistant.py`
- `rex/assistant_errors.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `rex/assistant.py` has a fully implemented `generate_reply()` method that routes through tools and injects system context
- [ ] `rex/cli.py` entry point is complete — no missing subcommand handlers
- [ ] `rex/config.py` uses Pydantic v2 correctly — no deprecated `.dict()` calls (use `.model_dump()`)
- [ ] `AppConfig.whisper_device` defaults to `"auto"` and resolves at model-load time per CLAUDE.md rule
- [ ] `rex/app.py` Flask app has CORS, rate limiting, and auth correctly wired
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-006: `rex/` Voice Pipeline
**Description:** As a code reviewer, I want to audit the `rex/` voice pipeline modules so that STT, TTS, and voice loop logic are correct and complete.

**Files in scope:**
- `rex/voice_loop.py`
- `rex/voice_loop_optimized.py`
- `rex/voice_latency.py`
- `rex/tts_utils.py`
- `rex/tts_voices.py`
- `rex/wake_acknowledgment.py`
- `rex/wakeword_utils.py`
- `rex/llm_client.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `rex/voice_loop.py` is confirmed separate from root `voice_loop.py` — no accidental shared state
- [ ] TTS lazy imports use `find_spec()` before `import_module()` (per CLAUDE.md rule)
- [ ] `rex/llm_client.py` whisper device resolution uses `torch.cuda.is_available()` when `device == "auto"` (per CLAUDE.md rule)
- [ ] No truncated audio processing pipelines
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-007: `rex/` Services — Email, Calendar, Messaging, Reminder, Scheduler
**Description:** As a code reviewer, I want to audit the high-level service layer in `rex/` so that all integration services are fully implemented and degrade gracefully when unconfigured.

**Files in scope:**
- `rex/email_service.py`
- `rex/calendar_service.py`
- `rex/messaging_service.py`
- `rex/reminder_service.py`
- `rex/scheduler.py`
- `rex/integrations.py`
- `rex/cue_store.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] All services degrade gracefully when not configured (no uncaught import errors or crashes)
- [ ] No service makes blocking network calls without a timeout parameter
- [ ] `rex/scheduler.py` has no partial job registration logic
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-008: `rex/` Tools and Execution — Tool Registry, Workflow, Planner, Executor
**Description:** As a code reviewer, I want to audit the tool routing and workflow execution layer so that tool calls, planning, and execution are fully implemented.

**Files in scope:**
- `rex/tool_registry.py`
- `rex/tool_router.py`
- `rex/workflow.py`
- `rex/workflow_runner.py`
- `rex/executor.py`
- `rex/planner.py`
- `rex/autonomy_modes.py`
- `rex/automation_registry.py`
- `rex/replay.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] Tool registration is complete — no tool stubs with empty handler bodies
- [ ] `rex/tool_router.py` dispatches to all registered tools without silent no-ops
- [ ] `rex/planner.py` and `rex/workflow_runner.py` have no truncated planning loops
- [ ] `rex/replay.py` can replay logged interactions without crashing on edge cases
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-009: `rex/` Infrastructure — Events, Startup, Services, Retry, Shutdown
**Description:** As a code reviewer, I want to audit the infrastructure layer so that event bussing, startup sequencing, service management, retry logic, and shutdown are robust.

**Files in scope:**
- `rex/event_bus.py`
- `rex/event_triggers.py`
- `rex/startup.py`
- `rex/startup_validation.py`
- `rex/services.py`
- `rex/service_supervisor.py`
- `rex/graceful_shutdown.py`
- `rex/retry.py`
- `rex/process_monitor.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] Event bus has no incomplete subscriber/publisher logic
- [ ] `rex/startup_validation.py` validates all required config fields before the app starts
- [ ] `rex/graceful_shutdown.py` correctly unregisters all resources — no resource leaks
- [ ] `rex/retry.py` implements exponential backoff with jitter — no busy loops
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-010: `rex/` Security and API Layer
**Description:** As a code reviewer, I want to audit the authentication, authorization, rate limiting, and validation modules so that no security regressions exist.

**Files in scope:**
- `rex/api_key_auth.py`
- `rex/credentials.py`
- `rex/rate_limiter.py`
- `rex/policy.py`
- `rex/policy_engine.py`
- `rex/validation.py`
- `rex/audit.py`
- `rex/http_errors.py`
- `rex/request_logging.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] No secrets hardcoded in any file
- [ ] All API key checks fail closed (deny by default, not allow by default)
- [ ] Rate limiter is applied to all externally-facing endpoints — no bypass paths
- [ ] `rex/validation.py` validates all inputs at system boundaries — no raw `request.json` used without schema validation
- [ ] `rex/audit.py` logs all security-relevant events (auth failures, policy denials)
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-011: `rex/` Utilities, Notifications, and Background Jobs
**Description:** As a code reviewer, I want to audit the utility, notification, and background job modules so that supporting infrastructure is complete and correct.

**Files in scope:**
- `rex/logging_utils.py`
- `rex/db_pool.py`
- `rex/dep_errors.py`
- `rex/health.py`
- `rex/doctor.py`
- `rex/retention.py`
- `rex/quiet_hours.py`
- `rex/production_config.py`
- `rex/notification.py`
- `rex/notification_priority.py`
- `rex/priority_notification_router.py`
- `rex/followup_engine.py`
- `rex/digest_job.py`
- `rex/escalation_job.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `rex/health.py` returns meaningful health check data — no stub `{"status": "ok"}` without checks
- [ ] `rex/retention.py` has complete data cleanup logic
- [ ] Background jobs (`digest_job.py`, `escalation_job.py`) have no truncated scheduling or dispatch logic
- [ ] `rex/quiet_hours.py` correctly computes time windows — no off-by-one errors in hour comparisons
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-012: `rex/` External Integrations — HA, MQTT, Plex, GitHub, VS Code, OS, Browser
**Description:** As a code reviewer, I want to audit all external integration modules so that third-party service clients are correct, safe, and gracefully optional.

**Files in scope:**
- `rex/ha_bridge.py`
- `rex/mqtt_client.py`
- `rex/mqtt_audio_router.py`
- `rex/plex_client.py`
- `rex/github_service.py`
- `rex/vscode_service.py`
- `rex/os_automation.py`
- `rex/browser_automation.py`
- `rex/geolocation.py`
- `rex/weather.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] All integrations are guarded with availability checks — missing config does not crash startup
- [ ] `rex/os_automation.py` and `rex/browser_automation.py` do not construct shell commands via string concatenation (no injection risk)
- [ ] All HTTP clients have timeout parameters set
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-013: `rex/` GUI, Windows Service, and App Launch
**Description:** As a code reviewer, I want to audit the desktop GUI bindings, Windows service, and app launcher modules so that platform-specific code is complete and safe.

**Files in scope:**
- `rex/gui_app.py`
- `rex/windows_service.py`
- `rex/app_launcher.py`
- `rex/first_run.py`
- `rex/exception_handler.py`
- `rex/compat.py`
- `rex/memory.py`
- `rex/memory_utils.py`
- `rex/identity.py`
- `rex/profile_manager.py`
- `rex/knowledge_base.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `rex/windows_service.py` has no Windows-only calls outside of platform guards
- [ ] `rex/first_run.py` completes its setup flow — no half-initialized state paths
- [ ] `rex/memory.py` and `rex/knowledge_base.py` have no truncated read/write operations
- [ ] `rex/identity.py` has complete identity resolution logic
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-014: `rex/autonomy/` Subpackage
**Description:** As a code reviewer, I want to audit the entire `rex/autonomy/` subpackage so that goal parsing, planning, replanning, cost estimation, and preference learning are fully implemented.

**Files in scope (all 16 files):**
- `rex/autonomy/__init__.py`
- `rex/autonomy/clarifier.py`
- `rex/autonomy/cost_estimator.py`
- `rex/autonomy/feedback.py`
- `rex/autonomy/goal_graph.py`
- `rex/autonomy/goal_parser.py`
- `rex/autonomy/history.py`
- `rex/autonomy/llm_planner.py`
- `rex/autonomy/models.py`
- `rex/autonomy/preference_learner.py`
- `rex/autonomy/preferences.py`
- `rex/autonomy/replanner.py`
- `rex/autonomy/retry.py`
- `rex/autonomy/rule_planner.py`
- `rex/autonomy/runner.py`
- `rex/autonomy/tool_cache.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `rex/autonomy/models.py` uses Pydantic v2 — no `.dict()` calls
- [ ] `rex/autonomy/runner.py` has a complete execution loop with no truncated branches
- [ ] `rex/autonomy/replanner.py` handles replanning trigger conditions — no empty handler blocks
- [ ] `rex/autonomy/cost_estimator.py` returns real estimates — no `return 0` stubs
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-015: `rex/calendar_backends/` Subpackage
**Description:** As a code reviewer, I want to audit the calendar backends so that ICS parsing, free/busy lookup, meeting invites, and the stub backend are all fully implemented.

**Files in scope (all 9 files):**
- `rex/calendar_backends/__init__.py`
- `rex/calendar_backends/base.py`
- `rex/calendar_backends/factory.py`
- `rex/calendar_backends/free_busy_stub.py`
- `rex/calendar_backends/free_time_finder.py`
- `rex/calendar_backends/ics_backend.py`
- `rex/calendar_backends/ics_parser.py`
- `rex/calendar_backends/meeting_invite.py`
- `rex/calendar_backends/stub.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `ics_parser.py` handles malformed ICS data without raising unhandled exceptions
- [ ] `free_time_finder.py` has no truncated time-window iteration logic
- [ ] `meeting_invite.py` generates complete RFC 5545-compliant VEVENT blocks
- [ ] All backend classes implement the full interface defined in `base.py`
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-016: `rex/email_backends/` Subpackage
**Description:** As a code reviewer, I want to audit the email backends so that IMAP/SMTP, triage, and routing logic are fully implemented and secure.

**Files in scope (all 9 files):**
- `rex/email_backends/__init__.py`
- `rex/email_backends/account_config.py`
- `rex/email_backends/account_router.py`
- `rex/email_backends/base.py`
- `rex/email_backends/imap_smtp.py`
- `rex/email_backends/inbox_stub.py`
- `rex/email_backends/stub.py`
- `rex/email_backends/triage.py`
- `rex/email_backends/triage_rules.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `imap_smtp.py` connects with TLS and has timeout handling on all socket operations
- [ ] `triage.py` and `triage_rules.py` have no incomplete rule evaluation branches
- [ ] No email credentials are logged at any log level
- [ ] All backend classes implement the full interface from `base.py`
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-017: `rex/messaging_backends/` Subpackage
**Description:** As a code reviewer, I want to audit the messaging backends so that SMS sending/receiving, Twilio integration, and webhook handling are fully implemented and secure.

**Files in scope (all 14 files):**
- `rex/messaging_backends/__init__.py`
- `rex/messaging_backends/account_config.py`
- `rex/messaging_backends/base.py`
- `rex/messaging_backends/factory.py`
- `rex/messaging_backends/inbound_store.py`
- `rex/messaging_backends/inbound_webhook.py`
- `rex/messaging_backends/message_router.py`
- `rex/messaging_backends/sms_receiver_stub.py`
- `rex/messaging_backends/sms_sender_stub.py`
- `rex/messaging_backends/stub.py`
- `rex/messaging_backends/twilio_adapter.py`
- `rex/messaging_backends/twilio_backend.py`
- `rex/messaging_backends/twilio_signature.py`
- `rex/messaging_backends/webhook_wiring.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `twilio_signature.py` validates Twilio webhook signatures before processing (not just a pass-through)
- [ ] `inbound_webhook.py` treats all incoming SMS content as untrusted input
- [ ] No Twilio credentials are logged
- [ ] All backend classes implement the full interface from `base.py`
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-018: `rex/notifications/` and `rex/voice_identity/` Subpackages
**Description:** As a code reviewer, I want to audit the notifications and voice identity subpackages so that notification routing, digest, escalation, and speaker recognition are fully implemented.

**Files in scope:**

`rex/notifications/` (6 files):
- `rex/notifications/__init__.py`
- `rex/notifications/digest.py`
- `rex/notifications/escalation.py`
- `rex/notifications/models.py`
- `rex/notifications/quiet_hours.py`
- `rex/notifications/router.py`

`rex/voice_identity/` (8 files):
- `rex/voice_identity/__init__.py`
- `rex/voice_identity/calibration.py`
- `rex/voice_identity/embedding_backends.py`
- `rex/voice_identity/embeddings_store.py`
- `rex/voice_identity/fallback_flow.py`
- `rex/voice_identity/optional_deps.py`
- `rex/voice_identity/recognizer.py`
- `rex/voice_identity/types.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `notifications/router.py` routes to all registered channels — no unreachable channel branches
- [ ] `notifications/escalation.py` has complete escalation timer logic
- [ ] `voice_identity/recognizer.py` has a complete recognition pipeline — no empty `identify()` bodies
- [ ] `voice_identity/optional_deps.py` uses `find_spec()` before importing optional dependencies
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-019: `rex/wakeword/` and `rex/computers/` Subpackages
**Description:** As a code reviewer, I want to audit the wakeword and computers subpackages so that wake word selection, embedding, and the agent computer-use server are fully implemented.

**Files in scope:**

`rex/wakeword/` (4–5 files):
- `rex/wakeword/__init__.py`
- `rex/wakeword/embedding.py`
- `rex/wakeword/listener.py`
- `rex/wakeword/selection.py`
- `rex/wakeword/utils.py`

`rex/computers/` (6 files):
- `rex/computers/__init__.py`
- `rex/computers/agent_server.py`
- `rex/computers/client.py`
- `rex/computers/config.py`
- `rex/computers/pc_run_policy.py`
- `rex/computers/service.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `wakeword/listener.py` has a complete audio stream ingestion loop — no truncated callback logic
- [ ] `computers/agent_server.py` binds to localhost and requires auth before executing commands
- [ ] `computers/pc_run_policy.py` has a complete policy evaluation — no always-allow default
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-020: `rex/dashboard/` and `rex/integrations/` Subpackages
**Description:** As a code reviewer, I want to audit the dashboard routes and the integrations subpackage so that the SSE dashboard and all integration orchestration are fully implemented.

**Files in scope:**

`rex/dashboard/` (4 Python + 1 JS):
- `rex/dashboard/__init__.py`
- `rex/dashboard/auth.py`
- `rex/dashboard/routes.py`
- `rex/dashboard/sse.py`
- `rex/dashboard/static/js/dashboard.js`

`rex/integrations/` (8 files):
- `rex/integrations/__init__.py`
- `rex/integrations/calendar_service.py`
- `rex/integrations/email_service.py`
- `rex/integrations/message_router.py`
- `rex/integrations/models.py`
- `rex/integrations/scheduling_engine.py`
- `rex/integrations/sms_service.py`
- `rex/integrations/triage_engine.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `dashboard/sse.py` has a complete SSE event loop — no truncated generator bodies
- [ ] `dashboard/auth.py` enforces auth on all dashboard routes — no unauthenticated endpoints
- [ ] `dashboard/static/js/dashboard.js` has no incomplete event handlers or dead code blocks
- [ ] `integrations/triage_engine.py` fully implements triage decision logic
- [ ] All Python files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-021: Remaining `rex/` Subpackages
**Description:** As a code reviewer, I want to audit the remaining smaller subpackages so that WooCommerce, WordPress, Home Assistant TTS, compat shims, contracts, and plugin stubs are fully correct.

**Files in scope:**

`rex/woocommerce/` (5 files): all `.py` files
`rex/wordpress/` (4 files): all `.py` files
`rex/ha_tts/` (3 files): all `.py` files
`rex/compat/` (2 files): all `.py` files
`rex/contracts/` (3 files): all `.py` files
`rex/capabilities/` (1 file): `__init__.py`
`rex/plugins/` (1 file): `__init__.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `woocommerce/write_policy.py` has a complete policy evaluation — no stub returning `True`
- [ ] `ha_tts/client.py` has correct async/await usage if async, or blocking if sync — no mixed models
- [ ] `compat/transformers_shims.py` applies shims before any import of the shimmed module (per CLAUDE.md rule)
- [ ] `contracts/core.py` has complete contract validation logic
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-022: `scripts/` and Root `plugins/`
**Description:** As a code reviewer, I want to audit all operational scripts and root-level plugins so that maintenance tooling and the plugin system work correctly.

**Files in scope:**
- `scripts/doctor.py`
- `scripts/export_contract_schemas.py`
- `scripts/security_audit.py`
- `scripts/test_voice_pipeline.py`
- `scripts/validate_deployment.py`
- `scripts/validate_wakeword_model.py`
- `scripts/windows_agent.py`
- `plugins/__init__.py`
- `plugins/web_search.py`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `scripts/doctor.py` checks all critical dependencies and reports clearly — no silent pass paths
- [ ] `scripts/security_audit.py` has complete audit logic — not a stub
- [ ] `plugins/web_search.py` sanitizes search queries before forwarding to any search provider
- [ ] All files pass `ruff check` and `black --check`
- [ ] Typecheck passes

---

### US-023: Test Suite — Core Tests (Voice, LLM, Config, CLI)
**Description:** As a code reviewer, I want to audit all tests related to core functionality so that voice pipeline, LLM, config, and CLI tests are complete, have real assertions, and pass.

**Files in scope:** All test files in `tests/` whose names match: `test_voice_*`, `test_llm_*`, `test_config_*`, `test_cli_*`, `test_assistant_*`, `test_wake*`, `test_stt_*`, `test_tts_*`, `test_audio_*`, `test_rex_loop*`, `test_tools*`, `test_tool_*`

**Acceptance Criteria:**
- [ ] Every file in scope is fully read before any edits begin
- [ ] No test body contains only `pass`, `assert True`, or empty assertion blocks
- [ ] All fixtures referenced in tests are defined (no `fixture 'X' not found` errors)
- [ ] No test imports a module that does not exist
- [ ] Tests in scope pass: `pytest -q <files>`
- [ ] Typecheck passes

---

### US-024: Test Suite — Integration and Service Tests
**Description:** As a code reviewer, I want to audit all integration and service tests so that email, calendar, messaging, autonomy, and scheduler tests are complete and passing.

**Files in scope:** All test files in `tests/` whose names match: `test_email_*`, `test_calendar_*`, `test_messaging_*`, `test_sms_*`, `test_autonomy_*`, `test_scheduler_*`, `test_reminder_*`, `test_workflow_*`, `test_executor_*`, `test_planner_*`, `test_integrat*`, `test_triage_*`, `US[0-9]*.py` (user story tests)

**Acceptance Criteria:**
- [ ] Every file in scope is fully read before any edits begin
- [ ] No test body contains only `pass`, `assert True`, or empty assertion blocks
- [ ] All mocks correctly model the interface of what they're mocking — no `MagicMock()` used where a real object with specific behavior is needed
- [ ] Tests in scope pass: `pytest -q <files>`
- [ ] Typecheck passes

---

### US-025: Test Suite — Infrastructure, Security, and Subpackage Tests
**Description:** As a code reviewer, I want to audit all remaining test files covering infrastructure, security, notifications, voice identity, computers, and dashboard so that the full test suite is clean.

**Files in scope:** All test files in `tests/` not covered by US-023 or US-024 — including: `test_auth_*`, `test_policy_*`, `test_rate_*`, `test_notification_*`, `test_dashboard_*`, `test_voice_identity_*`, `test_computer*`, `test_event_*`, `test_startup_*`, `test_shutdown_*`, `test_health_*`, `test_plugin_*`, `test_memory_*`, `test_identity_*`, `test_db_*`, `test_compat_*`, `test_wakeword_*`, `conftest.py`

**Acceptance Criteria:**
- [ ] Every file in scope is fully read before any edits begin
- [ ] No test body contains only `pass`, `assert True`, or empty assertion blocks
- [ ] `conftest.py` at `tests/` level has all shared fixtures complete
- [ ] Tests in scope pass: `pytest -q <files>`
- [ ] Typecheck passes

---

### US-026: Electron GUI — Main Process and IPC Handlers
**Description:** As a code reviewer, I want to audit the Electron main process and all IPC handlers so that the desktop GUI communicates correctly with the Python backend.

**Files in scope:**
- `gui/src/main/index.ts`
- `gui/src/main/tray.ts`
- `gui/src/main/handlers/calendar.ts`
- `gui/src/main/handlers/chat.ts`
- `gui/src/main/handlers/email.ts`
- `gui/src/main/handlers/memories.ts`
- `gui/src/main/handlers/notifications.ts`
- `gui/src/main/handlers/reminders.ts`
- `gui/src/main/handlers/sms.ts`
- `gui/src/main/handlers/tasks.ts`
- `gui/src/main/handlers/voice.ts`
- `gui/src/preload/index.ts`
- `gui/src/preload/index.d.ts`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] All IPC handlers are complete — no handler that immediately returns `undefined` or `null` with no logic
- [ ] Bridge calls to Python processes include error handling for non-zero exit codes and stderr
- [ ] Preload exposes only the minimum required API surface (no `shell.openExternal` exposed blindly)
- [ ] `tsc --noEmit` passes on the gui/ project
- [ ] Typecheck passes

---

### US-027: Electron GUI — Renderer, Pages, and UI Components
**Description:** As a code reviewer, I want to audit all React pages and UI components so that the renderer is complete, type-safe, and free of broken logic.

**Files in scope:**
- `gui/src/renderer/src/App.tsx`
- `gui/src/renderer/src/main.tsx`
- `gui/src/pages/` (all 10 pages)
- `gui/src/components/` (all 23 components including `chat/`, `calendar/`, `voice/`, `ui/`)
- `gui/src/layouts/AppLayout.tsx`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] No page or component renders a blank screen due to missing `return` statements or unconditional `null` returns
- [ ] No `useEffect` with an async callback missing cleanup (no potential state-after-unmount updates)
- [ ] `ChatInput.tsx` mic button wires correctly to voice dictation (per recent feature addition)
- [ ] No `@ts-ignore` or `@ts-expect-error` comments hiding real type errors
- [ ] `tsc --noEmit` passes on the gui/ project
- [ ] Typecheck passes

---

### US-028: Electron GUI — Hooks, Store, Types, and Build Config
**Description:** As a code reviewer, I want to audit the remaining GUI infrastructure files so that state management, global hooks, type definitions, and build config are correct.

**Files in scope:**
- `gui/src/hooks/useGlobalShortcuts.ts`
- `gui/src/store/notificationsStore.ts`
- `gui/src/types/ipc.ts`
- `gui/electron.vite.config.ts`
- `gui/tailwind.config.ts`
- `gui/tsconfig.json`
- `gui/tsconfig.node.json`
- `gui/tsconfig.web.json`
- `gui/postcss.config.js`
- `gui/package.json`

**Acceptance Criteria:**
- [ ] Every file listed above is fully read before any edits begin
- [ ] `notificationsStore.ts` has complete state actions — no action creators that do nothing
- [ ] `ipc.ts` type definitions cover all IPC channels used in handlers (no untyped channels)
- [ ] `tsconfig.json` paths are correct and no non-existent path aliases are referenced
- [ ] Build config (`electron.vite.config.ts`) has no hardcoded absolute paths
- [ ] `tsc --noEmit` passes on the gui/ project
- [ ] Typecheck passes

---

### US-029: Full Python Lint and Formatting Pass
**Description:** As a code reviewer, I want to run Ruff and Black against every changed and pre-existing Python file so that the entire codebase conforms to project style standards.

**Steps:**
1. Run `ruff check --fix` on all Python files
2. Run `ruff check` to confirm zero remaining violations
3. Run `black .` on all Python files
4. Run `black --check --diff .` to confirm zero formatting diffs
5. Fix any violations that auto-fix did not resolve

**Acceptance Criteria:**
- [ ] `ruff check .` exits with code 0 — zero violations
- [ ] `black --check --diff .` exits with code 0 — zero formatting diffs
- [ ] No `# noqa` suppressions added to hide legitimate violations — all violations are properly resolved
- [ ] Typecheck passes

---

### US-030: Full Test Suite Pass and Coverage Verification
**Description:** As a code reviewer, I want to run the complete pytest suite so that all tests pass after all remediation work, and confirm that test coverage has not regressed.

**Steps:**
1. Run `pytest -q` and capture output
2. Fix any remaining failures (do not skip tests — fix the code or the test)
3. Run `pytest -q --tb=short` to confirm zero failures
4. Run `pytest --cov=rex --cov-report=term-missing -q` to check coverage
5. Ensure coverage is not lower than the baseline captured before this audit began

**Acceptance Criteria:**
- [ ] `pytest -q` exits with 0 failures and 0 errors
- [ ] No test is marked `xfail` or `skip` as a workaround for a broken implementation — only legitimate skips remain
- [ ] Coverage report shows no newly uncovered lines introduced by fixes in this audit
- [ ] All tests pass
- [ ] Typecheck passes

---

## Non-Goals

- Adding new features not already stubbed or partially implemented in the codebase
- Refactoring working code that has no issues (do not clean up style when content is correct)
- Changing architecture or splitting modules
- Adding documentation or docstrings to functions that weren't touched during remediation
- Upgrading dependencies beyond what is needed to fix a specific identified bug
- Investigating or modifying `CODEX_REPO_AUDIT_ISSUES.json` — treat it as read-only reference

---

## Technical Considerations

- Per CLAUDE.md: lazy TTS imports must use `find_spec()` before `import_module()` — check all TTS-adjacent files
- Per CLAUDE.md: `rex/voice_loop.py` and root `voice_loop.py` are separate — do not conflate them
- Per CLAUDE.md: `AppConfig.whisper_device = "auto"` resolves at model-load time via `torch.cuda.is_available()`
- Per CLAUDE.md: voice loop must use `Assistant.generate_reply()`, not bare LLM calls
- Lint preflight command from CLAUDE.md:
  ```bash
  BASE_REF="master"
  git fetch origin "$BASE_REF"
  files=$(git diff --name-only "origin/$BASE_REF...HEAD" -- '*.py')
  ruff check --fix $files
  ruff check $files
  black $files
  black --check --diff $files
  ```
- TypeScript typecheck command: `cd gui && npx tsc --noEmit`
- Use Conventional Commits for every commit: `fix(scope): description`
