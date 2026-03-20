# PRD: Full Repository Test, Lint, and Fix Audit

## Introduction

Rex AI Assistant has accumulated ~108 source modules, ~290 test files, and numerous integration subsystems. This PRD instructs Claude Code to systematically test every part of the codebase, fix all failures, enforce lint and formatting standards, identify coverage gaps, and leave the repo in a fully passing, clean state. The priority order is: blocking import/crash errors first, then test failures, then lint/formatting, then coverage gaps.

## Goals

- Every module in `rex/` imports without error
- `pytest -q` passes with zero failures across all ~290 test files
- `ruff check` passes with zero errors on all Python files
- `black --check` passes on all Python files
- Untested modules are identified and receive at least smoke-level test coverage
- No regressions are introduced by fixes
- A final verification run confirms the entire suite is green

## User Stories

### US-001: Environment Setup and Dependency Validation

**Description:** As a developer, I need the environment set up and all dependencies installed so that subsequent stories can run tests and lint without missing-package errors.

**Acceptance Criteria:**

- [x] Virtual environment created and activated
- [x] `pip install -e ".[dev]"` completes without error
- [x] `pip install -r requirements-dev.txt` completes without error
- [x] `python scripts/doctor.py` runs and its output is captured for reference
- [x] `python -c "import rex"` succeeds
- [x] `pytest --co -q` (collect-only) runs and reports the total number of collected tests
- [x] Any missing optional dependencies noted but not blocking (audio, ml extras are optional)
- [x] Document the Python version and pip freeze output in a `test-audit-env.txt` file at repo root

---

### US-002: Full Import Scan -- Identify and Fix Broken Imports

**Description:** As a developer, I need every Python module in `rex/` to be importable so that tests can actually load the code they're testing.

**Acceptance Criteria:**

- [x] Write and run a script that attempts `importlib.import_module()` on every `.py` file under `rex/` (recursively), skipping `__pycache__`
- [x] Capture all `ImportError`, `ModuleNotFoundError`, and `AttributeError` exceptions with module path and traceback
- [x] Fix each broken import by: adding missing lazy-import guards, correcting typos, or adding conditional availability checks
- [x] Respect the CLAUDE.md rule: use `find_spec()` before `import_module()` for modules with side-effect imports (e.g., TTS/transformers)
- [x] Re-run the import scan: zero failures
- [x] Do NOT add new heavy dependencies; use optional-import patterns for anything not in core `dependencies`
- [x] Ruff check passes on all modified files
- [x] Black formatting passes on all modified files

---

### US-003: Ruff Lint -- Full Codebase Pass

**Description:** As a developer, I need `ruff check` to pass on every Python file so the codebase meets its own lint standards.

**Acceptance Criteria:**

- [x] Run `ruff check rex/ tests/ *.py scripts/ utils/ plugins/` capturing all errors
- [x] Triage errors into auto-fixable vs. manual-fix categories
- [x] Run `ruff check --fix` on auto-fixable errors
- [x] Manually fix remaining errors (unused imports, undefined names, syntax issues)
- [x] Do NOT change program behavior to satisfy lint; if a lint rule conflicts with correctness, add a `# noqa` with explanation
- [x] Final `ruff check rex/ tests/ *.py scripts/ utils/ plugins/` reports zero errors

---

### US-004: Black Formatting -- Full Codebase Pass

**Description:** As a developer, I need `black --check` to pass on every Python file so formatting is consistent.

**Acceptance Criteria:**

- [x] Run `black --check --diff rex/ tests/ *.py scripts/ utils/ plugins/` capturing all reformatting needed
- [x] Run `black rex/ tests/ *.py scripts/ utils/ plugins/` to apply formatting
- [x] Final `black --check rex/ tests/ *.py scripts/ utils/ plugins/` reports "All done! ✨ 🍰 ✨" with no changes needed
- [x] Verify no behavioral changes introduced (formatting only)

---

### US-005: Run Full Test Suite -- Triage Failures

**Description:** As a developer, I need to run the entire pytest suite, capture every failure, and produce a triage report so subsequent stories can fix failures systematically.

**Acceptance Criteria:**

- [x] Run `pytest -q --tb=short -x` first to find the earliest hard crash (if any)
- [x] Then run `pytest -q --tb=short` (no `-x`) to collect ALL failures
- [x] Capture output to `test-audit-results.txt` at repo root
- [x] Create a triage file `test-audit-triage.json` grouping failures by subsystem:
  - `core` (assistant, config, cli, doctor, llm_client)
  - `voice` (voice_loop, wakeword, tts, audio, wake_acknowledgment, stt)
  - `autonomy` (all test_autonomy_* files)
  - `integrations` (email, calendar, messaging, sms, github, ha_bridge, woocommerce, wordpress)
  - `dashboard_notifications` (dashboard, notification, notifications)
  - `tools_planner_workflow` (tool_registry, tool_router, planner, executor, workflow)
  - `memory_knowledge` (memory, knowledge_base, conversation_memory)
  - `security` (credentials, security_audit, auth, policy, validation)
  - `infrastructure` (event_bus, event_triggers, retry, graceful_shutdown, logging, health, startup, db_pool, rate_limiter)
  - `computers` (computers, windows_agent, os_automation, browser_automation)
  - `gui` (all test_us149-166 files)
  - `docs_install` (repo_integrity, install_scripts, readme, doctor)
  - `misc` (plugin_loader, web_search, plex, vscode, flask_proxy, env_schema)
- [x] For each failure, record: test file, test name, error type, one-line summary
- [x] No code changes in this story -- analysis only

---

### US-006: Fix Core Subsystem Test Failures

**Description:** As a developer, I need all core subsystem tests to pass (assistant, config, CLI, doctor, LLM client).

**Acceptance Criteria:**

- [x] All tests pass in: `test_assistant.py`, `test_rex_assistant.py`, `test_config_loading.py`, `test_config_cli.py`, `test_config_openai.py`, `test_cli.py`, `test_cli_devtools.py`, `test_cli_tools.py`, `test_doctor.py`, `test_llm_client.py`
- [x] Fixes do not change public API signatures unless the test was testing the wrong signature
- [x] Ruff check passes on all modified files
- [x] `pytest -q tests/test_assistant.py tests/test_rex_assistant.py tests/test_config_loading.py tests/test_config_cli.py tests/test_config_openai.py tests/test_cli.py tests/test_cli_devtools.py tests/test_cli_tools.py tests/test_doctor.py tests/test_llm_client.py` reports zero failures

---

### US-007: Fix Voice Subsystem Test Failures

**Description:** As a developer, I need all voice-related tests to pass (voice loop, wakeword, TTS, audio, STT, wake acknowledgment).

**Acceptance Criteria:**

- [x] All tests pass in: `test_voice_loop.py`, `test_voice_loop_fixes.py`, `test_voice_loop_optional_imports.py`, `test_wakeword.py`, `test_wakeword_callback.py`, `test_wakeword_model_selection.py`, `test_wakeword_utils.py`, `test_wake_acknowledgment.py`, `test_tts_voices.py`, `test_audio_config.py`, `test_audio_device_selection.py`, `test_placeholder_voice.py`, `test_optional_voice_id_imports.py`, `test_voice_id_mvp.py`, `test_voice_identity_fallback.py`
- [x] All US-series voice tests pass: `test_us017_wake_word_detection.py` through `test_us020_full_voice_loop.py`, `test_us135_tts_text_delivery.py` through `test_us138_voice_roundtrip.py`, `test_us156_voice_toggle.py` through `test_us158_voice_transcript.py`, `test_us167_voice_latency.py` through `test_us174_voice_max_tokens.py`
- [x] Hardware-dependent tests (microphone, audio playback) use mocks and do not require actual hardware
- [x] Ruff check passes on all modified files

---

### US-008: Fix Autonomy Subsystem Test Failures

**Description:** As a developer, I need all autonomy subsystem tests to pass.

**Acceptance Criteria:**

- [x] All tests pass in: `test_autonomy_alternatives.py`, `test_autonomy_apply_preferences.py`, `test_autonomy_budget_config.py`, `test_autonomy_clarifier.py`, `test_autonomy_cost_estimator.py`, `test_autonomy_cost_tracking.py`, `test_autonomy_feedback.py`, `test_autonomy_feedback_injection.py`, `test_autonomy_goal_graph.py`, `test_autonomy_goal_graph_runner.py`, `test_autonomy_goal_parser.py`, `test_autonomy_history.py`, `test_autonomy_history_integration.py`, `test_autonomy_models.py`, `test_autonomy_preference_learner.py`, `test_autonomy_preferences.py`, `test_autonomy_replanner.py`, `test_autonomy_retry.py`, `test_autonomy_runner.py`, `test_autonomy_tool_cache.py`
- [x] `pytest -q tests/test_autonomy_*.py` reports zero failures
- [x] Ruff check passes on all modified files

---

### US-009: Fix Integrations Subsystem Test Failures

**Description:** As a developer, I need all integration tests to pass (email, calendar, messaging/SMS, GitHub, Home Assistant, WooCommerce, WordPress).

**Acceptance Criteria:**

- [x] All tests pass in: `test_email_service.py`, `test_email_service_backend.py`, `test_email_account_config.py`, `test_email_account_router.py`, `test_email_backend_imap_smtp.py`, `test_calendar_service.py`, `test_calendar_ics_backend.py`, `test_messaging_backends.py`, `test_messaging_service.py`, `test_sms_integration.py`, `test_sms_inbound_integration.py`, `test_twilio_signature.py`, `test_github_service.py`, `test_ha_bridge_optional.py`, `test_ha_tts.py`, `test_woocommerce.py`, `test_wordpress.py`, `test_geolocation.py`, `test_weather.py`
- [x] All US-series integration tests pass: `test_us042` through `test_us045`, `test_us055` through `test_us057`, `test_us078` through `test_us092`
- [x] All `test_integrations_*.py` files pass
- [x] External services are mocked (no real API calls)
- [x] Ruff check passes on all modified files

---

### US-010: Fix Dashboard and Notifications Test Failures

**Description:** As a developer, I need all dashboard and notification tests to pass.

**Acceptance Criteria:**

- [x] All tests pass in: `test_dashboard.py`, `test_dashboard_store.py`, `test_notification.py`, `test_notification_email_channel.py`, `test_notification_sse.py`, `test_notifications.py`, `test_notifications_digest.py`, `test_notifications_escalation.py`, `test_notifications_models.py`, `test_notifications_quiet_hours.py`, `test_notifications_router.py`
- [x] All US-series tests pass: `test_us030_notification_routing.py`, `test_us031_dashboard_notifications.py`, `test_us046_dashboard_server.py`, `test_us047_dashboard_auth.py`, `test_us088` through `test_us092`
- [x] Ruff check passes on all modified files

---

### US-011: Fix Tools, Planner, and Workflow Test Failures

**Description:** As a developer, I need all tool registry, tool router, planner, executor, and workflow tests to pass.

**Acceptance Criteria:**

- [x] All tests pass in: `test_tool_registry.py`, `test_tool_router.py`, `test_tool_router_policy.py`, `test_planner_executor.py`, `test_llm_planner.py`, `test_workflow_engine.py`, `test_workflow_cli.py`, `test_replay.py`
- [x] All US-series tests pass: `test_us021` through `test_us029`, `test_us034` through `test_us039`, `test_us059` through `test_us065`
- [x] All CLI planner/scheduler tests pass: `test_cli_planner_executor.py`, `test_cli_scheduler.py`, `test_cli_scheduler_email_calendar.py`, `test_cli_email_accounts.py`, `test_cli_memory_kb.py`, `test_cli_messaging_notification.py`
- [x] Ruff check passes on all modified files

---

### US-012: Fix Memory and Knowledge Base Test Failures

**Description:** As a developer, I need all memory and knowledge base tests to pass.

**Acceptance Criteria:**

- [x] All tests pass in: `test_memory.py`, `test_memory_cleanup.py`, `test_memory_utils.py`, `test_conversation_memory.py`, `test_knowledge_base.py`, `test_profile_manager.py`
- [x] US-series tests pass: `test_us032_memory_storage.py`, `test_us033_user_profiles.py`, `test_us040_knowledge_ingestion.py`, `test_us041_knowledge_queries.py`, `test_us070_memory_search.py`, `test_us074_document_indexing.py`
- [x] Ruff check passes on all modified files

---

### US-013: Fix Security and Policy Test Failures

**Description:** As a developer, I need all security, credentials, auth, and policy tests to pass.

**Acceptance Criteria:**

- [x] All tests pass in: `test_credentials.py`, `test_security_audit.py`, `test_policy.py`, `test_pc_run_policy.py`, `test_env_schema.py`, `test_env_writer.py`
- [x] US-series tests pass: `test_us053_secret_management.py`, `test_us054_api_key_validation.py`, `test_us093_dependency_scan.py`, `test_us094_input_validation.py`, `test_us095_auth_session_security.py`, `test_us096_secret_scan.py`, `test_us097_security_headers.py`, `test_us131_security_scan.py`
- [x] Ruff check passes on all modified files

---

### US-014: Fix Infrastructure Test Failures

**Description:** As a developer, I need all infrastructure tests to pass (event bus, retry, shutdown, logging, health, startup, db pool, rate limiting, error handling).

**Acceptance Criteria:**

- [ ] All tests pass in: `test_event_bus.py`, `test_event_queue.py`, `test_escalation.py`, `test_followup_engine.py`, `test_service_supervisor.py`, `test_flask_proxy.py`, `test_rex_speak_api.py`, `test_scheduler.py`, `test_retention_scheduling.py`, `test_transformers_shim.py`
- [ ] US-series tests pass: `test_us028_event_bus.py`, `test_us029_event_triggers.py`, `test_us036_scheduler.py`, `test_us037_automation_registry.py`, `test_us098` through `test_us102`, `test_us103` through `test_us116`, `test_us117` through `test_us122`, `test_us127` through `test_us129`
- [ ] Ruff check passes on all modified files

---

### US-015: Fix Computers, GUI, and Browser Automation Test Failures

**Description:** As a developer, I need all computer control, GUI, and browser automation tests to pass.

**Acceptance Criteria:**

- [ ] All tests pass in: `test_computers.py`, `test_windows_agent.py`, `test_os_automation.py`, `test_browser_automation.py`, `test_vscode_service.py`
- [ ] US-series tests pass: `test_us038_application_launching.py`, `test_us039_browser_automation.py`, `test_us072_process_monitoring.py`
- [ ] All GUI tests pass: `test_us149_gui_shell.py` through `test_us166_responsive_layout.py`
- [ ] Platform-specific code uses appropriate mocks on non-target platforms
- [ ] Ruff check passes on all modified files

---

### US-016: Fix Documentation, Installation, and Repo Integrity Test Failures

**Description:** As a developer, I need all doc-validation, install-script, and repo-integrity tests to pass.

**Acceptance Criteria:**

- [ ] All tests pass in: `test_repo_integrity.py`, `test_repository_integrity.py`, `test_install_scripts.py`, `test_contracts_core.py`
- [ ] US-series tests pass: `test_us050_web_ui_server.py`, `test_us051_chat_interface.py`, `test_us052_voice_interface.py`, `test_us057_ci_pipeline.py`, `test_us113_production_config_defaults.py`, `test_us123_deployment_guide.py` through `test_us126_api_reference.py`, `test_us129_smoke.py`, `test_us139_install_scripts.py` through `test_us146_readme_visual.py`, `test_us147_first_run.py`, `test_us148_friendly_errors.py`
- [ ] Ruff check passes on all modified files

---

### US-017: Fix Remaining Miscellaneous Test Failures

**Description:** As a developer, I need all remaining test files not covered by previous stories to pass.

**Acceptance Criteria:**

- [ ] All tests pass in: `test_plugin_loader.py`, `test_web_search_plugin.py`, `test_inbound_store.py`, `test_inbound_user_association.py`, `test_inbound_webhook.py`, `test_inbound_webhook_wiring.py`, `test_wc_write_actions.py`, `test_us048_plex_api_client.py`, `test_us049_plex_playback_control.py`, `test_time_weather_integration.py`, `test_identity.py`
- [ ] Any test files not explicitly listed in US-006 through US-016 are identified and confirmed passing
- [ ] Ruff check passes on all modified files

---

### US-018: Coverage Gap Analysis

**Description:** As a developer, I need to identify which source modules have no corresponding tests so I can prioritize new test creation.

**Acceptance Criteria:**

- [ ] Run `pytest --cov=rex --cov-report=term-missing -q` and capture output to `test-audit-coverage.txt`
- [ ] Generate a list of all `rex/**/*.py` modules with 0% coverage or no test file counterpart
- [ ] Rank uncovered modules by criticality: `assistant.py`, `cli.py`, `config.py`, `llm_client.py`, `tool_router.py` are highest priority
- [ ] Save the gap analysis to `test-audit-coverage-gaps.json` with fields: module_path, current_coverage_pct, has_test_file, priority (high/medium/low)
- [ ] No code changes in this story -- analysis only

---

### US-019: Write Tests for High-Priority Uncovered Modules (Batch 1)

**Description:** As a developer, I need smoke-level tests for the highest-priority uncovered modules identified in US-018.

**Acceptance Criteria:**

- [ ] For each high-priority uncovered module (up to 10 modules), create a test file in `tests/` following the existing naming convention `test_<module_name>.py`
- [ ] Each test file must include at minimum: an import test (module loads without error), and 2-3 tests exercising core functions with mocked dependencies
- [ ] Tests must not require real external services, hardware, or API keys
- [ ] All new tests pass: `pytest -q tests/test_<new_file>.py`
- [ ] Ruff check and black formatting pass on all new files

---

### US-020: Write Tests for Medium-Priority Uncovered Modules (Batch 2)

**Description:** As a developer, I need smoke-level tests for medium-priority uncovered modules identified in US-018.

**Acceptance Criteria:**

- [ ] For each medium-priority uncovered module (up to 10 modules), create a test file in `tests/`
- [ ] Each test file includes: import test + 1-2 functional tests with mocks
- [ ] All new tests pass
- [ ] Ruff check and black formatting pass on all new files

---

### US-021: Final Full Verification Run

**Description:** As a developer, I need a final clean run of the entire test suite plus lint to confirm everything is green.

**Acceptance Criteria:**

- [ ] `ruff check rex/ tests/ *.py scripts/ utils/ plugins/` reports zero errors
- [ ] `black --check rex/ tests/ *.py scripts/ utils/ plugins/` reports no changes needed
- [ ] `pytest -q` runs to completion with zero failures and zero errors
- [ ] Capture final results to `test-audit-final-results.txt` with: total tests collected, total passed, total time, ruff status, black status
- [ ] If any failures remain, document them in `test-audit-known-issues.md` with explanation of why they cannot be fixed (e.g., requires real hardware, external service, GPU)
- [ ] Remove temporary audit files (`test-audit-env.txt`, `test-audit-results.txt`, `test-audit-triage.json`) -- keep only `test-audit-final-results.txt` and `test-audit-known-issues.md` (if needed)
- [ ] Verify no regressions: the fix process has not broken previously-passing tests

---

## Non-Goals

- No new feature development; this is a quality/testing audit only
- No GPU/CUDA testing (requires hardware not available in CI)
- No real external API calls (all integrations must be mocked)
- No microphone or speaker hardware testing (audio tests use mocks)
- No performance optimization beyond what's needed to fix test failures
- No refactoring for style preferences; only fix what's broken or violates lint rules
- No changes to `pyproject.toml` dependencies unless a test fix requires it
- No changes to `.env` or secrets
- The uploaded `index.js` (Kapture MCP server) is out of scope

## Technical Considerations

- **Python version:** 3.9-3.13 supported; 3.10+ preferred. Tests should pass on the installed version.
- **Optional dependencies:** Many modules (TTS, Whisper, torch, openwakeword) are optional. Tests for these must use `pytest.importorskip()` or conditional `find_spec()` checks and skip gracefully if unavailable.
- **CLAUDE.md learned rules:** When lazy-importing modules that trigger side-effect imports (e.g., TTS importing from transformers), use `find_spec()` to check availability BEFORE calling `import_module()`.
- **Two voice_loops:** The root-level `voice_loop.py` and `rex/voice_loop.py` are separate implementations. `rex_loop.py` uses the ROOT `voice_loop.py`. Do not confuse them.
- **Config split:** Secrets in `.env`, runtime config in `config/rex_config.json`. Never commit secrets.
- **Lint preflight:** Both `ruff check` and `black --check` must pass before any code is considered complete.
- **Existing conftest.py:** Uses `REX_TESTING=true` env var and adds repo root + tests dir to `sys.path`. New tests should rely on this, not add their own path manipulation.
- **Test fixtures:** Shared fixtures live in `tests/fixtures/`. Use them where possible.
