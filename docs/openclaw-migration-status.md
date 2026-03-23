# OpenClaw Migration Status

Tracks every Rex module's migration state as Rex pivots to an OpenClaw-based architecture.

**Classifications:**
- **Replace** — module will be replaced by an OpenClaw equivalent
- **Keep** — module is Rex-specific and stays unchanged
- **Keep + Wrap** — module stays but gets wrapped as an OpenClaw hook/adapter
- **Keep + Adapt** — module stays but gets a compatibility adapter
- **Keep + Update** — module stays but will need import updates as others retire
- **Wrap** — module's logic is preserved but execution delegates to OpenClaw
- **Retire** — module is deleted once OpenClaw fully covers it

**Statuses:**
- `Pending` — not yet started
- `Marked` — freeze/wrap marker added
- `Contracted` — Protocol/interface defined in `rex/contracts/`
- `Bridged` — OpenClaw bridge implemented, dual-mode running
- `Migrated` — fully on OpenClaw, old code retired

---

| Module | Classification | Status | Notes |
|--------|----------------|--------|-------|
| `rex/assistant.py` | Wrap | Pending | Central orchestration hub. Delegate to OpenClaw agent via voice_bridge. Keep as thin coordinator. |
| `rex/browser_automation.py` | Replace | Bridged (US-P4-027) | Generic Playwright wrapper. BrowserBridge wraps service. Both cli callers migrated to bridge. Login + step-format gaps remain (backlog). See audit notes below. |
| `rex/dashboard/__init__.py` | Replace | Pending | Flask dashboard. OpenClaw provides dashboard/UI. Run both in parallel during transition. |
| `rex/dashboard/routes.py` | Replace | Pending | Dashboard routes. Retires with dashboard. |
| `rex/dashboard/sse.py` | Replace | Pending | SSE streaming. Retires with dashboard. |
| `rex/dashboard/auth.py` | Replace | Pending | Dashboard auth. Retires with dashboard. |
| `rex/dashboard_store.py` | Replace | Pending | Dashboard persistence. Retires when dashboard retires. |
| `rex/messaging_backends/` | Replace | Pending | Twilio, SMS, webhooks (11 files). OpenClaw owns channels. Migrate last in Phase 4 due to webhook complexity. |
| `rex/messaging_service.py` | Replace | Pending | Messaging orchestration. Retires with messaging_backends. |
| `rex/integrations/message_router.py` | Replace | Pending | Routes messages between channels. Retires with messaging. |
| `rex/tool_registry.py` | Replace | Pending | Tool metadata + health checks. OpenClaw has skill/tool system. Define Protocol first (Phase 1), then bridge (Phase 4). |
| `rex/tool_router.py` | Replace | Audited (US-P4-002) | Central tool dispatch (960 lines). Highest-risk replacement. Feature flag required. Test every tool through bridge before retirement. See audit notes below. Tool classification: 6 generic-replace, 3 adapter-needed, 2 Rex-specific. |
| `rex/plugin_loader.py` | Replace | Pending | Dynamic plugin discovery (56 lines). OpenClaw has plugins. Small file, easy replacement. |
| `rex/executor.py` | Replace | Pending | Task execution engine. Replaced by OpenClaw task execution via workflow bridge. |
| `rex/event_bus.py` | Replace | Audited (US-P4-014) | Pub-sub event system (436 lines). Dual API (simple + rich) must be preserved in bridge. 10 Rex-specific event types; 0 framework-level. Wildcard subscriber requires full fan-out in bridge. See audit notes below. |
| `rex/computers/` | Replace | Pending | Windows agent server/client (~400 lines, 5 files). Replace with OpenClaw workspace/agent model. |
| `rex/workflow.py` | Wrap | Audited (US-P4-028) | Workflow data models (668 lines). Rex-specific definitions. 7 callers. Models translate to OpenClaw skill/task definitions; bridge translates at execution time. See audit notes below. |
| `rex/workflow_runner.py` | Wrap | Audited (US-P4-028) | Workflow execution (864 lines). Has Rex policy hooks. 3 callers (cli, executor, approval_adapter). Bridge preserves policy gating; Rex policy is authority. See audit notes below. |
| `rex/autonomy/__init__.py` | Wrap | Pending | Autonomy package init. Wraps OpenClaw multi-agent primitives. |
| `rex/autonomy/runner.py` | Wrap | Pending | Autonomy runner. High-level planning logic wraps OpenClaw primitives. Keep Rex's goal decomposition and replanning. |
| `rex/autonomy/llm_planner.py` | Wrap | Pending | LLM-based planner. Preserved; wraps OpenClaw. |
| `rex/autonomy/rule_planner.py` | Wrap | Pending | Rule-based planner. Preserved; wraps OpenClaw. |
| `rex/scheduler.py` | Wrap | Pending | Cron-like scheduling (675 lines). Evaluate OpenClaw scheduling; wrap if available, keep if not. |
| `rex/planner.py` | Wrap | Pending | Task planning (640 lines). Rex's planning logic wraps OpenClaw primitives. |
| `rex/notification.py` | Wrap | Pending | Notification system (884 lines). Route through OpenClaw's notification/event system if available. |
| `rex/policy.py` | Keep + Wrap | Audited (US-P3-007) | Policy models (150 lines). Rex-specific risk classification. Keep models; wrap as OpenClaw middleware. Rex policy is always the authority. See audit notes below. |
| `rex/policy_engine.py` | Keep + Wrap | Audited (US-P3-007) | Policy evaluation (350 lines). Keep engine; wrap as OpenClaw hook. See audit notes below. |
| `rex/identity.py` | Wrap | Audited (US-P3-012) | User identity resolution (322 lines). Map to OpenClaw session/user model. Keep Rex's resolution logic. See audit notes below. |
| `rex/profile_manager.py` | Wrap | Audited (US-P3-015) | Profile merging (100 lines). Keep merge logic; wire into OpenClaw agent config. See audit notes below. |
| `rex/voice_identity/` | Keep | Pending | Speaker recognition (7 files). Uniquely Rex. No OpenClaw equivalent. Phase 6: feed into OpenClaw session identity. |
| `rex/wakeword/` | Keep | Pending | Wake word detection (4 files). Uniquely Rex. Unchanged in migration. |
| `rex/voice_loop.py` | Keep | Pending | Core voice loop (800 lines). Update to call OpenClaw backend via voice_bridge. Feature flag for rollback. |
| `rex/voice_loop_optimized.py` | Keep | Pending | Low-latency voice loop (550 lines). Same treatment as voice_loop.py. |
| `rex/ha_bridge.py` | Keep | Audited (US-P5-001) | Home Assistant bridge (600 lines). `ha_tool.py` already wraps it. Full audit below. |
| `rex/ha_tts/` | Keep | Audited (US-P5-002) | HA TTS integration (3 files). `build_ha_tts_client` factory; `HaTtsClient.speak()` is the one callable. Full audit below. |
| `rex/wordpress/` | Keep | Audited + Bridged (US-P5-008/009) | WordPress client (3 files, read-only). `wordpress_tool.py` wraps service as `wordpress_health_check`. Full audit below. |
| `rex/woocommerce/` | Keep | Audited + Bridged (US-P5-012/013) | WooCommerce client (4 files, read + approval-gated write). 5 tools: list_orders, list_products, set_order_status, create_coupon, disable_coupon. Write policy preserved. |
| `rex/plex_client.py` | Keep | Pending | Plex media control. Register as OpenClaw skill in Phase 5. |
| `rex/memory.py` | Keep + Adapt | Audited (US-P3-001) | See audit notes below. |
| `rex/memory_utils.py` | Keep | Audited (US-P3-004) | See audit notes below. |
| `rex/llm_client.py` | Keep | Pending | Multi-provider LLM client. Orthogonal to migration. Not in scope. |
| `rex/config.py` | Keep | Pending | Pydantic settings (600 lines). Add OpenClaw-specific fields as needed. |
| `rex/cli.py` | Keep + Update | Pending | CLI (4941 lines). Update imports as modules are retired. Do not rewrite. One command at a time. |
| `rex/app.py` | Retire | Pending | Flask app factory. Retires when dashboard retires and OpenClaw handles HTTP. |
| `rex/api_key_auth.py` | Retire | Pending | API key auth. Retires when OpenClaw handles auth. |
| `rex/credentials.py` | Keep | Pending | Credential management (450 lines). May need adapter for OpenClaw tools that need credentials. |
| `rex/email_backends/` | Keep | Pending | IMAP/SMTP email (~600 lines). Rex-specific. Register as OpenClaw skill. |
| `rex/email_service.py` | Keep | Pending | Email orchestration (662 lines). Register as OpenClaw skill. |
| `rex/calendar_backends/` | Keep | Pending | Calendar integrations (~500 lines). Rex-specific. Register as OpenClaw skill. |
| `rex/calendar_service.py` | Keep | Pending | Calendar orchestration (700 lines). Register as OpenClaw skill. |
| `rex/audit.py` | Keep | Pending | Audit logging. Security-critical. Stays; may also feed into OpenClaw's audit if available. |

---

## Audit Notes

### rex/memory.py — US-P3-001

**Public API (`__all__`):**

Conversation history (re-exported from `rex.memory_utils`):
- `trim_history(history, limit)` — trim conversation list to limit
- `append_history_entry(user_id, role, content)` — append turn to file-based history
- `load_recent_history(user_id, n)` — load N recent turns from file
- `export_transcript(user_id, output_path)` — export history to text file
- `load_memory_profile(user_id)` — load user profile dict from JSON
- `load_all_profiles()` — load all user profiles
- `load_users_map()` — load user-id→name mapping
- `resolve_user_key(user_id)` — normalise user key
- `extract_voice_reference(text)` — extract name/pronoun references from transcript

Working memory (singleton, short-term, disk-backed):
- `WorkingMemory` — class: `add_entry`, `get_recent`, `get_recent_with_timestamps`, `clear`, `stats`
- `get_working_memory()` — global singleton getter
- `set_working_memory(wm)` — global singleton setter (for testing)

Long-term memory (structured, expiry-aware, disk-backed):
- `MemoryEntry` — Pydantic model: `is_expired()`, `to_safe_dict()`
- `LongTermMemory` — class: `add_entry`, `get_entry`, `search`, `forget`, `run_retention_policy`, `compact`, `list_categories`, `count_by_category`, `stats`
- `get_long_term_memory()` — global singleton getter
- `set_long_term_memory(ltm)` — global singleton setter (for testing)

Convenience functions:
- `add_user_preference(key, value, expires_in, sensitive)`
- `get_user_preferences(key)`
- `add_fact(topic, content, expires_in)`
- `remember_context(summary)`
- `get_recent_context(n)`
- `schedule_memory_cleanup(scheduler, interval_seconds, job_id)`

**Callers (by import pattern):**

| File | What it uses |
|------|-------------|
| `rex/assistant.py` | `trim_history` (via `from .memory import`) |
| `rex/app.py` | `get_long_term_memory`, `get_working_memory` |
| `rex/cli.py` | `get_long_term_memory`, `get_working_memory` (lazy import in command) |
| `rex_memories_bridge.py` | `get_long_term_memory` (lazy imports inside functions) |
| `voice_loop.py` | uses `rex.memory_utils` directly (not rex.memory) |
| `rex_speak_api.py` | uses `rex.memory_utils` directly (not rex.memory) |
| Tests | `WorkingMemory`, `LongTermMemory`, convenience functions |

**Classification for OpenClaw adapter:**

| API group | Rex-specific? | Adapter priority |
|-----------|--------------|-----------------|
| `trim_history`, `append_history_entry`, `load_recent_history` | Rex-specific (file-based) | High — used by assistant.py voice path |
| `WorkingMemory` / `get_working_memory` | Generic pattern, Rex impl | Medium — used by app.py and cli.py |
| `LongTermMemory` / `get_long_term_memory` | Generic pattern, Rex impl | Medium — used by app.py, cli.py, memories bridge |
| `load_memory_profile`, `load_users_map` etc | Rex-specific (file-based) | Low — identity/profile concern, not conversation |
| `schedule_memory_cleanup` | Rex-specific (scheduler API) | Low — utility, not core path |

**Key findings:**
- `trim_history` is the most critical caller path (assistant.py → voice loop)
- `WorkingMemory` and `LongTermMemory` use file-based persistence (`data/memory/`)
- OpenClaw adapter should delegate to these classes and add a future hook for OpenClaw storage
- No callers import `MemoryEntry` directly except tests — safe to wrap transparently

---

### Audit Notes: rex/memory_utils.py (US-P3-004)

**Public API (`__all__`):**

Conversation history:
- `trim_history(history, limit)` — trim an in-memory list to the N most recent items
- `append_history_entry(user_key, entry, memory_root, max_turns)` — append turn to JSONL file; enforces max_turns limit
- `load_recent_history(user_key, limit, memory_root)` — read recent turns from JSONL; returns `[]` if no file
- `export_transcript(user_key, conversation, transcripts_dir)` — write conversation to dated text file in transcripts dir

Identity / profile:
- `load_users_map(users_path)` — load `users.json` email→username mapping
- `resolve_user_key(identifier, users_map, memory_root, profiles)` — resolve voice/email/name to a canonical user key
- `load_memory_profile(user_key, memory_root)` — load `core.json` for a user; enforces size limit
- `load_all_profiles(memory_root)` — load all `core.json` files under Memory/
- `extract_voice_reference(profile, user_key, memory_root, repo_root)` — resolve voice sample path from profile dict

Private helpers (not exported): `_sanitize_user_key`, `_validate_path_within`, `_ensure_directory`,
`_history_path`, `_metadata_path`, `_looks_like_placeholder`, `_normalise_voice_path`

**Caller map:**

| Caller | Functions used |
|--------|----------------|
| `rex/memory.py` | re-exports all 9 via `from .memory_utils import ...` |
| `voice_loop.py` (root) | `append_history_entry`, `export_transcript` (direct import) |
| `rex_speak_api.py` | `extract_voice_reference`, `load_all_profiles` (direct import) |
| `memory_utils.py` (root compat) | re-exports all 9 for legacy callers |
| `flask_proxy.py` | `load_memory_profile`, `load_users_map`, `resolve_user_key` (via root compat) |
| `gui.py` | `load_recent_history` (via root compat) |
| `tests/test_memory_utils.py` | `append_history_entry`, `export_transcript`, others (via root compat) |

**Classification:**

| Function | Rex-specific? | Adapter priority |
|----------|--------------|-----------------|
| `trim_history` | Generic pattern, Rex impl | High — already wrapped by MemoryAdapter (US-P3-002) |
| `append_history_entry` | Generic pattern, Rex impl | High — already wrapped by MemoryAdapter |
| `load_recent_history` | Generic pattern, Rex impl | High — already wrapped by MemoryAdapter |
| `export_transcript` | Rex-specific (file path conventions) | Low — transcript export, not core conversation path |
| `load_users_map` | Rex-specific (users.json format) | Low — identity concern, not OpenClaw storage |
| `resolve_user_key` | Rex-specific (voice/email matching) | Low — identity concern |
| `load_memory_profile` | Rex-specific (core.json format) | Low — profile concern |
| `load_all_profiles` | Rex-specific (Memory/ directory layout) | Low — profile concern |
| `extract_voice_reference` | Rex-specific (voice cloning) | Low — TTS concern, not agent storage |

**Key findings:**
- The three conversation-history functions (`trim_history`, `append_history_entry`, `load_recent_history`) are already wrapped by `MemoryAdapter` (US-P3-002/003) — no further adapter work needed for them.
- Five identity/profile functions are Rex-specific and should remain in `rex.memory_utils` unchanged; they have no OpenClaw equivalent to map to.
- `export_transcript` is Rex-specific (path conventions, config toggle); no adapter needed.
- Root-level `memory_utils.py` is a legacy compat shim — `flask_proxy.py`, `gui.py`, and old tests use it. Do not remove until those callers are migrated.
- Security: `_sanitize_user_key` and `_validate_path_within` provide path-traversal protection — must be preserved in any refactor.

---

### Audit Notes: rex/policy.py and rex/policy_engine.py (US-P3-007)

**rex/policy.py — Public API (`__all__`):**

- `RiskLevel` — re-exported from `rex.contracts`; enum: `LOW`, `MEDIUM`, `HIGH`
- `ActionPolicy(BaseModel)` — tool policy config: `tool_name`, `risk`, `allow_auto`, `allowed_recipients`, `denied_recipients`, `allowed_domains`, `denied_domains`
- `PolicyDecision(BaseModel)` — evaluation result: `allowed`, `reason`, `requires_approval`, `denied`

**rex/policy_engine.py — Public API (`__all__`):**

- `PolicyEngine` — evaluation class
  - `__init__(policies, default_policy)` — merge custom policies over DEFAULT_POLICIES
  - `policies` (property) — read-only copy of current registry
  - `get_policy(tool_name)` — look up policy or return default
  - `decide(tool_call, metadata)` — evaluate and return `PolicyDecision`
  - `add_policy(policy)` — register/override a policy at runtime
  - `remove_policy(tool_name)` — remove a policy; returns bool
- `DEFAULT_POLICIES` — list of 11 built-in `ActionPolicy` objects (see gated tools below)
- `get_policy_engine()` — module-level singleton accessor
- `reset_policy_engine()` — reset singleton (for tests)

**PolicyDecision consumers (who checks `.denied` / `.requires_approval`):**

| Caller | How it uses PolicyDecision |
|--------|---------------------------|
| `rex/tool_router.py:execute_tool()` | Primary gateway — checks `.denied` (raises `PolicyDeniedError`), `.requires_approval` (raises `ApprovalRequiredError`); `skip_policy_check=True` bypasses |
| `rex/computers/pc_run_policy.py` | Constructs `PolicyDecision` directly for `pc_run` approval gating |
| `rex/workflow_runner.py` | Uses `get_policy_engine()` for workflow step policy checks |
| `rex/executor.py` | Uses `get_policy_engine()` for planner-driven tool execution |
| `rex/browser_automation.py` | Uses `get_policy_engine()` for browser tool actions |
| `rex/cli.py` | Passes `get_policy_engine()` to `Planner` on `plan` command |

**Policy-gated tools (DEFAULT_POLICIES):**

| Tool | Risk | Auto-execute? |
|------|------|---------------|
| `time_now` | LOW | Yes |
| `weather_now` | LOW | Yes |
| `web_search` | LOW | Yes |
| `send_email` | MEDIUM | No (requires approval) |
| `calendar_create_event` | MEDIUM | No |
| `calendar_delete_event` | MEDIUM | No |
| `home_assistant_call_service` | MEDIUM | No |
| `execute_command` | HIGH | No |
| `pc_run` | HIGH | No |
| `file_write` | HIGH | No |
| `file_delete` | HIGH | No |

**Key findings:**
- `rex/tool_router.py:execute_tool()` is the single policy enforcement point for all tool calls. The `skip_policy_check=True` flag (used by OpenClaw tool adapters in Phase 2) bypasses the engine entirely.
- OpenClaw adapter strategy: wrap `PolicyEngine.decide()` as an OpenClaw pre-execution hook so Rex policy fires before any OpenClaw-dispatched tool. Rex policy is always the authority.
- `PolicyEngine` is injectable (constructor param, no globals forced) — easy to test and wrap.
- `get_policy_engine()` singleton is reset-safe via `reset_policy_engine()` — existing test infrastructure is solid.
- Both modules are marked `# OPENCLAW-WRAP` — they were pre-identified for wrapping.

---

### Audit Notes: rex/identity.py (US-P3-012)

**Public API (`__all__`):**

Session state (OS temp-file backed):
- `get_session_user()` — read `active_user` from `rex-ai/session.json`; returns `str | None`
- `set_session_user(user_id)` — write `active_user` to session file; persists across CLI invocations
- `clear_session_user()` — remove `active_user` from session file

User resolution:
- `resolve_active_user(explicit_user, config)` — 4-level priority: explicit arg → session file → `runtime.active_user` → `runtime.user_id` in config; returns `str | None`
- `require_active_user(explicit_user, config, action)` — calls `resolve_active_user` or raises `SystemExit` with helpful message

Profile management (file-based in `Memory/`):
- `create_user_profile(user_id, name, role, preferences, memory_dir, overwrite)` — writes `core.json`; raises `ValueError` on invalid id, `FileExistsError` if exists and not overwrite
- `get_user_profile(user_id, memory_dir)` — loads `core.json`; returns `dict | None`
- `update_user_preferences(user_id, preferences, memory_dir)` — merges preference dict into existing profile; returns `bool`
- `list_known_users()` — scans `Memory/` for subdirs with `core.json`; returns `[{id, name, role}]`

Private helpers (not exported): `_session_state_path`, `_known_user_ids`, `_load_session`, `_save_session`

**Session state behavior:**
- Stored at `LOCALAPPDATA\rex-ai\session.json` (Windows) or `XDG_RUNTIME_DIR/rex-ai/session.json` (Linux/Mac)
- Persists across CLI invocations until `clear_session_user()` or file deleted
- File write failures are swallowed with a warning (non-fatal)

**Caller map:**

| Caller | Functions used |
|--------|----------------|
| `rex/cli.py` | `resolve_active_user`, `get_session_user`, `set_session_user`, `list_known_users` (in `identify` command) |
| `rex/openclaw/session.py` | `resolve_active_user` (already used in session bridge US-P2-003) |
| `rex/voice_identity/fallback_flow.py` | `get_session_user`, `set_session_user`, `resolve_active_user` |
| Tests | `test_identity.py`, `test_us033_user_profiles.py`, `test_voice_id_mvp.py`, `test_voice_identity_fallback.py` |

**Classification for OpenClaw adapter:**

| Function group | Rex-specific? | Adapter priority |
|---------------|--------------|-----------------|
| `resolve_active_user`, `get/set/clear_session_user` | Rex-specific (session file) | High — session bridge already wraps this (US-P2-003) |
| `create_user_profile`, `get_user_profile`, `update_user_preferences`, `list_known_users` | Rex-specific (Memory/ layout) | Medium — profile CRUD, needs adapter for OpenClaw session |
| `require_active_user` | Rex-specific (SystemExit) | Low — CLI convenience; not in agent path |

**Key findings:**
- `rex/openclaw/session.py` (US-P2-003) already wraps `resolve_active_user` into the OpenClaw session context — the identity adapter (US-P3-013) builds on this.
- Session file is OS-temp-backed; OpenClaw adapter should delegate `get/set_session_user` to OpenClaw's session management when available, falling back to the file.
- Profile CRUD functions (`create_user_profile` etc.) are Rex-specific directory-format ops — keep as-is, expose through adapter for OpenClaw agent to query.
- Module is marked `# OPENCLAW-WRAP` — pre-identified for wrapping.

---

### Audit Notes: rex/profile_manager.py (US-P3-015)

**Public API (`__all__`):**

- `DEFAULT_PROFILES_DIR = "profiles"` — default directory constant for profile JSON files
- `load_profile(name, profiles_dir)` — load a named profile JSON from disk; validates against `profile.schema.json` if present; raises `FileNotFoundError` if profile missing
- `apply_profile(base_config, profile)` — deep-merge profile `overrides` dict into base config dict; replace `capabilities` list entirely with profile's capabilities; returns merged dict
- `get_active_profile_name(config)` — read `active_profile` from config dict; returns `"default"` if absent or falsy

**Private helpers (not exported):**
- `_deep_merge(base, overlay)` — recursive dict merge; overlay wins for non-dict values
- `_basic_validate(profile, required)` — validates required fields and type constraints on `profile_version`, `name`, `description`, `capabilities`, `overrides`
- `_validate_profile(profile, schema_path)` — schema-driven validation; no-ops if schema file missing

**Merge behavior:**
- `apply_profile()` performs a deep-recursive merge: nested dicts are merged, not replaced. Scalars and lists are replaced by the overlay value.
- Exception: `capabilities` is always wholesale-replaced (not merged) with the profile's capabilities list.
- Profile `overrides` key is optional (`{}` default); `capabilities` key is optional (`[]` default).

**Caller map:**

| Caller | Functions used |
|--------|----------------|
| `rex/config.py` (lines 235–238) | `get_active_profile_name`, `load_profile`, `apply_profile` — called at config load time to apply active profile to base config |
| `tests/test_profile_manager.py` | `apply_profile`, `get_active_profile_name`, `load_profile` |

**Classification for OpenClaw:**

| Function | Rex-specific? | OpenClaw action |
|----------|--------------|-----------------|
| `load_profile` | Rex-specific (JSON file format) | Keep as-is; already feeds into AppConfig at load time |
| `apply_profile` | Rex-specific (deep merge logic) | Keep as-is; profile applied before AppConfig constructed |
| `get_active_profile_name` | Rex-specific (config dict key) | Keep as-is |

**Key findings:**
- All three public functions are called exclusively by `rex/config.py` during `AppConfig` construction. By the time `rex/openclaw/config.py::build_agent_config()` runs, the profile is already baked into `AppConfig` (via `active_profile`, `capabilities`, and overridden config fields).
- No direct OpenClaw wiring is needed for profile_manager itself — the profile already influences `AppConfig.capabilities`, `AppConfig.active_profile`, etc., which `build_agent_config()` and `build_system_prompt()` read.
- US-P3-016 ("Wire profile manager into OpenClaw agent") means verifying that `build_agent_config()` correctly reflects profile-applied AppConfig — not adding new profile-loading code.
- Module is marked `# OPENCLAW-WRAP` — pre-identified for wrapping.

---

### Audit Notes: Approval System (US-P3-017)

**Storage convention:**

All approvals are stored as JSON files under `data/approvals/` (configurable via `approval_dir` parameter). The canonical path constant is `DEFAULT_APPROVAL_DIR = Path("data/approvals")` in `rex/workflow.py`.

**Approval record type:**

`WorkflowApproval` (Pydantic model, `rex/workflow.py` lines 192–301):

| Field | Type | Purpose |
|-------|------|---------|
| `approval_id` | `str` | Auto-generated unique ID (`apr_<12hex>`) |
| `workflow_id` | `str` | ID of the parent workflow (or sentinel string for non-workflow uses) |
| `step_id` | `str` | Step or action being approved; deterministic for idempotent re-runs |
| `status` | `Literal["pending","approved","denied","expired"]` | Current decision state |
| `reason` | `str \| None` | Reason for the decision (set at decision time) |
| `requested_by` | `str \| None` | Who/what requested (e.g., `"workflow_runner"`) |
| `decided_by` | `str \| None` | Who decided (e.g., `"cli_user"`) |
| `requested_at` | `datetime` | When approval was created (UTC) |
| `decided_at` | `datetime \| None` | When the decision was made |
| `step_description` | `str \| None` | Human-readable step context |
| `tool_call_summary` | `str \| None` | Summary of the tool call being approved |

**Modules that create or check approvals (`data/approvals/`):**

| Module | Role | Key functions |
|--------|------|---------------|
| `rex/workflow.py` | Defines `WorkflowApproval`; `DEFAULT_APPROVAL_DIR`; `generate_approval_id()`. Handles `.save()` and `.load()` I/O. | `WorkflowApproval.save(approval_dir)`, `WorkflowApproval.load(approval_id, approval_dir)` |
| `rex/workflow_runner.py` | Creates approvals when a workflow step is policy-gated (`requires_approval`). Resumes blocked workflows after approval. Exposes approve/deny/list helpers. | `WorkflowRunner._make_approval(step)`, `WorkflowRunner.resume_after_approval()`, `approve_workflow()`, `deny_workflow()`, `list_pending_approvals()`, `ApprovalBlockedError` |
| `rex/executor.py` | Wraps `WorkflowRunner`; surfaces `blocking_approval_id` in its result object. Passes `approval_dir` through. | `ExecutorResult.blocking_approval_id`, `TaskExecutor.run()`, `TaskExecutor.resume()` |
| `rex/computers/pc_run_policy.py` | Creates approvals for `pc_run` (remote command execution). Uses `WorkflowApproval` with `workflow_id="pc_run"` sentinel and deterministic `step_id` (SHA-256 of computer+command+args). | `check_pc_run_policy()` |
| `rex/woocommerce/write_policy.py` | Creates approvals for WooCommerce write actions (`wc_order_set_status`, `wc_coupon_create`, `wc_coupon_disable`). Uses `WorkflowApproval` with `WC_WRITE_WORKFLOW_ID` sentinel and deterministic `step_id`. | `check_wc_write_policy()` |
| `rex/cli.py` | User-facing `rex approvals` command: list, approve, deny, show. Calls `approve_workflow()`, `deny_workflow()`, `list_pending_approvals()`. | `cmd_approvals(args)` |
| `rex/tool_router.py` | Does **not** write to `data/approvals/`. Raises `ApprovalRequiredError` in-memory when policy says `requires_approval`. The caller (workflow_runner / executor) handles persistence. | `ApprovalRequiredError` |

**Approval flow:**

```
tool_router.execute_tool()
  → policy_engine.decide() → requires_approval
  → raise ApprovalRequiredError (in-memory only)
      ↓
workflow_runner.WorkflowRunner._run_step()
  → catch ApprovalRequiredError
  → _make_approval(step) → WorkflowApproval.save("data/approvals/")
  → raise ApprovalBlockedError(approval_id, step_id)
      ↓
executor / cli surface blocking_approval_id to user
      ↓
user: rex approvals --approve <id>
  → cli.cmd_approvals() → approve_workflow() → WorkflowApproval.load + update + save
      ↓
workflow_runner.WorkflowRunner.resume_after_approval()
  → WorkflowApproval.load → status == "approved" → continue execution
```

**Non-workflow callers (pc_run, woocommerce):**
- `pc_run_policy.py` and `woocommerce/write_policy.py` follow the same pattern: policy_engine → create `WorkflowApproval` directly → CLI polls/approves via `rex approvals`.
- They use deterministic `step_id` so re-running the same command finds the existing pending/approved record without an index.

**Classification for OpenClaw adapter:**

| Component | Rex-specific? | OpenClaw action |
|-----------|--------------|-----------------|
| `WorkflowApproval` model | Rex-specific (file JSON) | Adapter should expose create/read/update; OpenClaw may have native approval model |
| `data/approvals/` storage | Rex-specific (local files) | Adapter should allow swapping storage backend |
| `approve_workflow` / `deny_workflow` | Rex API | Wrap in `ApprovalAdapter` for OpenClaw hook |
| `list_pending_approvals` | Rex API | Wrap in `ApprovalAdapter` |
| `ApprovalBlockedError` | Rex-specific | Map to OpenClaw's gate/pause mechanism |

**Key findings:**
- `WorkflowApproval` is a Pydantic model in `rex/workflow.py` — the single source of truth for approval records.
- All approval I/O goes through `WorkflowApproval.save()` / `WorkflowApproval.load()` with `DEFAULT_APPROVAL_DIR = Path("data/approvals")`.
- Three producers: `workflow_runner._make_approval()`, `pc_run_policy.check_pc_run_policy()`, `woocommerce/write_policy.check_wc_write_policy()`.
- One consumer / decision point: `approve_workflow()` / `deny_workflow()` in `workflow_runner.py`, called via CLI.
- `tool_router.py` raises `ApprovalRequiredError` in-memory — it does NOT write files; that responsibility is always one layer up.
- US-P3-018 (`ApprovalAdapter`) should wrap: `WorkflowApproval.save/load`, `approve_workflow`, `deny_workflow`, `list_pending_approvals`, and expose an `ApprovalBlockedError`→OpenClaw gate bridge.

---

### Audit Notes: rex/tool_router.py (US-P4-001)

**Tool name / handler mapping (lines 280–288 + `supported_tools` set line 235):**

| Tool name | Handler function | Notes |
|-----------|-----------------|-------|
| `time_now` | `_execute_time_now(args, default_context)` | Returns local time/date/timezone. Resolves timezone via `_resolve_timezone()` then `ZoneInfo`. Falls back to `default_context["location"]`. No credentials required. |
| `weather_now` | `_execute_weather_now(args, default_context)` | Calls `rex.weather.get_weather()` async. Requires `OPENWEATHERMAP_API_KEY` env var. Falls back to `default_context["location"]` then `get_cached_city()`. |
| `web_search` | `_execute_web_search(args, default_context)` | Delegates to `plugins.web_search.search_web(query)`. No hard credential requirement; uses any configured search API (brave/serpapi). Returns error dict if plugin not installed. |

**Tool registry vs router gap:**

| Tool | In `supported_tools`? | In `tool_registry` builtins? | In DEFAULT_POLICIES? | Status |
|------|-----------------------|------------------------------|---------------------|--------|
| `time_now` | Yes | Yes | Yes (LOW) | Fully implemented |
| `weather_now` | Yes | Yes | Yes (LOW) | Fully implemented |
| `web_search` | Yes | Yes | Yes (LOW) | Fully implemented |
| `send_email` | **No** | Yes (stub, health=False) | Yes (MEDIUM) | Policy-gated but no handler — returns "Unknown tool" |
| `home_assistant` / `home_assistant_call_service` | **No** | `home_assistant` registered | Yes (MEDIUM) | Policy-gated but no handler |
| `calendar_create_event` | **No** | No | Yes (MEDIUM) | Policy-gated but no handler |
| `calendar_delete_event` | **No** | No | Yes (MEDIUM) | Policy-gated but no handler |
| `execute_command` | **No** | No | Yes (HIGH) | Policy-gated but no handler |
| `pc_run` | **No** | No | Yes (HIGH) | Handled by `pc_run_policy.py` + approval flow, not via `tool_router` |
| `file_write` | **No** | No | Yes (HIGH) | Policy-gated but no handler |
| `file_delete` | **No** | No | Yes (HIGH) | Policy-gated but no handler |

**Public API of `rex/tool_router.py`:**

- `TOOL_REQUEST_PREFIX = "TOOL_REQUEST:"` — sentinel prefix for LLM-emitted tool requests
- `TOOL_RESULT_PREFIX = "TOOL_RESULT:"` — sentinel prefix for tool result injection
- `ToolError` — frozen dataclass: `message`
- `PolicyDeniedError(tool, reason)` — raised when policy denies
- `ApprovalRequiredError(tool, reason)` — raised when policy requires approval (in-memory only; no file I/O)
- `CredentialMissingError(tool, missing_credentials)` — raised when tool credentials absent
- `parse_tool_request(text) -> dict | None` — parse single-line TOOL_REQUEST JSON; rejects multi-line
- `execute_tool(request, default_context, *, policy_engine, tool_registry, skip_policy_check, skip_credential_check, task_id, requested_by, skip_audit_log) -> dict` — full policy→credential→execute→audit pipeline
- `format_tool_result(tool, args, result) -> str` — format as TOOL_RESULT JSON line
- `route_if_tool_request(llm_text, default_context, model_call_fn, *, policy_engine, skip_policy_check) -> str` — full request→execute→re-call pipeline

**Key findings:**
- Only 3 of the 11 policy-gated tools have actual execution handlers in `tool_router.py`. The other 8 (`send_email`, `calendar_*`, `home_assistant_call_service`, `execute_command`, `pc_run`, `file_write`, `file_delete`) are gated by DEFAULT_POLICIES but return "Unknown tool" if called through `execute_tool()`.
- `pc_run` is the exception — it has its own policy+approval path via `rex/computers/pc_run_policy.py`, bypassing `tool_router.py` entirely.
- The LLM uses `TOOL_REQUEST: {...}` single-line format to invoke tools; multi-line requests are rejected by `parse_tool_request`.
- `skip_policy_check=True` also forces `skip_credential_check=True` — these two flags are coupled in `execute_tool()`.
- US-P4-002 (tool routing bridge) should expose all 3 implemented tools (`time_now`, `weather_now`, `web_search`) through the OpenClaw bridge. The 8 unimplemented tools should be documented as "stub registered in policy, not yet implemented".
- `_CITY_TIMEZONES` dict (~200 entries, lines 545–900+) is an internal lookup table for `_resolve_timezone()`. It is private and not part of the migration surface.

---

### Audit Notes: Tool Classification (US-P4-002)

**Classification schema:**
- **Rex-specific** — tool is tightly coupled to Rex's unique functionality (HA, PC agent, WooCommerce). Must remain as a Rex-owned skill/tool registered with OpenClaw; cannot be replaced generically.
- **Generic (replace)** — tool does something any agent framework can do natively. Should be replaced by OpenClaw's equivalent capability; Rex implementation retired after migration.
- **Adapter-needed** — tool wraps a Rex service that stays in Rex (email, calendar) but needs a thin bridge to route calls through OpenClaw's tool dispatch. Rex service code stays; only the routing layer changes.

**Tool classification table:**

| Tool | In tool_router? | Risk | Classification | Notes |
|------|-----------------|------|----------------|-------|
| `time_now` | Yes (implemented) | LOW | Generic (replace) | Generic time/date/timezone query. OpenClaw or any agent framework can provide this. Currently in `_execute_time_now` using `ZoneInfo`. Replace with OpenClaw tool in Phase 4. |
| `weather_now` | Yes (implemented) | LOW | Generic (replace) | Generic weather query via `OPENWEATHERMAP_API_KEY`. OpenClaw can provide a weather tool. Currently in `_execute_weather_now`. Replace with OpenClaw tool in Phase 4. |
| `web_search` | Yes (implemented) | LOW | Generic (replace) | Generic web search via Rex plugin (`plugins.web_search.search_web`). OpenClaw can provide search. Rex's search-provider selection (brave/serpapi/ddg) is a config concern, not a Rex-unique feature. Replace with OpenClaw tool in Phase 4. |
| `send_email` | No (stub in policy) | MEDIUM | Adapter-needed | Rex has full IMAP/SMTP backend (`rex/email_backends/`, `rex/email_service.py`). Policy-gated but `tool_router.execute_tool()` returns "Unknown tool" — caller must route directly to `email_service`. Bridge must call `rex.email_service` and enforce Rex policy (MEDIUM, requires approval). |
| `home_assistant_call_service` | No (stub in policy) | MEDIUM | Rex-specific | HA bridge (`rex/ha_bridge.py`) is uniquely Rex. Policy-gated (MEDIUM, approval) but no handler in tool_router. Will be registered as an HA OpenClaw skill in Phase 5. Do not genericise. |
| `calendar_create_event` | No (stub in policy) | MEDIUM | Adapter-needed | Rex has `rex/calendar_backends/` + `rex/calendar_service.py`. Policy-gated (MEDIUM, approval) but no handler in tool_router. Bridge must delegate to `rex.calendar_service.create_event()`. |
| `calendar_delete_event` | No (stub in policy) | MEDIUM | Adapter-needed | Same as `calendar_create_event`. Delegate to `rex.calendar_service.delete_event()`. |
| `execute_command` | No (stub in policy) | HIGH | Generic (replace) | Generic shell command execution. OpenClaw's workspace/agent model handles this. Rex's `computers/` agent server is the Rex-specific execution target, but the tool itself is generic. Replace with OpenClaw workspace command tool in Phase 4d. |
| `pc_run` | No (own path) | HIGH | Rex-specific | Windows-specific remote execution via `rex/computers/` (agent server + client). Has its own approval path in `pc_run_policy.py`, bypassing `tool_router` entirely. Will become a Rex skill over OpenClaw's workspace model in Phase 4d/5. Do not genericise. |
| `file_write` | No (stub in policy) | HIGH | Generic (replace) | Generic file write. OpenClaw workspace tools cover this. Replace with OpenClaw's file-write capability in Phase 4d. |
| `file_delete` | No (stub in policy) | HIGH | Generic (replace) | Generic file delete. OpenClaw workspace tools cover this. Replace with OpenClaw's file-delete capability in Phase 4d. |

**Classification summary:**

| Classification | Count | Tools |
|----------------|-------|-------|
| Generic (replace) | 6 | `time_now`, `weather_now`, `web_search`, `execute_command`, `file_write`, `file_delete` |
| Adapter-needed | 3 | `send_email`, `calendar_create_event`, `calendar_delete_event` |
| Rex-specific | 2 | `home_assistant_call_service`, `pc_run` |

**Migration sequencing implications:**
- Phase 4a (tool bridge): start with the 3 *implemented* generics (`time_now`, `weather_now`, `web_search`). These are ready now — no new service code needed.
- Phase 4a extension: add adapter stubs for the 3 adapter-needed tools (`send_email`, `calendar_*`) — each delegates to the existing Rex service and enforces Rex policy.
- Phase 4d (workspace): tackle the 3 generic unimplemented tools (`execute_command`, `file_write`, `file_delete`) once OpenClaw workspace model is confirmed.
- Phase 5: register `home_assistant_call_service` as HA skill and `pc_run` as a Rex skill via OpenClaw workspace model.

---

### Audit Notes: rex/event_bus.py (US-P4-013)

**Module overview:**
- 436 lines. Dual API: legacy `publish(str, dict)` / `subscribe(str, callback(str, dict))` and rich `publish(Event)` / `subscribe(str, handler(Event))`.
- Thread-safe; wildcard subscriptions via `"*"`.
- `EventQueue` wraps `EventBus` with a bounded queue and daemon worker thread.
- Global singleton: `get_event_bus()` / `set_event_bus()`.

**Published event types (complete list):**

| Event Type | Publisher | Payload Keys | Notes |
|------------|-----------|--------------|-------|
| `email.unread` | `rex/integrations.py` (scheduler job) | `count`, `emails` | Published every 10 min by email check job |
| `email.unread` | `rex/integrations/_setup.py` (scheduler job) | `count`, `emails` | Duplicate of above (parallel impl) |
| `email.unread` | `rex/email_service.py:fetch_unread()` | varies | Internal publish on fetch |
| `email.triaged` | `rex/email_service.py` | varies | Published after triage operation |
| `email.read` | `rex/email_service.py:mark_read()` | `id` | Published when email marked read |
| `calendar.update` | `rex/integrations.py` (scheduler job) | `count`, `events` | Published every 1 hour by calendar sync job |
| `calendar.update` | `rex/integrations/_setup.py` (scheduler job) | `count`, `events` | Duplicate of above (parallel impl) |
| `calendar.connected` | `rex/calendar_service.py:connect()` | `connected`, `count`/`error` | Published on connect/disconnect |
| `calendar.upcoming` | `rex/calendar_service.py:list_upcoming()` | `count`, `events` | Published on list_upcoming call |
| `calendar.range` | `rex/calendar_service.py:get_events()` | `count`, `start`, `end` | Published on time-range query |
| `calendar.created` | `rex/calendar_service.py:create_event()` | event summary | Published after event creation |
| `calendar.updated` | `rex/calendar_service.py:update_event()` | `event`/`event_id` | Published after event update |
| `calendar.deleted` | `rex/calendar_service.py:delete_event()` | `event_id`, `deleted` | Published after event deletion |

**Subscribers (complete list):**

| Subscriber | Event Type | Handler | Location |
|------------|------------|---------|----------|
| `log_email_event` | `email.unread` | Logs count to info | `rex/integrations.py:setup_default_event_handlers()` |
| `log_calendar_event` | `calendar.update` | Logs count to info | `rex/integrations.py:setup_default_event_handlers()` |
| `log_email_event` | `email.unread` | Logs count to info | `rex/integrations/_setup.py:setup_default_event_handlers()` |
| `log_calendar_event` | `calendar.update` | Logs count to info | `rex/integrations/_setup.py:setup_default_event_handlers()` |
| `NotificationSystem._on_email_unread` | `email.unread` | Triggers notification | `rex/notification.py` (subscribed in start()) |
| `NotificationSystem._on_calendar_update` | `calendar.update` | Triggers notification | `rex/notification.py` (subscribed in start()) |
| `EventTriggerRegistry._bus_handler` | `*` (wildcard) | Dispatches to trigger fns | `rex/event_triggers.py:attach()` |

**Key findings for US-P4-014 classification:**

- `email.unread` — Rex-specific business event. Notifications/integrations consume it. Keep as Rex-specific.
- `email.triaged`, `email.read` — Rex-specific email workflow events. Keep.
- `calendar.*` events — Rex-specific calendar workflow events. Keep.
- All published events are Rex domain events, not framework-level infrastructure events.
- The event bus *mechanism* (pub-sub infrastructure) is framework-level → **Replace** with OpenClaw.
- The event *types* (email/calendar domain semantics) are Rex-specific → events bridge through OpenClaw's event system.
- **No framework-level infrastructure events found** (no workflow.started, tool.executed, session.created etc.).
- `EventTriggerRegistry` is the primary dynamic routing consumer (wildcard subscribe to all events).

**Duplicate integrations:**
- `rex/integrations.py` and `rex/integrations/_setup.py` appear to be parallel implementations of the same setup logic. Both are present in the codebase. The `_setup.py` version is likely newer. Both publish the same event types.

---

### Audit Notes: rex/event_bus.py event type classification (US-P4-014)

**Classification schema:**
- **Framework-level** — Infrastructure events that OpenClaw natively understands (e.g., workflow.started, task.completed). No bridge code needed; OpenClaw handles them.
- **Rex-specific** — Domain events unique to Rex with no OpenClaw equivalent. Preserved in bridge; OpenClaw passes them through as opaque events.
- **Bridge-needed** — Events that cross the OpenClaw/Rex boundary and require explicit mapping or translation in the event bridge.

**Event type classification table:**

| Event Type | Classification | Reasoning | Bridge Action |
|------------|----------------|-----------|---------------|
| `email.unread` | Rex-specific | Rex email domain event; no OpenClaw equivalent | Bridge passes through opaque; EventBridge republishes on OpenClaw bus |
| `email.triaged` | Rex-specific | Rex email triage workflow; no OpenClaw equivalent | Bridge passes through opaque |
| `email.read` | Rex-specific | Rex email state event; no OpenClaw equivalent | Bridge passes through opaque |
| `calendar.connected` | Rex-specific | Rex calendar connection state; no OpenClaw equivalent | Bridge passes through opaque |
| `calendar.upcoming` | Rex-specific | Rex calendar query result event; no OpenClaw equivalent | Bridge passes through opaque |
| `calendar.range` | Rex-specific | Rex calendar range query result; no OpenClaw equivalent | Bridge passes through opaque |
| `calendar.created` | Rex-specific | Rex calendar mutation event; no OpenClaw equivalent | Bridge passes through opaque |
| `calendar.updated` | Rex-specific | Rex calendar mutation event; no OpenClaw equivalent | Bridge passes through opaque |
| `calendar.deleted` | Rex-specific | Rex calendar mutation event; no OpenClaw equivalent | Bridge passes through opaque |
| `calendar.update` | Rex-specific | Rex calendar sync result event; no OpenClaw equivalent | Bridge passes through opaque |
| EventBus mechanism | Framework-level | Pub-sub infrastructure → Replace with OpenClaw event system | Replace (no bridge; migrate publishers/subscribers directly) |

**Summary:**
- **Framework-level**: 0 event *types* found (no workflow/task/session infrastructure events exist in Rex's current event bus)
- **Rex-specific**: 10 event types (all `email.*` and `calendar.*`)
- **Bridge-needed**: 10 (all Rex-specific events must flow through EventBridge during migration period)
- **EventBus mechanism** is framework-level and will be replaced; the event *types* it carries are Rex-specific

**Wildcard subscriber note:**
`EventTriggerRegistry._bus_handler` subscribes to `"*"` (all events). The EventBridge must forward all Rex-specific events to maintain this wildcard coverage. If OpenClaw's event system does not support wildcard subscriptions natively, the bridge must implement fan-out.

---

### Audit Notes: rex/browser_automation.py (US-P4-019 through US-P4-021)

**Module overview:**
- ~580 lines. Playwright-based browser automation. Optional dependency (gracefully absent if `playwright` not installed).
- Core classes: `BrowserSession` (async session API), `BrowserAutomationService` (session manager).
- Module-level helpers: `run_browser_script(steps)` (async), `run_browser_script_sync(steps)` (sync wrapper).
- Contract already defined at `rex/contracts/browser.py` (`BrowserSessionProtocol`, `BrowserAutomationProtocol`).

**US-P4-019: Importer list (complete):**

| File | Import | Usage pattern |
|------|--------|---------------|
| `rex/cli.py:50` | `get_browser_service` | Lazy wrapper; used in `sessions` and `screenshots` subcommands |
| `rex/cli.py:1665` | `run_browser_script_sync` | Lazy import in `cmd_browser`; runs JSON step scripts |

Only 2 production callers. No other rex/* modules import browser_automation directly.

**US-P4-020: Browser usage pattern classification:**

| Usage pattern | Functions/Methods | Caller | Notes |
|--------------|------------------|--------|-------|
| **Session management** | `BrowserAutomationService.list_sessions()`, `list_screenshots()` | `rex/cli.py` (`sessions`, `screenshots` subcmds) | Lists existing sessions/screenshots |
| **Script orchestration** | `run_browser_script_sync(steps)` | `rex/cli.py` (`run` subcmd) | Runs a JSON list of steps |
| **Navigation** | `BrowserSession.navigate(url)` | Used inside scripts | Direct URL navigation |
| **Interaction** | `BrowserSession.click()`, `type_text()`, `wait()`, `wait_for_selector()` | Used inside scripts | Page interaction |
| **Screenshot** | `BrowserSession.screenshot()` | Used inside scripts | Capture page state |
| **Login** | `BrowserSession.login()` | Used inside scripts | Credential-integrated login |
| **Content extraction** | `BrowserSession.get_content()`, `get_text()` | Used inside scripts | Scrape page content |

**US-P4-021: OpenClaw browser coverage mapping:**

| Rex function/pattern | OpenClaw equivalent | Coverage | Gap notes |
|---------------------|---------------------|----------|-----------|
| `BrowserSession.navigate()` | OpenClaw browser control (TBD) | Unknown — stub | Confirm once OpenClaw browser API is known |
| `BrowserSession.click()`, `type_text()` | OpenClaw browser control (TBD) | Unknown — stub | |
| `BrowserSession.screenshot()` | OpenClaw browser control (TBD) | Unknown — stub | |
| `BrowserSession.login()` | No direct equivalent | Gap | Rex login helper integrates Rex credential manager; OpenClaw would need a credential bridge |
| `run_browser_script(steps)` | No direct equivalent | Gap | Rex-specific JSON script format; bridge would need to translate |
| `BrowserAutomationService.list_sessions()` | No direct equivalent | Keep | Rex session management is Rex-specific |
| `get_browser_service()` → singleton | No direct equivalent | Wrap | Singleton lifecycle is Rex-specific |

**Key findings:**
- Only 2 callers in production — browser migration is low blast-radius.
- Login helper is the highest-risk gap: it reaches into Rex credential manager. OpenClaw browser bridge needs a credential adapter.
- `run_browser_script` format is Rex-specific; it will need translation or wrapper.
- Session lifecycle management (`BrowserAutomationService`) is Rex-specific and should stay.
- Browser contract (`rex/contracts/browser.py`) is already defined — bridge implementation (US-P4-022) can proceed directly.

---

### Browser migration backlog (US-P4-027)

**Completed during Phase 4:**
- `BrowserBridge` created (`rex/openclaw/browser_bridge.py`) — implements `BrowserAutomationProtocol`
- `rex/cli.py:get_browser_service()` updated to return `BrowserBridge()` instead of raw service
- `rex/cli.py:cmd_browser run` updated to call `bridge.execute_script()` instead of `run_browser_script_sync()`
- 19 tests cover bridge delegation and auth flow

**Remaining backlog (could not migrate in Phase 4):**

| Item | Reason not migrated | Priority | Notes |
|------|---------------------|----------|-------|
| `BrowserSession.login()` credential integration | Depends on Rex credential manager (`rex.credentials`). OpenClaw has no credential bridge yet. | High | Requires credential bridge (future phase) before OpenClaw can own the login flow |
| `run_browser_script` step format translation | Rex JSON step format (`navigate`, `click`, `type`, `login`, `screenshot`, `wait`, `download`) is Rex-specific. OpenClaw browser API shape is unknown. | Medium | Map once OpenClaw browser control API is confirmed (PRD §8.5) |
| `BrowserAutomationService` session storage | `data/browser_sessions/` directory management is Rex-specific. OpenClaw may have its own session storage. | Low | Evaluate at retirement time |
| `reset_browser_service()` | Module-level singleton reset utility. Not bridged. Used in tests only. | Low | Not needed during migration; delete at retirement |

---

### Audit Notes: rex/workflow.py and rex/workflow_runner.py (US-P4-028)

**Module overview:**
- `rex/workflow.py` (668 lines): Data models — `Workflow`, `WorkflowStep`, `WorkflowApproval`, `StepResult`. Pydantic v2. Persists to `data/workflows/{id}.json`. OPENCLAW-WRAP.
- `rex/workflow_runner.py` (864 lines): Execution engine — `WorkflowRunner`, `approve_workflow`, `deny_workflow`, `list_pending_approvals`. Policy-gated, approval-aware, idempotent, dry-run capable. OPENCLAW-WRAP.

**Callers of workflow.py public API (production code only):**

| File | What it imports/uses | Usage context |
|------|---------------------|---------------|
| `rex/cli.py` | `Workflow`, `WorkflowRunner`, `approve_workflow`, `deny_workflow`, `list_pending_approvals`, `WorkflowApproval` | `cmd_run_workflow` (runs workflow from JSON), approval subcommands |
| `rex/executor.py` | `Workflow`, `WorkflowStep`, `StepResult`, `WorkflowRunner`, `RunResult` | `Executor` wraps `WorkflowRunner`; used by autonomy system |
| `rex/autonomy_modes.py` | `Workflow` | Creates workflow objects for autonomy tasks |
| `rex/planner.py` | `Workflow`, `WorkflowStep`, `generate_step_id`, `generate_workflow_id` | Creates workflows from plans |
| `rex/computers/pc_run_policy.py` | `WorkflowApproval`, `DEFAULT_APPROVAL_DIR`, `generate_approval_id` | PC agent approval creation |
| `rex/openclaw/approval_adapter.py` | `WorkflowApproval`, `DEFAULT_APPROVAL_DIR`, `generate_approval_id`, `approve_workflow`, `deny_workflow`, `list_pending_approvals` | Already bridged (Phase 3, US-P3) |
| `rex/woocommerce/write_policy.py` | `WorkflowApproval`, `DEFAULT_APPROVAL_DIR`, `generate_approval_id` | WooCommerce write operation approvals |

---

### Audit Notes: Rex workflow → OpenClaw model mapping (US-P4-029)

**Rex concept → OpenClaw equivalent:**

| Rex concept | Rex class/field | OpenClaw equivalent | Notes |
|-------------|----------------|---------------------|-------|
| Workflow | `Workflow` (Pydantic model) | OpenClaw Task/Plan (TBD) | Workflow = ordered sequence of tool calls with policy gating |
| Workflow step | `WorkflowStep` | OpenClaw Task step / skill invocation (TBD) | Each step wraps a `ToolCall` (tool name + args) |
| Step result | `StepResult` | OpenClaw step result (TBD) | `success`, `output`, `error`, `skipped`, `skip_reason` |
| Precondition | `WorkflowStep.precondition` (str fn name) | No direct equivalent | Rex-specific: callable resolved from registry. Bridge must preserve. |
| Postcondition | `WorkflowStep.postcondition` (str fn name) | No direct equivalent | Rex-specific: result validator. Bridge must preserve. |
| Idempotency key | `WorkflowStep.idempotency_key` | No direct equivalent | Rex-specific: skip already-executed steps. Bridge must preserve. |
| Approval gate | `WorkflowStep.requires_approval`, `WorkflowApproval` | OpenClaw approval mechanism (TBD) | Already bridged via `ApprovalAdapter` (US-P3). |
| Policy check | `WorkflowRunner` consults `PolicyEngine` before each step | OpenClaw policy (TBD) | Rex `PolicyEngine` is authority; bridge must not bypass. |
| Persistence | `data/workflows/{id}.json` | OpenClaw state store (TBD) | Rex-specific disk persistence. Bridge should preserve during migration. |
| Dry run | `WorkflowRunner(dry_run=True)` | No direct equivalent | Rex-specific: previews actions without executing. Preserve. |
| Tool execution | `WorkflowRunner` calls tool_router per step | Via ToolBridge (bridged in US-P4) | ToolBridge already routes tool calls. |

**Key design decisions for US-P4-030 WorkflowBridge:**
- `WorkflowRunner` must be wrapped, not replaced — policy hooks, idempotency, preconditions, and persistence are all Rex-specific
- Bridge receives a `Workflow` object and delegates execution to `WorkflowRunner`
- `WorkflowRunner` must continue using Rex `PolicyEngine` as authority
- Approval flow already bridges via `ApprovalAdapter` — WorkflowBridge reuses it
- Dry-run mode must be preserved through the bridge
- Persistence path (`data/workflows/`) stays Rex-owned during migration

---

### Audit Notes: rex/ha_bridge.py (US-P5-001)

**Public API:**

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `IntentMatch` | dataclass: `domain`, `service`, `entity_id`, `data`, `description`, `source` | Passed to all intent execution paths |
| `HABridge` | class | Main class; all public methods below |
| `HABridge.__init__` | `(*, base_url, token, secret, verify_ssl, timeout, entity_map)` — all optional | Falls back to `settings` global |
| `HABridge.enabled` | property → `bool` | True iff base_url and token configured |
| `HABridge.secret` | property → `str` | Shared secret for blueprint auth |
| `HABridge.process_transcript` | `(transcript: str) → str \| None` | Intent detection + execution from voice input |
| `HABridge.post_process_response` | `(response: str) → str` | Executes inline `[[ha:...]]` tags in LLM replies |
| `HABridge.list_entities` | `() → list[dict]` | Returns entity cache (refreshed) |
| `HABridge.control_light` | `(entity_id, action, *, brightness_pct=None) → dict` | `turn_on`/`turn_off` with optional brightness |
| `HABridge.control_switch` | `(entity_id, action) → dict` | `turn_on`/`turn_off` switch |
| `HABridge.call_script` | `(script_id, variables=None) → dict` | Call an HA script via REST |
| `create_blueprint` | `(bridge=None) → Blueprint` | Flask blueprint factory for /ha/* routes |
| `HABridge.SUPPORTED_INTENTS` | class attribute `list[dict]` | 6 intents: turn_on, turn_off, set_temperature, set_percentage, activate, lock_control |

**Callers:**

| File | What it uses |
|------|-------------|
| `rex/assistant.py` | `HABridge()`, `process_transcript`, `post_process_response`, `enabled` |
| `rex/openclaw/tools/ha_tool.py` | `HABridge()`, `enabled`, `_execute_intent` (private!), `IntentMatch` |
| `rex_speak_api.py` | `create_blueprint` (Flask route registration) |

**Key finding — private method call in ha_tool.py:**
`ha_tool.py` constructs an `IntentMatch` and calls `bridge._execute_intent(intent)` (a private method).
This bypasses the public `control_light`, `control_switch`, and `call_script` methods.
For Phase 5 this is acceptable — the tool existed before the audit — but the gap should be noted:
- Future: either promote `_execute_intent` to public or refactor `ha_tool.py` to use public methods.

**OpenClaw migration notes:**
- `ha_tool.py` already exists and wraps HABridge as `home_assistant_call_service` — US-P5-003 may need only verification/test.
- `process_transcript` and `post_process_response` are called in the voice path (`assistant.py`). These hooks must survive the voice loop migration in Phase 6.
- `create_blueprint` is only used by `rex_speak_api.py`. Unaffected by OpenClaw migration until the HTTP layer is addressed.
- HABridge is stateful (entity cache, threading lock, requests.Session). OpenClaw tool registration gets a new instance per call via `_get_ha_bridge()` — safe for stateless tool calls, entity cache is rebuilt per invocation.

---

### Audit Notes: rex/ha_tts/* (US-P5-002)

**Public API (rex/ha_tts/client.py):**

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `TtsResult` | dataclass: `ok: bool`, `error: str \| None` | Return type for speak() |
| `HaTtsClient` | class | One public method: `speak()` |
| `HaTtsClient.__init__` | `(base_url, token, *, default_entity_id, tts_domain, tts_service, timeout, allow_http)` | SSRF-validated at construction |
| `HaTtsClient.speak` | `(message, *, entity_id=None, extra_data=None) → TtsResult` | Calls HA REST TTS service |
| `build_ha_tts_client` | `() → HaTtsClient \| None` | Factory: loads config + resolves token via CredentialManager; returns None if disabled/misconfigured |

**Public API (rex/ha_tts/config.py):**

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `HaTtsConfig` | Pydantic model (strict, extra=forbid) | Fields: enabled, base_url, token_ref, default_entity_id, default_tts_domain, default_tts_service, timeout_seconds, allow_http |
| `load_ha_tts_config` | `() → HaTtsConfig` | Reads `notifications.ha_tts` from rex_config.json |

**Callers:**

| File | What it uses |
|------|-------------|
| `rex/notification.py` | `build_ha_tts_client`, `HaTtsClient.speak` (via `_send_to_ha_tts`) |
| `rex/cli.py` | `load_ha_tts_config`, `build_ha_tts_client` (for `rex ha tts test` command) |

**OpenClaw migration notes:**
- HA TTS is a notification *channel*, not a tool — it's triggered by notification routing in `notification.py`, not by LLM tool calls.
- `speak()` is the only callable that matters for OpenClaw integration. If OpenClaw gains a TTS notification channel, `build_ha_tts_client().speak(message)` becomes the implementation.
- US-P5-006 will test this path. No new OpenClaw-specific code is needed for the audit; `build_ha_tts_client` and `speak()` are already the minimal surface.
- Security: SSRF validation is enforced at `HaTtsClient` construction. Token is resolved via `CredentialManager`, never stored in config. This pattern must be preserved if a TTS bridge is added.

---

### Audit Notes: rex/wordpress/* (US-P5-008)

**Public API:**

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `WPHealthResult` | dataclass: `ok`, `reachable`, `wp_detected`, `auth_ok`, `site_name`, `site_url`, `error` | Return type for health checks |
| `WordPressClient` | class | HTTP client; one public method: `health()` |
| `WordPressClient.__init__` | `(base_url, *, auth, timeout, site_id)` | SSRF-validated at construction |
| `WordPressClient.health` | `() → WPHealthResult` | GET /wp-json + optional GET /wp-json/wp/v2/users/me |
| `WordPressService` | class | Facade over config + credentials + client |
| `WordPressService.__init__` | `(wp_config, credential_manager)` | Accepts explicit deps for testing |
| `WordPressService.health` | `(site_id: str) → WPHealthResult` | Resolves site config + credentials, delegates to client |
| `get_wordpress_service` | `() → WordPressService` | Module-level singleton factory |
| `WordPressSiteConfig` | Pydantic model | `id`, `base_url`, `enabled`, `auth_method`, `credential_ref`, `timeout_seconds` |
| `WordPressConfig` | Pydantic model | `sites: list[WordPressSiteConfig]`, `get_site()`, `list_enabled()`, `list_all()` |
| `load_wordpress_config` | in `config.py` | Parses `wordpress.sites[]` from rex_config.json |
| `WordPressSiteNotFoundError` | exception | Unknown site_id |
| `WordPressSiteDisabledError` | exception | Site is disabled |
| `WordPressMissingCredentialError` | exception | Credential not configured |

**Callers:**

| File | What it uses |
|------|-------------|
| `rex/cli.py` | `get_wordpress_service()`, `service.health(site_id)` (for `rex wp health` command) |

**Key finding — read-only:**
The current WordPress implementation is strictly read-only. `WordPressClient.health()` is the only public method. There are no create/update/delete operations. US-P5-011 ("Test WordPress write through OpenClaw") is N/A for the current implementation.

**OpenClaw migration notes:**
- `wordpress_health_check` tool created in `rex/openclaw/tools/wordpress_tool.py` wraps `WordPressService.health()`.
- `ToolBridge.register_wordpress_tools()` registers the tool batch.
- Write operations are out of scope until the WordPress client is extended.
- Security: Basic Auth credentials resolved via `CredentialManager`. SSRF-validated `base_url`. Never stored in config.

---

### Audit Notes: rex/woocommerce/* (US-P5-012)

**Public API:**

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `OrdersResult` | dataclass: `ok`, `orders`, `error` | Read result |
| `ProductsResult` | dataclass: `ok`, `products`, `error` | Read result |
| `WriteResult` | dataclass: `ok`, `data`, `error` | Write result |
| `WooCommerceClient` | class | HTTP client |
| `WooCommerceClient.list_orders` | `(*, status, limit) → OrdersResult` | GET /wc/v3/orders |
| `WooCommerceClient.list_products` | `(*, limit, low_stock) → ProductsResult` | GET /wc/v3/products |
| `WooCommerceClient.set_order_status` | `(order_id, *, status) → WriteResult` | PUT /wc/v3/orders/<id> |
| `WooCommerceClient.add_order_note` | `(order_id, *, note, customer_note) → WriteResult` | POST /wc/v3/orders/<id>/notes |
| `WooCommerceClient.create_coupon` | `(*, code, amount, discount_type, ...) → WriteResult` | POST /wc/v3/coupons |
| `WooCommerceClient.disable_coupon` | `(coupon_id) → WriteResult` | PUT /wc/v3/coupons/<id> |
| `WooCommerceService` | class | Facade over config + credentials + client |
| `WooCommerceService.list_orders` | `(site_id, *, status, limit) → OrdersResult` | |
| `WooCommerceService.list_products` | `(site_id, *, limit, low_stock) → ProductsResult` | |
| `WooCommerceService.set_order_status` | `(site_id, order_id, *, status) → WriteResult` | HIGH risk |
| `WooCommerceService.add_order_note` | `(site_id, order_id, *, note, customer_note) → WriteResult` | HIGH risk |
| `WooCommerceService.create_coupon` | `(site_id, *, code, amount, discount_type, ...) → WriteResult` | HIGH risk |
| `WooCommerceService.disable_coupon` | `(site_id, coupon_id) → WriteResult` | HIGH risk |
| `get_woocommerce_service` | `() → WooCommerceService` | Module-level singleton |
| `check_wc_write_policy` | `(action, site_id, identifiers, params, *, step_description, ...) → (PolicyDecision, WorkflowApproval \| None)` | Policy + approval gate for write ops |
| `find_pending_or_approved_wc_approval` | `(action, site_id, identifiers, approval_dir) → WorkflowApproval \| None` | Scans approval dir for existing record |
| `WC_WRITE_WORKFLOW_ID`, `WC_ORDER_SET_STATUS_TOOL`, etc. | constants | Tool name constants |

**Callers:**

| File | What it uses |
|------|-------------|
| `rex/cli.py` | `get_woocommerce_service()`, read methods, `check_wc_write_policy` for write ops |

**OpenClaw migration notes:**
- 5 tools created in `rex/openclaw/tools/woocommerce_tool.py`:
  read (wc_list_orders, wc_list_products) + write (wc_set_order_status, wc_create_coupon, wc_disable_coupon)
- Write tools call `check_wc_write_policy` before any network call — Rex write policy is the authority
- `add_order_note` not exposed as a tool yet — can be added when needed
- `ToolBridge.register_woocommerce_tools()` registers the full batch
- Security: Consumer key/secret via `CredentialManager`. SSRF-validated URLs. Policy gate before writes.
