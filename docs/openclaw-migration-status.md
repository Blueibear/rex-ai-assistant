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
| `rex/browser_automation.py` | Replace | Pending | Generic Playwright wrapper. OpenClaw has browser control. Create browser bridge; migrate callers one at a time. |
| `rex/dashboard/__init__.py` | Replace | Pending | Flask dashboard. OpenClaw provides dashboard/UI. Run both in parallel during transition. |
| `rex/dashboard/routes.py` | Replace | Pending | Dashboard routes. Retires with dashboard. |
| `rex/dashboard/sse.py` | Replace | Pending | SSE streaming. Retires with dashboard. |
| `rex/dashboard/auth.py` | Replace | Pending | Dashboard auth. Retires with dashboard. |
| `rex/dashboard_store.py` | Replace | Pending | Dashboard persistence. Retires when dashboard retires. |
| `rex/messaging_backends/` | Replace | Pending | Twilio, SMS, webhooks (11 files). OpenClaw owns channels. Migrate last in Phase 4 due to webhook complexity. |
| `rex/messaging_service.py` | Replace | Pending | Messaging orchestration. Retires with messaging_backends. |
| `rex/integrations/message_router.py` | Replace | Pending | Routes messages between channels. Retires with messaging. |
| `rex/tool_registry.py` | Replace | Pending | Tool metadata + health checks. OpenClaw has skill/tool system. Define Protocol first (Phase 1), then bridge (Phase 4). |
| `rex/tool_router.py` | Replace | Pending | Central tool dispatch (960 lines). Highest-risk replacement. Feature flag required. Test every tool through bridge before retirement. |
| `rex/plugin_loader.py` | Replace | Pending | Dynamic plugin discovery (56 lines). OpenClaw has plugins. Small file, easy replacement. |
| `rex/executor.py` | Replace | Pending | Task execution engine. Replaced by OpenClaw task execution via workflow bridge. |
| `rex/event_bus.py` | Replace | Pending | Pub-sub event system (436 lines). Dual API (simple + rich) must be preserved in bridge. |
| `rex/computers/` | Replace | Pending | Windows agent server/client (~400 lines, 5 files). Replace with OpenClaw workspace/agent model. |
| `rex/workflow.py` | Wrap | Pending | Workflow data models (668 lines). Rex-specific definitions. Models translate to OpenClaw skill/task definitions; bridge translates at execution time. |
| `rex/workflow_runner.py` | Wrap | Pending | Workflow execution (864 lines). Has Rex policy hooks. Bridge preserves policy gating; Rex policy is authority. |
| `rex/autonomy/__init__.py` | Wrap | Pending | Autonomy package init. Wraps OpenClaw multi-agent primitives. |
| `rex/autonomy/runner.py` | Wrap | Pending | Autonomy runner. High-level planning logic wraps OpenClaw primitives. Keep Rex's goal decomposition and replanning. |
| `rex/autonomy/llm_planner.py` | Wrap | Pending | LLM-based planner. Preserved; wraps OpenClaw. |
| `rex/autonomy/rule_planner.py` | Wrap | Pending | Rule-based planner. Preserved; wraps OpenClaw. |
| `rex/scheduler.py` | Wrap | Pending | Cron-like scheduling (675 lines). Evaluate OpenClaw scheduling; wrap if available, keep if not. |
| `rex/planner.py` | Wrap | Pending | Task planning (640 lines). Rex's planning logic wraps OpenClaw primitives. |
| `rex/notification.py` | Wrap | Pending | Notification system (884 lines). Route through OpenClaw's notification/event system if available. |
| `rex/policy.py` | Keep + Wrap | Pending | Policy models (150 lines). Rex-specific risk classification. Keep models; wrap as OpenClaw middleware. Rex policy is always the authority. |
| `rex/policy_engine.py` | Keep + Wrap | Pending | Policy evaluation (350 lines). Keep engine; wrap as OpenClaw hook. |
| `rex/identity.py` | Wrap | Pending | User identity resolution (280 lines). Map to OpenClaw session/user model. Keep Rex's resolution logic. |
| `rex/profile_manager.py` | Wrap | Pending | Profile merging (100 lines). Keep merge logic; wire into OpenClaw agent config. |
| `rex/voice_identity/` | Keep | Pending | Speaker recognition (7 files). Uniquely Rex. No OpenClaw equivalent. Phase 6: feed into OpenClaw session identity. |
| `rex/wakeword/` | Keep | Pending | Wake word detection (4 files). Uniquely Rex. Unchanged in migration. |
| `rex/voice_loop.py` | Keep | Pending | Core voice loop (800 lines). Update to call OpenClaw backend via voice_bridge. Feature flag for rollback. |
| `rex/voice_loop_optimized.py` | Keep | Pending | Low-latency voice loop (550 lines). Same treatment as voice_loop.py. |
| `rex/ha_bridge.py` | Keep | Pending | Home Assistant bridge (600 lines). Register as OpenClaw skill in Phase 5. Code stays in Rex. |
| `rex/ha_tts/` | Keep | Pending | HA TTS integration (3 files). Register as OpenClaw skill in Phase 5. |
| `rex/wordpress/` | Keep | Pending | WordPress client (3 files). Register as OpenClaw skill in Phase 5. |
| `rex/woocommerce/` | Keep | Pending | WooCommerce client with write policy (4 files). Register as OpenClaw skill in Phase 5. Write policy preserved. |
| `rex/plex_client.py` | Keep | Pending | Plex media control. Register as OpenClaw skill in Phase 5. |
| `rex/memory.py` | Keep + Adapt | Audited (US-P3-001) | See audit notes below. |
| `rex/memory_utils.py` | Keep | Pending | Memory helpers. Re-exported via memory.py. Stays with memory.py. |
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
