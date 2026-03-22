# PRD: Pivot Rex AI Assistant to OpenClaw-Based Architecture

## 1. Executive Summary

Rex AI Assistant has grown from a voice-activated local AI companion into a custom agent framework with orchestration, browser automation, plugin loading, dashboard UI, messaging channels, workflow execution, and policy enforcement. Much of this infrastructure duplicates what OpenClaw already provides as a purpose-built agent engine.

This PRD defines a phased migration that replaces Rex's generic framework plumbing with OpenClaw while preserving Rex's unique value: persona, voice identity, wakeword/voice loop, Home Assistant integration, WordPress/WooCommerce/Plex integrations, business workflows, Nasteeshirts growth logic, memory conventions, and policy/approval behavior.

The migration follows a strict incremental approach: freeze duplication first, stand up a Rex-on-OpenClaw baseline, port unique components one at a time, and retire redundant subsystems only after replacements are proven. Every task is sized for a single Ralph Loop iteration.

**Codebase scale:** ~194 Python files across 20 subpackages in `rex/`, plus ~30 root-level support scripts. Key modules range from 380 lines (assistant.py) to 4,941 lines (cli.py).

---

## 2. Goals

- Stop rebuilding generic agent infrastructure inside Rex.
- Replace Rex's dashboard, browser automation, plugin/tool loading, messaging channels, agent-session plumbing, and event bus with OpenClaw equivalents.
- Preserve all Rex-specific value: persona, voice, wakeword, HA bridge, WordPress, WooCommerce, Plex, business workflows, Nasteeshirts logic.
- Wrap Rex's policy engine, identity, autonomy, and profile management around OpenClaw primitives rather than discarding them.
- Maintain a working Rex system at every migration step (no broken intermediate states).
- Produce a codebase where Rex is an opinionated application layer on top of OpenClaw, not a competing framework.
- Keep the migration executable by a Ralph Loop: tiny steps, explicit sequencing, verifiable checkpoints.

---

## 3. Non-Goals

- Big-bang rewrite of Rex.
- Migrating everything at once.
- Rewriting Rex integrations (HA, WordPress, WooCommerce, Plex) from scratch.
- Building new OpenClaw features that don't exist yet (the migration uses what OpenClaw provides today).
- Changing Rex's user-facing persona or voice behavior.
- Migrating the GUI (`gui.py`, `gui_settings_tab.py`, `run_gui.py`) in this phase. GUI migration is a separate future effort.
- Rearchitecting the LLM client layer (`rex/llm_client.py`). Rex's multi-provider LLM support is orthogonal to OpenClaw's agent infrastructure.
- Touching the test suite structure. Tests migrate alongside the modules they cover.

---

## 4. Current State

### 4.1 Four-Layer Model

Rex currently contains four interleaved layers:

**Layer 1: Generic Framework Plumbing**
Modules that replicate what an agent framework provides. These are candidates for replacement.

| Module | Lines | Purpose |
|--------|-------|---------|
| `rex/event_bus.py` | 436 | Pub-sub event system |
| `rex/tool_registry.py` | 458 | Tool metadata + health checks |
| `rex/tool_router.py` | 960 | Route tool calls to implementations |
| `rex/plugin_loader.py` | 56 | Dynamic plugin discovery |
| `rex/browser_automation.py` | 581 | Playwright-based browser control |
| `rex/dashboard/*` | ~600 | Flask dashboard + SSE + auth |
| `rex/dashboard_store.py` | ~400 | Dashboard persistence |
| `rex/messaging_backends/*` | ~800 | Twilio, SMS, webhooks |
| `rex/messaging_service.py` | 501 | Messaging orchestration |
| `rex/executor.py` | 413 | Task execution engine |

**Layer 2: Assistant Runtime / Orchestration**
Modules that wire the framework together into an assistant. These get wrapped or refactored.

| Module | Lines | Purpose |
|--------|-------|---------|
| `rex/assistant.py` | 380 | Main assistant orchestration |
| `rex/workflow.py` | 668 | Workflow data models |
| `rex/workflow_runner.py` | 864 | Workflow step execution |
| `rex/planner.py` | 640 | Task planning |
| `rex/autonomy/*` | ~1200 | LLM planner, goal graph, runner |
| `rex/autonomy_modes.py` | ~300 | Autonomy level configuration |
| `rex/scheduler.py` | 675 | Cron-like task scheduling |

**Layer 3: Unique Rex Functionality**
Modules that carry Rex's unique value. These are kept as-is or minimally adapted.

| Module | Lines | Purpose |
|--------|-------|---------|
| `rex/voice_identity/*` | ~500 | Speaker recognition |
| `rex/wakeword/*` | ~300 | Wake word detection |
| `rex/voice_loop.py` | ~800 | Core voice loop |
| `rex/voice_loop_optimized.py` | ~550 | Low-latency voice loop |
| `rex/ha_bridge.py` | ~600 | Home Assistant bridge |
| `rex/ha_tts/*` | ~300 | HA TTS integration |
| `rex/wordpress/*` | ~300 | WordPress client |
| `rex/woocommerce/*` | ~500 | WooCommerce client |
| `rex/plex_client.py` | ~250 | Plex media control |
| `rex/identity.py` | ~280 | User identity resolution |
| `rex/policy.py` | ~150 | Policy models |
| `rex/policy_engine.py` | ~350 | Policy evaluation |
| `rex/profile_manager.py` | ~100 | Profile merging |
| `rex/memory.py` | ~650 | Conversation memory |
| `rex/memory_utils.py` | ~350 | Memory helpers |

**Layer 4: Supporting Ops / Service Infrastructure**

| Module | Lines | Purpose |
|--------|-------|---------|
| `rex/cli.py` | 4941 | CLI interface |
| `rex/config.py` | ~600 | Pydantic settings |
| `rex/config_manager.py` | ~550 | Runtime config management |
| `rex/credentials.py` | ~450 | Credential management |
| `rex/app.py` | ~250 | Flask app factory |
| `rex/api_key_auth.py` | ~170 | API auth |
| `rex/computers/*` | ~400 | Windows agent server/client |
| `rex/email_backends/*` | ~600 | IMAP/SMTP email |
| `rex/email_service.py` | 662 | Email orchestration |
| `rex/calendar_backends/*` | ~500 | Calendar integrations |
| `rex/calendar_service.py` | ~700 | Calendar orchestration |

### 4.2 Key Dependencies Between Layers

- `assistant.py` imports from `llm_client`, `tool_router`, `memory`, `ha_bridge`, `calendar_service`, `followup_engine`
- `workflow_runner.py` depends on `policy_engine`, `tool_router`, `event_bus`
- `autonomy/runner.py` depends on `workflow_runner`, `policy_engine`, `tool_router`
- `browser_automation.py` depends on `policy_engine`, `credentials`, `audit`
- `dashboard/routes.py` depends on `dashboard_store`, `event_bus`, `config_manager`, `scheduler`
- `voice_loop.py` depends on `assistant.py` (via `generate_reply`)
- `tool_router.py` is the central dispatch hub, called by assistant, workflow runner, autonomy runner, and CLI

---

## 5. Target State Architecture

```
+-------------------------------------------------------------------+
|                         Rex Application Layer                      |
|  Persona | Memory | Policy | Voice | Wakeword | Business Logic    |
|  HA Bridge | WordPress | WooCommerce | Plex | Nasteeshirts        |
+-------------------------------------------------------------------+
|                      Rex-OpenClaw Adapter Layer                    |
|  Policy adapter | Identity adapter | Autonomy adapter             |
|  Memory conventions adapter | Workflow bridge                     |
+-------------------------------------------------------------------+
|                       OpenClaw Agent Engine                        |
|  Channels | Sessions | Workspaces | Browser | Dashboard/UI        |
|  Skills/Plugins | Auth/Pairing | Multi-agent orchestration        |
|  Event bus | Tool registry | Tool routing                         |
+-------------------------------------------------------------------+
```

### 5.1 What OpenClaw Owns

- Channel management (replaces `rex/messaging_backends/*`, `rex/messaging_service.py`)
- Session lifecycle (replaces agent-session plumbing in `rex/executor.py`, parts of `rex/assistant.py`)
- Agent workspaces (replaces `rex/computers/*` agent server)
- Browser control (replaces `rex/browser_automation.py`)
- Dashboard / control UI (replaces `rex/dashboard/*`, `rex/dashboard_store.py`)
- Skills / plugins (replaces `rex/plugin_loader.py`, `rex/tool_registry.py`)
- Tool routing (replaces `rex/tool_router.py`)
- Auth / pairing / ingress (replaces `rex/api_key_auth.py`, `rex/dashboard/auth.py`)
- Event bus (replaces `rex/event_bus.py`)
- Baseline multi-agent orchestration (replaces generic parts of `rex/autonomy/*`)

### 5.2 What Rex Owns

- Rex persona (system prompts, personality, conversation style)
- Memory conventions (`rex/memory.py`, `rex/memory_utils.py` -- adapted to OpenClaw storage)
- Policy and approval logic (`rex/policy.py`, `rex/policy_engine.py` -- wrapped around OpenClaw)
- Voice identity (`rex/voice_identity/*`)
- Wakeword and voice loop (`rex/wakeword/*`, `rex/voice_loop.py`, `rex/voice_loop_optimized.py`)
- Home Assistant (`rex/ha_bridge.py`, `rex/ha_tts/*`)
- WordPress (`rex/wordpress/*`)
- WooCommerce (`rex/woocommerce/*`)
- Plex (`rex/plex_client.py`)
- Business workflows and Nasteeshirts growth logic
- Orchestration behavior and specialist-agent strategy (high-level logic in `rex/autonomy/*`)

### 5.3 What Gets Wrapped

- `rex/autonomy/*` -- Rex's planning/goal/replanning logic wraps OpenClaw's multi-agent primitives
- `rex/policy.py` / `rex/policy_engine.py` -- Rex's risk classification and approval gates hook into OpenClaw's permission model
- `rex/identity.py` / `rex/profile_manager.py` -- Rex's user identity maps to OpenClaw session/user primitives
- `rex/workflow.py` / `rex/workflow_runner.py` -- Rex's workflow models become OpenClaw skill definitions with Rex policy hooks

---

## 6. Migration Principles

1. **No big bang.** Every step produces a working system.
2. **Prove before retire.** A Rex subsystem is only retired after its OpenClaw replacement passes the same validation.
3. **Adapter-first.** When replacing a Rex module with OpenClaw, create a thin adapter that preserves the existing internal API. Callers migrate to the new API later.
4. **One module at a time.** Never migrate two coupled subsystems simultaneously.
5. **Test parity.** Every migrated module must pass its existing tests via the adapter before the old code is removed.
6. **Freeze before port.** No new features added to subsystems marked for replacement.
7. **Voice loop last.** The voice pipeline is the most user-visible component. It migrates after everything it depends on is stable.
8. **Rollback by revert.** Each migration step is a single commit (or small commit group) that can be reverted cleanly.

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OpenClaw lacks a feature Rex currently uses | Migration blocks on that subsystem | Adapter pattern: keep Rex implementation behind an interface, swap later. Document gap in open questions. |
| Migration breaks voice loop | Primary user-facing feature broken | Voice loop migrates last (Phase 6). Integration test voice loop after every phase. |
| Policy/approval semantics differ between Rex and OpenClaw | Security regression | Wrap Rex policy engine around OpenClaw. Rex policy is the authority. OpenClaw is the executor. |
| Browser automation has Rex-specific login helpers | Scraping/form workflows break | Classify every browser usage. Migrate simple navigation first. Keep Rex login helpers as adapters until OpenClaw supports them. |
| CLI is 4,941 lines with deep coupling | CLI breaks during migration | CLI commands that depend on migrated modules get updated one command at a time. Never batch CLI changes. |
| Dashboard has custom SSE/auth that OpenClaw may handle differently | Dashboard outage during migration | Run both dashboards in parallel during transition. OpenClaw dashboard replaces Rex dashboard only after feature parity confirmed. |
| Messaging backends have Twilio webhook wiring | Inbound messages stop working | Migrate messaging last within Phase 4. Test webhook delivery before cutting over. |
| Test coverage gaps in some modules | Silent regressions | Run full test suite after every phase. Add integration smoke tests for migrated paths. |

---

## 8. Dependencies and Assumptions

### 8.1 OpenClaw Assumptions

- OpenClaw provides a Python-compatible agent registration mechanism.
- OpenClaw supports custom tool registration (Rex tools like `time_now`, `weather`, `send_email` can be registered).
- OpenClaw has a browser automation subsystem that supports navigation, clicking, typing, screenshots.
- OpenClaw has a plugin/skill system that can load Rex's existing integrations.
- OpenClaw provides session management with user identity support.
- OpenClaw has a dashboard/UI that can display agent status, conversations, and notifications.
- OpenClaw has an event system or equivalent pub-sub mechanism.
- OpenClaw supports policy hooks or middleware for gating tool execution.

### 8.2 Rex Assumptions

- The existing test suite (`tests/`) provides baseline coverage for regression detection.
- The `config/rex_config.json` + `.env` split is preserved.
- The Ralph Loop executor can run one user story per iteration with no memory of previous iterations.
- Git is the rollback mechanism: each story produces a revertible commit.

### 8.3 Open Dependencies

- [ ] Confirm OpenClaw's Python API surface for agent registration
- [ ] Confirm OpenClaw's tool registration mechanism and whether it supports Rex's `ToolMeta` model
- [ ] Confirm OpenClaw's browser automation API compatibility with Rex's `BrowserAction` model
- [ ] Confirm OpenClaw's session model and whether it supports Rex's user identity resolution
- [ ] Confirm OpenClaw's policy/permission hook points
- [ ] Confirm OpenClaw's event system semantics (sync vs async, at-least-once vs at-most-once)

---

## 9. Detailed Migration Plan

### Phase 1: Freeze Framework Duplication and Define Migration Boundaries

**Objective:** Stop adding features to subsystems marked for replacement. Establish clear interface boundaries.

**Duration estimate:** 5-8 Ralph Loop iterations

#### Phase 1 Tasks

1. **Create migration tracking document** (`docs/openclaw-migration-status.md`) with a table listing every Rex module, its classification (Replace/Keep/Wrap/Retire), and current migration status.

2. **Add `# OPENCLAW-REPLACE` markers** to every file classified as Replace. These are machine-searchable freeze markers. Files: `rex/event_bus.py`, `rex/tool_registry.py`, `rex/tool_router.py`, `rex/plugin_loader.py`, `rex/browser_automation.py`, `rex/dashboard/__init__.py`, `rex/dashboard/routes.py`, `rex/dashboard/sse.py`, `rex/dashboard/auth.py`, `rex/dashboard_store.py`, `rex/messaging_backends/__init__.py`, `rex/messaging_service.py`, `rex/executor.py`.

3. **Add `# OPENCLAW-WRAP` markers** to every file classified as Wrap. Files: `rex/autonomy/__init__.py`, `rex/autonomy/runner.py`, `rex/autonomy/llm_planner.py`, `rex/autonomy/rule_planner.py`, `rex/policy.py`, `rex/policy_engine.py`, `rex/identity.py`, `rex/profile_manager.py`, `rex/workflow.py`, `rex/workflow_runner.py`.

4. **Extract interface contracts for Replace modules.** For each Replace module, create a corresponding abstract base class or Protocol in `rex/contracts/` that captures the public API. Start with `rex/contracts/tool_routing.py` (Protocol for tool routing).

5. **Extract interface contract for event bus.** Create `rex/contracts/event_bus.py` with the EventBus Protocol matching the existing dual API (simple + rich).

6. **Extract interface contract for browser automation.** Create `rex/contracts/browser.py` with the BrowserAutomation Protocol.

7. **Extract interface contract for dashboard.** Create `rex/contracts/dashboard.py` with Dashboard Protocol covering routes, SSE, and auth.

8. **Extract interface contract for plugin/tool loading.** Create `rex/contracts/plugins.py` with PluginLoader and ToolRegistry Protocols.

#### Phase 1 Validation

- [ ] Every Replace module has `# OPENCLAW-REPLACE` header comment
- [ ] Every Wrap module has `# OPENCLAW-WRAP` header comment
- [ ] Protocol files exist in `rex/contracts/` for: tool routing, event bus, browser, dashboard, plugin loading
- [ ] Existing tests pass unchanged
- [ ] `docs/openclaw-migration-status.md` exists with complete module listing

---

### Phase 2: Stand Up a Dedicated Rex Agent/Workspace Inside OpenClaw

**Objective:** Get a minimal Rex agent running inside OpenClaw that can respond to a simple text prompt using Rex's LLM client and persona.

**Duration estimate:** 8-12 Ralph Loop iterations

#### Phase 2 Tasks

1. **Create `rex/openclaw/` subpackage** with `__init__.py`.

2. **Create `rex/openclaw/agent.py`** that registers a "Rex" agent with OpenClaw using OpenClaw's agent registration API. The agent should accept a text prompt and return a text response. Initially, it calls `rex.llm_client.LanguageModel.generate()` directly.

3. **Create `rex/openclaw/config.py`** that maps Rex's `Settings` (from `rex/config.py`) to whatever configuration OpenClaw expects for agent setup.

4. **Create `rex/openclaw/session.py`** that bridges OpenClaw's session model to Rex's user identity (from `rex/identity.py`).

5. **Wire Rex persona into the OpenClaw agent.** The system prompt, personality settings, and conversation style from Rex's config must be injected into the OpenClaw agent's context.

6. **Register Rex's `time_now` tool** with OpenClaw's tool system as a proof-of-concept. Verify the agent can call it.

7. **Register Rex's `weather` tool** with OpenClaw. Verify the agent can call it.

8. **Create a smoke test** (`tests/test_openclaw_agent_basic.py`) that starts the OpenClaw Rex agent, sends "What time is it?", and confirms a non-empty response.

9. **Create a smoke test for persona** that sends a prompt and verifies the response includes Rex's persona markers (name, style).

10. **Document the OpenClaw agent bootstrap** in `docs/openclaw-agent-setup.md`.

#### Phase 2 Validation

- [ ] `rex/openclaw/agent.py` exists and registers with OpenClaw
- [ ] Agent responds to text prompts using Rex's LLM client
- [ ] Rex persona is present in responses
- [ ] `time_now` and `weather` tools work through OpenClaw
- [ ] Smoke tests pass
- [ ] Existing Rex tests still pass (no regressions)

---

### Phase 3: Port Persona, Memory Conventions, Policy, and Approval Logic

**Objective:** Make the OpenClaw-based Rex agent use Rex's memory, policy, and approval systems.

**Duration estimate:** 12-18 Ralph Loop iterations

#### Phase 3 Tasks

**Memory:**

1. **Audit `rex/memory.py` public API.** List every public function and class. Document which are called from `assistant.py`, `voice_loop.py`, `cli.py`, and `autonomy/`.

2. **Create `rex/openclaw/memory_adapter.py`** that exposes Rex's memory API but stores data using OpenClaw's session/workspace storage if available, falling back to Rex's existing file-based storage.

3. **Test memory adapter** with a unit test that writes, reads, and trims conversation history through the adapter.

4. **Audit `rex/memory_utils.py` public API.** List callers. Determine which utils are Rex-specific vs generic.

5. **Wire memory adapter into the OpenClaw Rex agent.** The agent's `generate_reply` path should use the adapter.

6. **Verify memory persistence** across two consecutive agent interactions in a test.

**Policy:**

7. **Audit `rex/policy.py` and `rex/policy_engine.py` public API.** List every `PolicyDecision` consumer. List every tool that is policy-gated.

8. **Create `rex/openclaw/policy_adapter.py`** that wraps Rex's PolicyEngine as an OpenClaw middleware or hook. Rex's policy decisions must be the authority; OpenClaw executes but does not override.

9. **Test policy adapter** with a unit test: register a tool, set it to "requires approval", call it, confirm it blocks.

10. **Test policy adapter auto-allow** path: register a low-risk tool, confirm it executes without approval.

11. **Wire policy adapter into the OpenClaw Rex agent's tool execution path.**

**Identity:**

12. **Audit `rex/identity.py` public API.** List every caller. Document session state file behavior.

13. **Create `rex/openclaw/identity_adapter.py`** that maps Rex's user identity resolution to OpenClaw's session/user model.

14. **Test identity adapter** with a unit test: set active user, confirm it propagates to OpenClaw session.

**Profile:**

15. **Audit `rex/profile_manager.py` public API.** List every caller. Document profile merge behavior.

16. **Wire profile manager into OpenClaw agent config** so profiles affect persona/behavior within the OpenClaw session.

**Approval:**

17. **Audit Rex's approval system** (approval files in `data/approvals/`). List every module that creates or checks approvals.

18. **Create `rex/openclaw/approval_adapter.py`** that bridges Rex's file-based approval system to OpenClaw's approval/permission model.

19. **Test approval adapter** end-to-end: create a workflow step requiring approval, confirm it blocks, approve, confirm it proceeds.

#### Phase 3 Validation

- [ ] Memory adapter reads/writes correctly through OpenClaw
- [ ] Policy engine gates tool calls through OpenClaw
- [ ] Identity resolution works through OpenClaw sessions
- [ ] Profile merging affects OpenClaw agent behavior
- [ ] Approval flow works end-to-end through OpenClaw
- [ ] All existing Rex tests pass
- [ ] Voice loop still works (regression check)

---

### Phase 4: Port Browser-Dependent and Service-Dependent Workflows

**Objective:** Migrate the tool routing, event bus, browser automation, and workflow execution to use OpenClaw.

**Duration estimate:** 20-30 Ralph Loop iterations

#### Phase 4a: Tool Routing Migration

1. **List every tool registered in `rex/tool_router.py`.** Extract the complete tool name/handler mapping.

2. **Classify each tool** as: Rex-specific (keep), generic (replace with OpenClaw), or adapter-needed.

3. **Create `rex/openclaw/tool_bridge.py`** that implements the `ToolRouting` Protocol from `rex/contracts/` and delegates to OpenClaw's tool system.

4. **Register the first batch of simple tools** (read-only, no side effects) with OpenClaw via the bridge. Examples: `time_now`, `weather`, `geolocation`.

5. **Test the first batch** through the OpenClaw agent.

6. **Register the second batch of tools** (side effects, policy-gated) with OpenClaw via the bridge. Examples: `send_email`, `send_sms`, `calendar_create`.

7. **Test the second batch** with policy gating confirmed.

8. **Register HA tools** (`ha_bridge` functions) with OpenClaw.

9. **Test HA tools** through the OpenClaw agent.

10. **Update `rex/assistant.py`** to use the tool bridge instead of calling `tool_router.route_if_tool_request` directly. Use feature flag (`USE_OPENCLAW_TOOLS=true/false` in config).

11. **Test assistant with both old and new tool paths** to confirm parity.

#### Phase 4b: Event Bus Migration

12. **List every event type published on `rex/event_bus.py`.** List every subscriber.

13. **Classify each event** as: framework-level (replace), Rex-specific (keep), or bridge-needed.

14. **Create `rex/openclaw/event_bridge.py`** that implements the `EventBus` Protocol and delegates to OpenClaw's event system.

15. **Test event bridge** with a publish/subscribe round-trip test.

16. **Update the first consumer** (pick the simplest subscriber) to use the event bridge.

17. **Test that consumer** in isolation.

18. **Update remaining consumers** one at a time, testing after each.

#### Phase 4c: Browser Automation Migration

19. **List every file that imports `rex.browser_automation`.** Use grep.

20. **List every public function currently used from `rex.browser_automation`.** Classify each usage as: read-only navigation, login/session handling, scraping, form submission, or browser orchestration.

21. **Determine which usages OpenClaw browser already covers directly.** Mark unsupported usages for adapter or redesign.

22. **Create `rex/openclaw/browser_bridge.py`** that implements the `BrowserAutomation` Protocol and delegates to OpenClaw's browser control.

23. **Test one simple browser task** (navigate to URL, take screenshot) through OpenClaw.

24. **Test one authenticated browser task** (login flow) through OpenClaw. If OpenClaw doesn't support Rex's credential-based login helpers, document the gap and keep Rex's implementation as a fallback behind the bridge.

25. **Update one calling module** to use the browser bridge.

26. **Test that module** end-to-end.

27. **Update remaining browser callers** one at a time.

28. **Document remaining browser migration backlog** (anything that couldn't migrate yet).

#### Phase 4d: Workflow and Executor Migration

29. **Audit `rex/workflow.py` and `rex/workflow_runner.py` public API.** List every caller.

30. **Determine OpenClaw's equivalent of Rex workflows.** Map Rex's `Workflow` / `WorkflowStep` model to OpenClaw's skill/task model.

31. **Create `rex/openclaw/workflow_bridge.py`** that translates Rex workflow definitions into OpenClaw-native execution, preserving Rex's policy hooks.

32. **Test one simple workflow** (single-step, no approval) through the bridge.

33. **Test one approval-gated workflow** through the bridge.

34. **Test one multi-step workflow** through the bridge.

35. **Update `rex/autonomy/runner.py`** to execute workflows through the bridge.

36. **Test autonomy runner** with the bridge.

#### Phase 4 Validation

- [ ] All registered tools callable through OpenClaw
- [ ] Event publishing and subscribing works through OpenClaw
- [ ] Simple browser tasks work through OpenClaw
- [ ] Workflows execute through OpenClaw with policy gating
- [ ] Autonomy runner operates through the bridge
- [ ] Feature flag allows fallback to old paths
- [ ] All existing tests pass
- [ ] Voice loop still works

---

### Phase 5: Port Home Assistant, WordPress, WooCommerce, Plex, and Business Workflows

**Objective:** Re-home Rex's domain integrations so they register as OpenClaw skills/tools rather than being wired through Rex's internal plumbing.

**Duration estimate:** 15-20 Ralph Loop iterations

#### Phase 5a: Home Assistant

1. **Audit `rex/ha_bridge.py` public API.** List every function exposed as a tool or called by other modules.

2. **Audit `rex/ha_tts/*` public API.** List callers.

3. **Create `rex/openclaw/skills/ha_skill.py`** that registers HA bridge functions as OpenClaw skills/tools.

4. **Test HA skill registration** -- confirm tools appear in OpenClaw's tool listing.

5. **Test one HA command** (e.g., turn on a light) through the OpenClaw agent.

6. **Test HA TTS** through the OpenClaw agent.

7. **Wire HA event subscriptions** through OpenClaw's event system (if HA publishes state changes to Rex's event bus).

#### Phase 5b: WordPress

8. **Audit `rex/wordpress/*` public API.** List every function exposed as a tool.

9. **Create `rex/openclaw/skills/wordpress_skill.py`** that registers WordPress functions as OpenClaw skills.

10. **Test one WordPress read operation** through the OpenClaw agent.

11. **Test one WordPress write operation** through the OpenClaw agent (with policy gating).

#### Phase 5c: WooCommerce

12. **Audit `rex/woocommerce/*` public API.** List every function exposed as a tool. Note the write policy (`rex/woocommerce/write_policy.py`).

13. **Create `rex/openclaw/skills/woocommerce_skill.py`** that registers WooCommerce functions as OpenClaw skills, with Rex's write policy enforced.

14. **Test one WooCommerce read operation** through the OpenClaw agent.

15. **Test one WooCommerce write operation** (with write policy + general policy gating) through the OpenClaw agent.

#### Phase 5d: Plex

16. **Audit `rex/plex_client.py` public API.** List every function exposed as a tool.

17. **Create `rex/openclaw/skills/plex_skill.py`** that registers Plex functions as OpenClaw skills.

18. **Test one Plex command** (e.g., search, play) through the OpenClaw agent.

#### Phase 5e: Business Workflows and Nasteeshirts

19. **Identify all business-specific workflow definitions** in `data/workflows/` or code. List them.

20. **Identify Nasteeshirts-specific logic** scattered across modules. List files and functions.

21. **Create `rex/openclaw/skills/business_skill.py`** that registers business workflow triggers as OpenClaw skills.

22. **Test one business workflow** end-to-end through OpenClaw.

#### Phase 5 Validation

- [ ] HA bridge commands work through OpenClaw
- [ ] WordPress read/write works through OpenClaw
- [ ] WooCommerce read/write works through OpenClaw with write policy
- [ ] Plex commands work through OpenClaw
- [ ] At least one business workflow executes through OpenClaw
- [ ] All existing tests pass
- [ ] Voice loop still works

---

### Phase 6: Reattach Voice, Wakeword, and TTS/STT Layers on Top of OpenClaw

**Objective:** The voice pipeline continues to use Rex's voice_loop, wakeword, and voice_identity modules, but the assistant backend it talks to is now the OpenClaw-based Rex agent.

**Duration estimate:** 10-15 Ralph Loop iterations

#### Phase 6 Tasks

1. **Audit `voice_loop.py` (root-level) call path.** Trace: wakeword detection -> STT -> assistant.generate_reply -> TTS. Identify every module touched.

2. **Audit `rex/voice_loop.py` call path.** Same trace. Note differences from root-level `voice_loop.py`.

3. **Audit `rex/voice_loop_optimized.py` call path.** Same trace.

4. **Identify the exact interface between voice loop and assistant.** This is the seam where the voice loop calls the assistant. Currently it calls `Assistant.generate_reply()`.

5. **Create `rex/openclaw/voice_bridge.py`** that exposes a `generate_reply(user_text, context) -> str` interface, backed by the OpenClaw Rex agent.

6. **Add feature flag** `USE_OPENCLAW_VOICE_BACKEND=true/false` to config.

7. **Update root-level `voice_loop.py`** to use the voice bridge when the flag is enabled. Keep the old path as fallback.

8. **Test voice loop with OpenClaw backend** in text-only mode (skip actual audio, pass synthetic text).

9. **Update `rex/voice_loop.py`** to use the voice bridge when the flag is enabled.

10. **Test `rex/voice_loop.py` with OpenClaw backend** in text-only mode.

11. **Update `rex/voice_loop_optimized.py`** to use the voice bridge when the flag is enabled.

12. **Test `rex/voice_loop_optimized.py` with OpenClaw backend** in text-only mode.

13. **Integration test: wakeword -> STT -> OpenClaw agent -> TTS** end-to-end with synthetic audio input and captured audio output.

14. **Test voice identity** still works with the OpenClaw backend (speaker recognition feeds into OpenClaw session identity).

15. **Test HA TTS** path through voice loop + OpenClaw (TTS output routed to HA media player).

#### Phase 6 Validation

- [ ] Voice loop works with OpenClaw backend via feature flag
- [ ] Wakeword detection unchanged
- [ ] STT (Whisper) unchanged
- [ ] TTS output unchanged
- [ ] Voice identity recognition unchanged
- [ ] HA TTS routing unchanged
- [ ] Feature flag allows instant rollback to old path
- [ ] All existing tests pass

---

### Phase 7: Retire Redundant Rex Subsystems Safely

**Objective:** Remove the old Rex implementations that have been fully replaced by OpenClaw, after confirming the replacements are proven.

**Duration estimate:** 15-20 Ralph Loop iterations

#### Phase 7 Tasks

**Pre-retirement checklist (per module):**

For EACH module being retired, execute these steps in order:

1. **Confirm the OpenClaw replacement passes all tests** that the Rex module's tests cover.
2. **Confirm no module still imports the Rex implementation directly** (grep for imports).
3. **Confirm the feature flag has been set to OpenClaw for at least one full test cycle.**
4. **Remove the feature flag and hardcode the OpenClaw path.**
5. **Delete the Rex module file.**
6. **Delete the Rex module's dedicated tests** (if they only tested the now-removed implementation).
7. **Run full test suite.**

**Retirement order (safest first):**

8. **Retire `rex/event_bus.py`** (lowest coupling after migration).

9. **Retire `rex/plugin_loader.py`** (small file, simple replacement).

10. **Retire `rex/tool_registry.py`** (replaced by OpenClaw's tool system).

11. **Retire `rex/tool_router.py`** (replaced by OpenClaw tool routing via bridge).

12. **Retire `rex/executor.py`** (replaced by OpenClaw's task execution).

13. **Retire `rex/browser_automation.py`** (replaced by OpenClaw browser, but only after all browser-dependent callers are confirmed working).

14. **Retire `rex/dashboard/*` and `rex/dashboard_store.py`** (replaced by OpenClaw dashboard, but only after confirming SSE, auth, and all UI features are equivalent).

15. **Retire `rex/messaging_backends/*` and `rex/messaging_service.py`** (replaced by OpenClaw channels, but only after confirming Twilio webhook wiring works through OpenClaw).

16. **Clean up `rex/contracts/`** -- remove Protocol files that were only needed as migration interfaces.

17. **Update `rex/cli.py`** -- remove all command paths that referenced retired modules. This is the final and most delicate step because the CLI is 4,941 lines.

18. **Update `pyproject.toml` entry points** if any changed.

19. **Update `CLAUDE.md`** to reflect the new architecture.

20. **Update `README.md`** and `INSTALL.md`** to reflect the new architecture.

#### Phase 7 Validation

- [ ] No imports of retired modules remain in codebase (grep confirms)
- [ ] All tests pass
- [ ] Voice loop works end-to-end
- [ ] Dashboard (OpenClaw) works
- [ ] Messaging/channels work
- [ ] Browser automation works
- [ ] All integrations (HA, WordPress, WooCommerce, Plex) work
- [ ] CLI commands work
- [ ] No dead code remaining from retired modules

---

## 10. Task Breakdown for Ralph Loop Execution

Each task below is sized for one Ralph Loop iteration (~10 min of AI work, one context window).

### Phase 1 Tasks (US-P1-001 through US-P1-008)

**US-P1-001: Create migration tracking document**
As a developer, I need a migration status document so I can track which modules have been migrated.
- [x] Create `docs/openclaw-migration-status.md`
- [x] Include table with columns: Module, Classification, Status, Notes
- [x] Populate with all modules from Appendix A
- [x] Commit

**US-P1-002: Add OPENCLAW-REPLACE markers**
As a developer, I need freeze markers so no new features are added to modules being replaced.
- [x] Add `# OPENCLAW-REPLACE: This module will be replaced by OpenClaw. Do not add new features.` as the first comment line in: `rex/event_bus.py`, `rex/tool_registry.py`, `rex/tool_router.py`, `rex/plugin_loader.py`, `rex/browser_automation.py`, `rex/dashboard/__init__.py`, `rex/dashboard/routes.py`, `rex/dashboard/sse.py`, `rex/dashboard/auth.py`, `rex/dashboard_store.py`, `rex/messaging_service.py`, `rex/executor.py`
- [x] Tests pass unchanged

**US-P1-003: Add OPENCLAW-WRAP markers**
As a developer, I need wrap markers so I know which modules will be adapted rather than replaced.
- [x] Add `# OPENCLAW-WRAP: This module will be wrapped around OpenClaw. Preserve public API.` in: `rex/autonomy/__init__.py`, `rex/autonomy/runner.py`, `rex/autonomy/llm_planner.py`, `rex/autonomy/rule_planner.py`, `rex/policy.py`, `rex/policy_engine.py`, `rex/identity.py`, `rex/profile_manager.py`, `rex/workflow.py`, `rex/workflow_runner.py`
- [x] Tests pass unchanged

**US-P1-004: Extract tool routing Protocol**
As a developer, I need a formal interface contract for tool routing so adapters can be swapped in.
- [x] Create `rex/contracts/tool_routing.py`
- [x] Define `ToolRoutingProtocol` with methods matching `tool_router.py`'s public API
- [x] Add type hints and docstrings
- [x] Tests pass

**US-P1-005: Extract event bus Protocol**
As a developer, I need a formal interface contract for the event bus.
- [x] Create `rex/contracts/event_bus.py`
- [x] Define `EventBusProtocol` matching both the simple and rich API
- [x] Tests pass

**US-P1-006: Extract browser automation Protocol**
As a developer, I need a formal interface contract for browser automation.
- [x] Create `rex/contracts/browser.py`
- [x] Define `BrowserAutomationProtocol` matching `browser_automation.py` public API
- [x] Tests pass

**US-P1-007: Extract dashboard Protocol**
As a developer, I need a formal interface contract for the dashboard.
- [x] Create `rex/contracts/dashboard.py`
- [x] Define `DashboardProtocol` covering route registration, SSE, auth
- [x] Tests pass

**US-P1-008: Extract plugin/tool loading Protocol**
As a developer, I need a formal interface contract for plugin and tool loading.
- [x] Create `rex/contracts/plugins.py`
- [x] Define `PluginLoaderProtocol` and `ToolRegistryProtocol`
- [x] Tests pass

### Phase 2 Tasks (US-P2-001 through US-P2-010)

**US-P2-001: Create rex/openclaw subpackage**
- [x] Create `rex/openclaw/__init__.py` with module docstring
- [x] Tests pass

**US-P2-002: Create OpenClaw agent registration module**
- [x] Create `rex/openclaw/agent.py`
- [x] Implement `RexAgent` class that registers with OpenClaw's agent API
- [x] Accept text prompt, return text response via `rex.llm_client.LanguageModel.generate()`
- [x] Tests pass

**US-P2-003: Create OpenClaw config bridge**
- [x] Create `rex/openclaw/config.py`
- [x] Map `rex.config.Settings` fields to OpenClaw agent configuration
- [x] Tests pass

**US-P2-004: Create OpenClaw session bridge**
- [x] Create `rex/openclaw/session.py`
- [x] Map Rex user identity to OpenClaw session model
- [x] Tests pass

**US-P2-005: Wire Rex persona into OpenClaw agent**
- [x] Update `rex/openclaw/agent.py` to inject system prompt and personality from Rex config
- [x] Tests pass

**US-P2-006: Register time_now tool with OpenClaw**
- [x] Create `rex/openclaw/tools/__init__.py`
- [x] Create `rex/openclaw/tools/time_tool.py` that registers `time_now` with OpenClaw
- [x] Tests pass

**US-P2-007: Register weather tool with OpenClaw**
- [x] Create `rex/openclaw/tools/weather_tool.py` that registers `weather` with OpenClaw
- [x] Tests pass

**US-P2-008: Smoke test -- basic agent response**
- [x] Create `tests/test_openclaw_agent_basic.py`
- [x] Test: start agent, send "What time is it?", confirm non-empty response
- [x] Tests pass

**US-P2-009: Smoke test -- persona verification**
- [x] Add test in `tests/test_openclaw_agent_basic.py`
- [x] Test: send prompt, verify persona markers in response
- [x] Tests pass

**US-P2-010: Document OpenClaw agent setup**
- [x] Create `docs/openclaw-agent-setup.md`
- [x] Document bootstrap steps, config mapping, tool registration
- [x] Tests pass

### Phase 3 Tasks (US-P3-001 through US-P3-019)

**US-P3-001: Audit memory.py public API**
- [x] List every public function/class in `rex/memory.py`
- [x] List every caller (grep for `from rex.memory` and `import rex.memory`)
- [x] Document findings in `docs/openclaw-migration-status.md`

**US-P3-002: Create memory adapter**
- [x] Create `rex/openclaw/memory_adapter.py`
- [x] Implement adapter that delegates to OpenClaw storage with Rex fallback
- [x] Tests pass

**US-P3-003: Test memory adapter**
- [x] Create `tests/test_openclaw_memory.py`
- [x] Test: write, read, trim conversation history
- [x] Tests pass

**US-P3-004: Audit memory_utils.py public API**
- [x] List public functions, list callers
- [x] Classify as Rex-specific vs generic
- [x] Document in migration status

**US-P3-005: Wire memory adapter into OpenClaw agent**
- [x] Update `rex/openclaw/agent.py` to use memory adapter in generate_reply path
- [x] Tests pass

**US-P3-006: Test memory persistence across interactions**
- [x] Add test: two consecutive agent interactions, verify history persists
- [x] Tests pass

**US-P3-007: Audit policy.py and policy_engine.py public API**
- [x] List every PolicyDecision consumer
- [x] List every policy-gated tool
- [x] Document in migration status

**US-P3-008: Create policy adapter**
- [x] Create `rex/openclaw/policy_adapter.py`
- [x] Wrap Rex PolicyEngine as OpenClaw middleware/hook
- [x] Tests pass

**US-P3-009: Test policy adapter -- block path**
- [x] Test: register tool, set requires-approval, call it, confirm blocks
- [x] Tests pass

**US-P3-010: Test policy adapter -- allow path**
- [x] Test: register low-risk tool, confirm auto-executes
- [x] Tests pass

**US-P3-011: Wire policy adapter into OpenClaw agent**
- [x] Update tool execution path to pass through policy adapter
- [x] Tests pass

**US-P3-012: Audit identity.py public API**
- [x] List callers, document session state behavior
- [x] Document in migration status

**US-P3-013: Create identity adapter**
- [x] Create `rex/openclaw/identity_adapter.py`
- [x] Map Rex identity to OpenClaw session/user
- [x] Tests pass

**US-P3-014: Test identity adapter**
- [x] Test: set active user, confirm propagates to OpenClaw session
- [x] Tests pass

**US-P3-015: Audit profile_manager.py public API**
- [x] List callers, document merge behavior
- [x] Document in migration status

**US-P3-016: Wire profile manager into OpenClaw agent**
- [x] Profiles affect persona/behavior within OpenClaw session
- [x] Tests pass

**US-P3-017: Audit approval system**
- [x] List every module that creates/checks approvals in `data/approvals/`
- [x] Document in migration status

**US-P3-018: Create approval adapter**
- [x] Create `rex/openclaw/approval_adapter.py`
- [x] Bridge file-based approvals to OpenClaw model
- [x] Tests pass

**US-P3-019: Test approval adapter end-to-end**
- [x] Test: workflow step requiring approval -> blocks -> approve -> proceeds
- [x] Tests pass

### Phase 4 Tasks (US-P4-001 through US-P4-036)

**US-P4-001: List all registered tools in tool_router.py**
- [x] Extract complete tool name/handler mapping
- [x] Document in migration status

**US-P4-002: Classify each tool**
- [ ] Mark each as: Rex-specific, generic (replace), adapter-needed
- [ ] Document in migration status

**US-P4-003: Create tool bridge**
- [ ] Create `rex/openclaw/tool_bridge.py` implementing ToolRouting Protocol
- [ ] Delegates to OpenClaw tool system
- [ ] Tests pass

**US-P4-004: Register simple read-only tools batch**
- [ ] Register `time_now`, `weather`, `geolocation` via bridge
- [ ] Tests pass

**US-P4-005: Test simple tools through OpenClaw**
- [ ] Verify each tool returns correct results
- [ ] Tests pass

**US-P4-006: Register policy-gated tools batch**
- [ ] Register `send_email`, `send_sms`, `calendar_create` via bridge
- [ ] Tests pass

**US-P4-007: Test policy-gated tools**
- [ ] Confirm policy gating works for each
- [ ] Tests pass

**US-P4-008: Register HA tools**
- [ ] Register `ha_bridge` functions via bridge
- [ ] Tests pass

**US-P4-009: Test HA tools through OpenClaw**
- [ ] Verify HA commands execute correctly
- [ ] Tests pass

**US-P4-010: Add feature flag for tool routing**
- [ ] Add `USE_OPENCLAW_TOOLS` to config
- [ ] Tests pass

**US-P4-011: Update assistant.py to use tool bridge with feature flag**
- [ ] When flag enabled, use bridge; otherwise use old `tool_router`
- [ ] Tests pass

**US-P4-012: Test assistant with both tool paths**
- [ ] Confirm parity between old and new
- [ ] Tests pass

**US-P4-013: List all event types on event bus**
- [ ] List every publish call and every subscriber
- [ ] Document in migration status

**US-P4-014: Classify each event type**
- [ ] Mark as framework-level, Rex-specific, or bridge-needed
- [ ] Document

**US-P4-015: Create event bridge**
- [ ] Create `rex/openclaw/event_bridge.py` implementing EventBus Protocol
- [ ] Tests pass

**US-P4-016: Test event bridge round-trip**
- [ ] Publish event, confirm subscriber receives it
- [ ] Tests pass

**US-P4-017: Update first event consumer to use bridge**
- [ ] Pick simplest subscriber, update, test
- [ ] Tests pass

**US-P4-018: Update remaining event consumers**
- [ ] One at a time, test after each
- [ ] Tests pass

**US-P4-019: List all browser_automation importers**
- [ ] Grep for imports, list every file
- [ ] Document

**US-P4-020: Classify browser usage patterns**
- [ ] Categorize: navigation, login, scraping, form, orchestration
- [ ] Document

**US-P4-021: Determine OpenClaw browser coverage**
- [ ] Map Rex browser functions to OpenClaw equivalents
- [ ] Mark gaps
- [ ] Document

**US-P4-022: Create browser bridge**
- [ ] Create `rex/openclaw/browser_bridge.py` implementing Browser Protocol
- [ ] Tests pass

**US-P4-023: Test simple browser task through OpenClaw**
- [ ] Navigate to URL, take screenshot
- [ ] Tests pass

**US-P4-024: Test authenticated browser task**
- [ ] Login flow through OpenClaw
- [ ] Document gaps if any
- [ ] Tests pass

**US-P4-025: Update first browser caller to use bridge**
- [ ] Update, test
- [ ] Tests pass

**US-P4-026: Update remaining browser callers**
- [ ] One at a time, test after each
- [ ] Tests pass

**US-P4-027: Document browser migration backlog**
- [ ] List anything that couldn't migrate yet
- [ ] Document

**US-P4-028: Audit workflow.py and workflow_runner.py public API**
- [ ] List callers
- [ ] Document

**US-P4-029: Map Rex workflows to OpenClaw model**
- [ ] Document the mapping between Workflow/WorkflowStep and OpenClaw equivalents
- [ ] Document

**US-P4-030: Create workflow bridge**
- [ ] Create `rex/openclaw/workflow_bridge.py`
- [ ] Preserve Rex policy hooks
- [ ] Tests pass

**US-P4-031: Test simple workflow through bridge**
- [ ] Single-step, no approval
- [ ] Tests pass

**US-P4-032: Test approval-gated workflow through bridge**
- [ ] Multi-step with approval gate
- [ ] Tests pass

**US-P4-033: Test multi-step workflow through bridge**
- [ ] 3+ steps, mixed policies
- [ ] Tests pass

**US-P4-034: Update autonomy/runner.py to use workflow bridge**
- [ ] Tests pass

**US-P4-035: Test autonomy runner with bridge**
- [ ] End-to-end autonomy execution
- [ ] Tests pass

**US-P4-036: Phase 4 regression test**
- [ ] Full test suite pass
- [ ] Voice loop regression check

### Phase 5 Tasks (US-P5-001 through US-P5-022)

**US-P5-001:** Audit ha_bridge.py public API
**US-P5-002:** Audit ha_tts/* public API
**US-P5-003:** Create HA skill for OpenClaw
**US-P5-004:** Test HA skill registration
**US-P5-005:** Test one HA command through OpenClaw
**US-P5-006:** Test HA TTS through OpenClaw
**US-P5-007:** Wire HA event subscriptions through OpenClaw
**US-P5-008:** Audit wordpress/* public API
**US-P5-009:** Create WordPress skill for OpenClaw
**US-P5-010:** Test WordPress read through OpenClaw
**US-P5-011:** Test WordPress write through OpenClaw
**US-P5-012:** Audit woocommerce/* public API
**US-P5-013:** Create WooCommerce skill for OpenClaw
**US-P5-014:** Test WooCommerce read through OpenClaw
**US-P5-015:** Test WooCommerce write through OpenClaw (with write policy)
**US-P5-016:** Audit plex_client.py public API
**US-P5-017:** Create Plex skill for OpenClaw
**US-P5-018:** Test Plex command through OpenClaw
**US-P5-019:** Identify all business workflow definitions
**US-P5-020:** Identify Nasteeshirts-specific logic
**US-P5-021:** Create business skill for OpenClaw
**US-P5-022:** Test one business workflow end-to-end

### Phase 6 Tasks (US-P6-001 through US-P6-015)

**US-P6-001:** Audit root voice_loop.py call path
**US-P6-002:** Audit rex/voice_loop.py call path
**US-P6-003:** Audit rex/voice_loop_optimized.py call path
**US-P6-004:** Identify voice loop -> assistant seam
**US-P6-005:** Create voice bridge
**US-P6-006:** Add USE_OPENCLAW_VOICE_BACKEND feature flag
**US-P6-007:** Update root voice_loop.py with feature flag
**US-P6-008:** Test root voice loop with OpenClaw (text mode)
**US-P6-009:** Update rex/voice_loop.py with feature flag
**US-P6-010:** Test rex/voice_loop.py with OpenClaw (text mode)
**US-P6-011:** Update rex/voice_loop_optimized.py with feature flag
**US-P6-012:** Test rex/voice_loop_optimized.py with OpenClaw (text mode)
**US-P6-013:** Integration test: wakeword -> STT -> OpenClaw -> TTS
**US-P6-014:** Test voice identity with OpenClaw backend
**US-P6-015:** Test HA TTS through voice loop + OpenClaw

### Phase 7 Tasks (US-P7-001 through US-P7-020)

**US-P7-001:** Pre-retirement check for event_bus.py
**US-P7-002:** Retire event_bus.py
**US-P7-003:** Pre-retirement check for plugin_loader.py
**US-P7-004:** Retire plugin_loader.py
**US-P7-005:** Pre-retirement check for tool_registry.py
**US-P7-006:** Retire tool_registry.py
**US-P7-007:** Pre-retirement check for tool_router.py
**US-P7-008:** Retire tool_router.py
**US-P7-009:** Pre-retirement check for executor.py
**US-P7-010:** Retire executor.py
**US-P7-011:** Pre-retirement check for browser_automation.py
**US-P7-012:** Retire browser_automation.py
**US-P7-013:** Pre-retirement check for dashboard/* and dashboard_store.py
**US-P7-014:** Retire dashboard/* and dashboard_store.py
**US-P7-015:** Pre-retirement check for messaging_backends/* and messaging_service.py
**US-P7-016:** Retire messaging_backends/* and messaging_service.py
**US-P7-017:** Clean up rex/contracts/ migration interfaces
**US-P7-018:** Update cli.py to remove references to retired modules
**US-P7-019:** Update pyproject.toml entry points
**US-P7-020:** Update CLAUDE.md, README.md, INSTALL.md

---

## 11. Validation and Acceptance Criteria

### Per-Phase Validation

Every phase must pass these checks before the next phase begins:

1. `pytest -q` passes with no new failures
2. No import errors when running `python -c "import rex"`
3. Voice loop smoke test: `python rex_loop.py` starts without error (can be killed after 5s)
4. CLI smoke test: `python -m rex --help` works
5. No `# OPENCLAW-REPLACE` module has gained new features since Phase 1

### Final Acceptance Criteria

- [ ] Rex runs as an OpenClaw agent for all text interactions
- [ ] Voice loop works end-to-end through OpenClaw backend
- [ ] All integrations (HA, WordPress, WooCommerce, Plex) work as OpenClaw skills
- [ ] Policy engine gates all tool calls through OpenClaw
- [ ] No retired modules remain in codebase
- [ ] CLI works with all commands
- [ ] Dashboard runs via OpenClaw
- [ ] All tests pass
- [ ] `docs/openclaw-migration-status.md` shows all modules at "Complete"

---

## 12. Rollback and Recovery Plan

### Per-Commit Rollback

Every Ralph Loop iteration produces one commit. Rollback = `git revert <commit>`.

### Per-Phase Rollback

Every phase uses feature flags for the new path. Rollback = set flag to `false`.

Feature flags:
- `USE_OPENCLAW_TOOLS` (Phase 4a)
- `USE_OPENCLAW_EVENTS` (Phase 4b)
- `USE_OPENCLAW_BROWSER` (Phase 4c)
- `USE_OPENCLAW_WORKFLOWS` (Phase 4d)
- `USE_OPENCLAW_VOICE_BACKEND` (Phase 6)

### Emergency Rollback

If a phase introduces a regression that can't be quickly fixed:

1. Revert all commits in that phase: `git revert --no-commit HEAD~N..HEAD && git commit`
2. Set all feature flags to `false`
3. Run full test suite to confirm clean state
4. Document the failure in `docs/openclaw-migration-status.md`

### Point of No Return

Phase 7 (retirement) is the point of no return for each module. Before retiring any module:
- The OpenClaw replacement must have run successfully for at least one full test cycle with the feature flag enabled
- A human must confirm the retirement (not automated)

---

## 13. Open Questions

| # | Question | Impact | Status |
|---|----------|--------|--------|
| 1 | What is OpenClaw's Python API for agent registration? | Blocks Phase 2 | Open |
| 2 | Does OpenClaw support custom tool registration with metadata (descriptions, risk levels, required credentials)? | Affects tool bridge design | Open |
| 3 | Does OpenClaw's browser automation support credential-based login helpers, or only raw Playwright actions? | Affects browser bridge complexity | Open |
| 4 | Does OpenClaw have a policy/permission hook system, or do we need to inject Rex's policy as middleware? | Affects policy adapter design | Open |
| 5 | Does OpenClaw's event system support both sync and async subscribers? Rex's event bus supports both. | Affects event bridge design | Open |
| 6 | Does OpenClaw's dashboard support SSE for real-time notifications? | Affects dashboard migration timeline | Open |
| 7 | How does OpenClaw handle multi-user sessions? Rex uses file-based session state. | Affects identity adapter design | Open |
| 8 | Does OpenClaw have a concept of "approval gates" in workflows, or is this Rex-only? | Affects workflow bridge design | Open |
| 9 | What is the performance overhead of routing tool calls through OpenClaw vs direct Rex dispatch? | Affects voice loop latency | Open |
| 10 | Should Rex's 4,941-line CLI be migrated to use OpenClaw's CLI framework (if one exists), or kept as-is with updated imports? | Affects Phase 7 scope | Open |
| 11 | Does OpenClaw support MQTT for audio routing (Rex uses MQTT for HA audio)? | Affects HA integration | Open |
| 12 | How should the GUI (`gui.py`, `gui_settings_tab.py`) interact with OpenClaw? This is explicitly out of scope but needs a future plan. | Future work | Deferred |

---

## 14. Suggested Execution Order

```
Phase 1 (8 tasks)      -- Foundation: markers, contracts, tracking
   |
Phase 2 (10 tasks)     -- Baseline: Rex agent in OpenClaw
   |
Phase 3 (19 tasks)     -- Core: memory, policy, identity, approvals
   |
Phase 4a (12 tasks)    -- Tools: bridge and registration
Phase 4b (6 tasks)     -- Events: bridge and consumers
Phase 4c (9 tasks)     -- Browser: bridge and callers
Phase 4d (8 tasks)     -- Workflows: bridge and autonomy
   |
Phase 5a (7 tasks)     -- HA integration
Phase 5b (4 tasks)     -- WordPress integration
Phase 5c (4 tasks)     -- WooCommerce integration
Phase 5d (3 tasks)     -- Plex integration
Phase 5e (4 tasks)     -- Business workflows
   |
Phase 6 (15 tasks)     -- Voice pipeline
   |
Phase 7 (20 tasks)     -- Retirement

Total: ~129 Ralph Loop iterations
```

Phase 4 sub-phases (a/b/c/d) can potentially be parallelized if separate developers work on them, but for a single Ralph Loop they should be sequential: 4a -> 4b -> 4c -> 4d.

Phase 5 sub-phases (a/b/c/d/e) are independent and can run in any order.

---

## Appendix A: Initial Replace / Keep / Wrap / Retire Map

| Subsystem | Category | Why | Migration Notes |
|-----------|----------|-----|-----------------|
| `rex/assistant.py` | **Wrap** | Central orchestration hub. Too coupled to rewrite. Needs to delegate to OpenClaw agent while preserving Rex's generate_reply interface. | Wire to OpenClaw agent via voice_bridge. Keep as thin coordinator. |
| `rex/browser_automation.py` | **Replace** | Generic Playwright wrapper. OpenClaw has browser control. | Create browser bridge. Migrate callers one at a time. Keep Rex login helpers as adapter if OpenClaw gaps exist. |
| `rex/dashboard/*` | **Replace** | Flask dashboard + SSE + auth. OpenClaw provides dashboard/UI. | Run both in parallel during transition. Replace only after feature parity confirmed. 4 files: `__init__`, `routes`, `sse`, `auth`. |
| `rex/dashboard_store.py` | **Replace** | Dashboard persistence layer. Coupled to dashboard. | Retires when dashboard retires. |
| `rex/messaging_backends/*` | **Replace** | Twilio, SMS, webhooks. OpenClaw owns channels. | 11 files. Migrate last in Phase 4 due to webhook complexity. Test webhook delivery before cutover. |
| `rex/messaging_service.py` | **Replace** | Messaging orchestration. | Retires with messaging_backends. |
| `rex/integrations/message_router.py` | **Replace** | Routes messages between channels. OpenClaw handles this. | Retires with messaging. |
| `rex/tool_registry.py` | **Replace** | Tool metadata + health checks. OpenClaw has skill/tool system. | Create Protocol first (Phase 1), then bridge (Phase 4). |
| `rex/tool_router.py` | **Replace** | Central tool dispatch (960 lines). OpenClaw routes tools. | Highest-risk replacement. Feature flag required. Test every registered tool through bridge before retirement. |
| `rex/plugin_loader.py` | **Replace** | Dynamic plugin discovery (56 lines). OpenClaw has plugins. | Small file, easy replacement. |
| `rex/executor.py` | **Replace** | Task execution engine. OpenClaw executes tasks. | Replaced by OpenClaw task execution via workflow bridge. |
| `rex/event_bus.py` | **Replace** | Pub-sub event system (436 lines). OpenClaw has events. | Dual API (simple + rich) must be preserved in bridge. |
| `rex/workflow.py` | **Wrap** | Workflow data models (668 lines). Rex-specific workflow definitions. | Models translate to OpenClaw skill/task definitions. Keep Rex models as the definition format; bridge translates at execution time. |
| `rex/workflow_runner.py` | **Wrap** | Workflow execution (864 lines). Has Rex policy hooks. | Bridge preserves policy gating. Rex policy is authority. |
| `rex/autonomy/*` | **Wrap** | LLM planner, goal graph, runner (~1200 lines). Rex's planning intelligence is unique. | High-level planning logic wraps OpenClaw's multi-agent primitives. Keep Rex's goal decomposition and replanning. |
| `rex/policy.py` | **Keep + Wrap** | Policy models (150 lines). Rex-specific risk classification. | Keep models. Wrap as OpenClaw middleware. Rex policy is always the authority. |
| `rex/policy_engine.py` | **Keep + Wrap** | Policy evaluation (350 lines). Rex-specific approval logic. | Keep engine. Wrap as OpenClaw hook. |
| `rex/identity.py` | **Wrap** | User identity resolution (280 lines). | Map to OpenClaw session/user model. Keep Rex's resolution logic (CLI flag, config, session file). |
| `rex/profile_manager.py` | **Wrap** | Profile merging (100 lines). | Keep merge logic. Wire into OpenClaw agent config. |
| `rex/voice_identity/*` | **Keep** | Speaker recognition. Uniquely Rex. | 7 files. No OpenClaw equivalent. Migrates in Phase 6 to feed into OpenClaw session identity. |
| `rex/wakeword/*` | **Keep** | Wake word detection. Uniquely Rex. | 4 files. No OpenClaw equivalent. Unchanged in migration. |
| `rex/voice_loop.py` | **Keep** | Core voice loop (800 lines). Uniquely Rex. | Update to call OpenClaw backend via voice_bridge. Feature flag for rollback. |
| `rex/voice_loop_optimized.py` | **Keep** | Low-latency voice loop (550 lines). Uniquely Rex. | Same treatment as voice_loop.py. |
| `rex/ha_bridge.py` | **Keep** | Home Assistant bridge (600 lines). Uniquely Rex. | Register as OpenClaw skill in Phase 5. Code stays in Rex. |
| `rex/ha_tts/*` | **Keep** | HA TTS integration. Uniquely Rex. | 3 files. Register as OpenClaw skill in Phase 5. |
| `rex/wordpress/*` | **Keep** | WordPress client. Uniquely Rex. | 3 files. Register as OpenClaw skill in Phase 5. |
| `rex/woocommerce/*` | **Keep** | WooCommerce client with write policy. Uniquely Rex. | 4 files. Register as OpenClaw skill in Phase 5. Write policy preserved. |
| `rex/plex_client.py` | **Keep** | Plex media control. Uniquely Rex. | Register as OpenClaw skill in Phase 5. |
| `rex/memory.py` | **Keep + Adapt** | Conversation memory (650 lines). Rex-specific conventions. | Keep memory logic. Adapter stores via OpenClaw if available, falls back to file storage. |
| `rex/memory_utils.py` | **Keep** | Memory helpers (350 lines). | Stays with memory.py. |
| `rex/llm_client.py` | **Keep** | Multi-provider LLM client. Orthogonal to migration. | Not in scope. Rex keeps its LLM layer. |
| `rex/config.py` | **Keep** | Pydantic settings (600 lines). | Add OpenClaw-specific fields as needed. |
| `rex/cli.py` | **Keep + Update** | CLI (4941 lines). | Update imports as modules are retired. Do not rewrite. One command at a time. |
| `rex/app.py` | **Retire (late)** | Flask app factory. | Retires when dashboard retires and OpenClaw handles HTTP. |
| `rex/api_key_auth.py` | **Retire (late)** | API key auth. | Retires when OpenClaw handles auth. |
| `rex/credentials.py` | **Keep** | Credential management (450 lines). | Rex-specific credential storage. May need adapter for OpenClaw tools that need credentials. |
| `rex/scheduler.py` | **Wrap** | Cron-like scheduling (675 lines). | Evaluate whether OpenClaw has scheduling. If yes, wrap. If no, keep. |
| `rex/planner.py` | **Wrap** | Task planning (640 lines). | Rex's planning logic wraps OpenClaw primitives. |
| `rex/notification.py` | **Wrap** | Notification system (884 lines). | Route through OpenClaw's notification/event system if available. |
| `rex/computers/*` | **Replace** | Windows agent server/client (~400 lines). OpenClaw has workspaces. | 5 files. Replace with OpenClaw workspace/agent model. |
| `rex/email_backends/*` | **Keep** | IMAP/SMTP email (~600 lines). | Rex-specific email integration. Register as OpenClaw skill. |
| `rex/email_service.py` | **Keep** | Email orchestration (662 lines). | Register as OpenClaw skill. |
| `rex/calendar_backends/*` | **Keep** | Calendar integrations (~500 lines). | Rex-specific. Register as OpenClaw skill. |
| `rex/calendar_service.py` | **Keep** | Calendar orchestration (700 lines). | Register as OpenClaw skill. |
| `rex/audit.py` | **Keep** | Audit logging. Security-critical. | Stays. May also feed into OpenClaw's audit if available. |
