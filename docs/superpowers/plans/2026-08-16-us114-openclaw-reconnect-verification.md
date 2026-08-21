# US-114 OpenClaw Reconnect and Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven development and verification-before-completion for every behavior change.

**Goal:** Reconcile OpenClaw authority after outages and keep remote mutation outcomes truthful under transport uncertainty.

**Architecture:** Add one process-wide reconnect coordinator beside the US-113 capability synchronizer. A detected transport outage immediately disables remote bindings, then a bounded daemon reconnect loop probes health and performs an authenticated capability resync before remote tools are enabled again. Mutation transport failures after dispatch are represented as outcome-unknown evidence and mapped into Rex's existing canonical action lifecycle; remote `verified` claims never bypass Rex-side postconditions.

**Tech Stack:** Python 3.11, requests, threading, existing OpenClaw HTTP/WebSocket clients, CapabilityRegistry/ToolRegistry, ActionLifecycle/ToolExecutionLifecycle, pytest.

## Global Constraints

- Reconnection restores connectivity, not trust; resync and policy evaluation come first.
- Core local Rex must remain functional while OpenClaw is down.
- No gateway token, prompt, tool args, private result, URL credentials, or user content in health events/log metadata.
- No mutation retry after an outcome becomes uncertain.
- Reads may use existing bounded retry/fallback policy.
- Reuse canonical action lifecycle; do not create a parallel success state machine.
- Keep US-113 inventory validation and authority rules unchanged.

---
### Task 1: Reconnect coordinator and privacy-safe health state

**Files:**
- Create: `rex/openclaw/reconnect.py`
- Create: `tests/rex2/test_openclaw_reconnect.py`

**Produces:** `OpenClawReconnectController`, `OpenClawReconnectState`, and a content-free health transition event.

- [ ] Write failing tests for outage transition, capped exponential backoff, no hot loop, single worker, and health-up-but-resync-failed staying unavailable.
- [ ] Run only the new reconnect tests and confirm RED because the coordinator does not exist.
- [ ] Implement the minimum thread-safe controller: `mark_disconnected()`, `require_ready()`, `run_until_recovered()`, `close()`, and privacy-safe event projection.
- [ ] Prove recovery becomes `ready` only after `health_probe()` succeeds **and** the injected authenticated `resync()` returns a fresh non-stale success.
- [ ] Run reconnect tests to GREEN and refactor only after they pass.

### Task 2: Wire reconnect authority into US-113 capability sync and dispatch

**Files:**
- Modify: `rex/openclaw/capability_sync.py`
- Modify: `rex/assistant.py` only if initialization needs explicit lifecycle wiring
- Test: `tests/rex2/test_openclaw_reconnect.py`

**Produces:** process-wide reconnect initialization and remote handler gating against stale/down authority.
- [ ] Write failing tests showing an outage disables remote executable bindings, local tools remain available, and a recovered gateway cannot dispatch until a fresh capability resync has applied schema/removal changes.
- [ ] Confirm RED against current handler behavior.
- [ ] Expose a narrow controller callback for marking current OpenClaw bindings unavailable and for `refresh(reason="reconnect")`.
- [ ] Initialize the reconnect controller from the same runtime config/client lifecycle and gate remote handlers with `require_ready()`.
- [ ] On transport failure, mark OpenClaw disconnected once and let the reconnect worker resync before re-enabling anything.
- [ ] Run US-113 capability-sync tests plus new reconnect tests to GREEN.

### Task 3: Normalize uncertain mutations and trusted verification

**Files:**
- Modify: `rex/openclaw/errors.py`
- Modify: `rex/openclaw/capability_sync.py`
- Modify: `rex/tools/execution.py`
- Create: `tests/rex2/test_openclaw_verification.py`

**Produces:** an explicit outcome-unknown signal for remote mutations and canonical lifecycle projection.

- [ ] Write failing tests for outage after mutation dispatch => `attempted_unverified`; no automatic mutation retry; remote self-declared `verified` => unverified without a Rex verifier; accepted Rex verifier/postcondition => `verified`.
- [ ] Confirm RED for the mutation transport-error path.
- [ ] Add a sanitized `OpenClawOutcomeUnknownError` (or equivalent marker) and wrap ambiguous post-dispatch transport/5xx failures in the remote handler.
- [ ] Teach `ToolExecutionLifecycle` to map that marker to `ATTEMPTED_UNVERIFIED` only for mutations; reads continue existing bounded retry/fallback behavior.
- [ ] Keep the existing `tool.verifier` callback as the only accepted independent postcondition path; remote result status alone never promotes a mutation.
- [ ] Run verification tests and existing tool lifecycle/action lifecycle tests to GREEN.
### Task 4: Story closure, regression gates, docs, and exact-head review

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/openclaw-migration-status.md`
- Modify: `PRD-production-readiness.md`
- Modify: `docs/archive/progress/progress-production-readiness.txt`

- [ ] Run the authoritative command: `pytest tests/rex2/test_openclaw_reconnect.py tests/rex2/test_openclaw_verification.py -q`.
- [ ] Run the broader OpenClaw/capability/tool lifecycle matrix, Ruff, Black, mypy, security release gate, pre-commit, and `git diff --check`.
- [ ] Run full `pytest -q`; preserve a clean working tree except intended story files.
- [ ] Independently review protocol/reconnect races, trust restoration, mutation uncertainty, health-event privacy, and verification authority with Codex; fix concrete findings test-first.
- [ ] Document the reconnect/verification invariants in `CLAUDE.md` and user/operator OpenClaw status docs.
- [ ] Mark local US-114 criteria complete but leave GitHub checks unchecked until the exact implementation head passes remotely.
- [ ] Commit, push, open a PR, require all exact-head checks plus review/thread cleanliness, then record the GitHub criterion on a docs-only closure head and re-run the gate before merge.

## Plan self-review

- All five US-114 behavioral criteria map to Tasks 1-3 and their named tests.
- The design has no second action-success state machine; all mutation truth stays in `ToolExecutionLifecycle` / `ActionLifecycle`.
- Reconnect state carries only bounded enum/reason/attempt/delay metadata.
- No requirement depends on US-115 or future Forge/composition work.
- No placeholder/TODO steps remain.
