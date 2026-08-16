# US-120 Timers and Alarms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver first-class, persistent, per-user concurrent timers and alarms with precise expiration, recurrence, snooze, restart recovery, and canonical Rex tool exposure.

**Architecture:** Add a focused `rex.timekeeping` package. Domain state and persistence live in a clock-injected `TimekeepingService`; a separate `TimekeepingRuntime` owns the wakeup thread and event delivery so deterministic service tests never depend on wall-clock sleeps. Two canonical tools expose read and mutation operations through the existing Capability Registry and ToolExecutionLifecycle.

**Tech Stack:** Python 3.11+, Pydantic, `zoneinfo`, `threading.Condition`, existing Rex identity/profile/notification/tool lifecycle infrastructure, pytest.

## Global Constraints

- Timers are not reminders and must not use the reminder service's 60-second polling job.
- All user-facing operations validate and isolate `user_id`; same display names may exist for different users.
- Timer deadlines persist as UTC instants; alarm rules retain their IANA timezone so DST recurrence is recalculated in local wall time.
- Mutations must use the canonical ToolExecutionLifecycle and independent state verification.
- No feature is advertised as available until its tests and capability registration are complete.

---

### Task 1: Canonical timer/alarm domain and persistence

**Files:**
- Create: `rex/timekeeping/models.py`
- Create: `rex/timekeeping/service.py`
- Create: `rex/timekeeping/__init__.py`
- Test: `tests/timekeeping/test_service.py`

**Interfaces:**
- `TimekeepingService(storage_path, now_func)` owns all persisted timer/alarm state.
- `TimerRecord` stores owner, optional name, active/paused/fired/canceled state, UTC deadline, pause remainder, and optional output target.
- `AlarmRecord` stores owner, optional name, timezone, local clock time, optional local date, recurring weekdays, next UTC occurrence, enabled/status state, snooze state, and optional output target.

- [ ] Write failing tests for concurrent timers, owner isolation, remaining-time calculation, pause/resume, rename, adjustment, cancellation, persistence, and overdue restart reconciliation.
- [ ] Implement immutable validation plus atomic JSON persistence under the canonical data root.
- [ ] Write failing tests for one-shot alarms, selected-weekday recurrence, enable/disable, edit, cancel, snooze, dismiss, timezone conversion, and DST boundary calculation.
- [ ] Implement alarm recurrence using `zoneinfo.ZoneInfo` and recompute the next local occurrence after each dismissal/fire.
- [ ] Run `pytest -q tests/timekeeping/test_service.py` and commit only when green.

### Task 2: Precise runtime scheduling and event delivery

**Files:**
- Create: `rex/timekeeping/runtime.py`
- Test: `tests/timekeeping/test_runtime.py`

**Interfaces:**
- `TimekeepingRuntime(service, event_handler, now_func)` starts/stops one daemon worker.
- The worker waits on a `threading.Condition` until the nearest active timer/alarm deadline, with immediate wakeup after mutations.
- `DueEvent` is emitted once per persisted firing transition; event delivery failure does not roll state backward or duplicate an already-recorded fire.

- [ ] Write a failing deterministic runtime test proving multiple deadlines fire in order without minute polling.
- [ ] Implement next-deadline wakeups with a bounded sub-second scheduling tolerance and explicit stop/wakeup signals.
- [ ] Add restart tests proving overdue items reconcile once and future items remain scheduled.
- [ ] Add notification adapter tests using the existing notifier contract without coupling the service to a delivery provider.
- [ ] Run `pytest -q tests/timekeeping/test_runtime.py tests/timekeeping/test_service.py` and commit when green.

### Task 3: Natural-language command parsing

**Files:**
- Create: `rex/timekeeping/parser.py`
- Test: `tests/timekeeping/test_parser.py`

**Interfaces:**
- `parse_timekeeping_command(transcript, *, user_timezone, now)` returns a typed `TimekeepingCommand` or `None`.
- Parsing is deterministic for supported timer/alarm verbs and units; ambiguous references return a structured ambiguity result instead of guessing.

- [ ] Write failing tests for timer durations in seconds/minutes/hours, named timers, list/query, add/subtract time, pause/resume/cancel/rename, absolute alarms, tomorrow, weekdays, selected weekdays, snooze, dismiss, and enable/disable.
- [ ] Implement only the explicit grammar required by the US-120 examples and tests; do not introduce a general NLP framework.
- [ ] Add tests for ambiguous same-name records and malformed/negative durations.
- [ ] Run `pytest -q tests/timekeeping/test_parser.py` and commit when green.

### Task 4: Canonical tools and verification

**Files:**
- Create: `rex/timekeeping/tools.py`
- Modify: `rex/tools/registry.py`
- Test: `tests/timekeeping/test_tools.py`
- Test: `tests/test_tools_registry.py`

**Interfaces:**
- `timekeeping_read`: read-only list/status/remaining-time handler.
- `timekeeping_manage`: mutation handler for timer/alarm create and state changes; requires identity.
- Mutation verifier re-reads persisted service state using the returned record ID and expected state before ToolExecutionLifecycle may report `verified`.

- [ ] Write failing registry tests proving both cards are canonical, enabled locally, identity-aware where required, and correctly classified read vs mutation.
- [ ] Write failing lifecycle tests for valid identity, cross-user denial, mutation verification, request dedupe, and malformed commands.
- [ ] Implement tool handlers and independent verifiers using the existing global service/runtime accessors.
- [ ] Run the focused registry/dispatcher/timekeeping matrices and commit when green.

### Task 5: TurnEngine and surface parity

**Files:**
- Modify only if required by tests: `rex/actions/dispatcher.py`
- Test: `tests/timekeeping/test_turn_integration.py`
- Test: existing mobile/chat integration suites as applicable

**Interfaces:**
- Desktop/voice typed turns may use normal pre-LLM tool selection.
- Mobile mutations remain structured post-LLM actions so existing strong-auth action binding is preserved.
- All surfaces resolve to the same `TimekeepingService`; no surface owns duplicate timer state.

- [ ] Add integration tests for typed chat, voice-mode dispatch, and mobile-safe mutation routing.
- [ ] Make the smallest dispatcher changes necessary only if canonical tool selection cannot satisfy those tests.
- [ ] Verify no timer/alarm path bypasses identity, permission, audit, or mutation dedupe stages.
- [ ] Run focused TurnEngine/action/mobile regressions and commit when green.

### Task 6: Capability truth, docs, and release evidence

**Files:**
- Modify: `PRD-production-readiness.md`
- Modify: `CLAUDE.md`
- Modify: `.claude/progress/progress-production-readiness.txt`
- Modify/add capability tests as needed

- [ ] Record exact US-120 behavior and operational boundaries, including restart reconciliation and scheduling tolerance.
- [ ] Mark US-120 acceptance criteria only where executable evidence exists.
- [ ] Run Ruff, Black, mypy on touched Python, the security release gate, pre-commit, and `git diff --check`.
- [ ] Run the authoritative US-120 matrix plus relevant full-regression suite.
- [ ] Open a PR only after fresh verification and leave the GitHub-check criterion pending until that exact head passes remotely.

## Self-review

- Spec coverage: all US-120 acceptance items map to Tasks 1-6; US-121 speaker routing and US-122 routing policies remain intentionally separate workstreams.
- No placeholders: every task names concrete files, interfaces, tests, and completion commands.
- Type consistency: service owns persistence; runtime owns waiting/delivery; parser returns typed commands; tools are the only user-facing execution boundary.
