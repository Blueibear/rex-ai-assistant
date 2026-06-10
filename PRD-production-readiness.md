# PRD: AskRex Assistant Production Readiness and Release Candidate Hardening

> **Ralph execution rule**
> A task means one full User Story, not one checkbox.
> Choose the first User Story whose acceptance criteria contain any unchecked `[ ]` box.
> Complete exactly one User Story per iteration.
> A User Story is only complete when current code, tests, and acceptance criteria prove it.
> When a story is complete, update `PRD-production-readiness.md` and `progress-production-readiness.txt` in the same commit as the implementation.
> Do not commit completed implementation work while leaving that story unchecked in this PRD.
> This PRD is the authoritative task tracker for the production-readiness workstream. `progress-production-readiness.txt` is supporting history only.
> A story is not done until all relevant local validations pass AND all required GitHub checks pass on the PR for that story.

---

## 1. Executive Summary

AskRex Assistant is a local-first, voice-activated AI companion targeting Windows 10/11, macOS, and Linux. The repository contains a Python package (`rex/`), root-level bridge scripts and compatibility shims, a Flask backend (`rex.gui_app`), and an Electron + React desktop GUI under `gui/`. Today the codebase is functional enough for development but it is NOT a release candidate. Production blockers exist across packaged runtime correctness, packaging truth, security audit triage, CI coverage, voice reliability, Home Assistant verification, OpenClaw boundary, and documentation honesty.

This PRD turns AskRex Assistant into a production-ready release candidate by closing every issue listed in the Blocker Inventory (Section 6) through small, dependency-ordered User Stories. Each story is sized for a single Ralph iteration, includes concrete validation commands, requires documentation updates whenever user-facing behavior changes, and requires all relevant GitHub checks to pass before it is marked complete.

The core AskRex principle is preserved end-to-end: Rex must never claim an action succeeded unless the action was verified. Stories that affect Home Assistant control, tool execution, OpenClaw routing, voice pipeline status reporting, and CLI/GUI status surfaces all enforce this rule.

---

## 2. Current State

The following facts were verified directly from the repository at the time this PRD was authored. They are baseline context, not checklist items, and are not action items by themselves.

### 2.1 Renderer raw `/api/` fetches still present
`gui/src/**` still contains direct browser-style fetches that depend on a Flask backend the packaged Electron app does NOT spawn. Confirmed call sites:
- `gui/src/pages/AboutPage.tsx` — `fetch('/api/status')`
- `gui/src/pages/CommandHistoryPage.tsx` — `fetch('/api/history?limit=50', ...)`
- `gui/src/pages/DevicesPage.tsx` — `fetch('/api/devices')` and `fetch(\`/api/devices/${entityId}/command\`, ...)`
- `gui/src/pages/QuickActionsPage.tsx` — `fetch('/api/quick-actions', ...)` (GET and POST), `fetch(\`/api/quick-actions/${id}\`, { method: 'DELETE', ... })`, `fetch(\`/api/quick-actions/${action.id}/run\`, ...)`
- `gui/src/pages/SetupWizardPage.tsx` — `fetch('/api/setup/complete', ...)`
- `gui/src/renderer/src/App.tsx` — `fetch('/api/setup/status')`

`SURFACE-CLASSIFICATION.md` already states: *"All core Electron GUI functionality uses IPC bridge scripts. Renderer fetch('/api/...') calls are dead in packaged mode (file:// protocol)."* This contradicts the renderer's actual behavior, so either the renderer must be migrated to typed IPC or the packaged app must explicitly own a backend lifecycle. This PRD chooses migration to typed IPC as the production direction.

### 2.2 Packaging metadata
- `pyproject.toml` declares `name = "askrex-assistant"`, `version = "0.1.0"`, `requires-python = ">=3.11,<3.12"`, and seven console scripts: `rex`, `rex-config`, `rex-speak-api`, `rex-agent`, `rex-gui`, `rex-tool-server`.
- `setup.py` still declares `py_modules = ["rex_assistant", "rex_speak_api", "llm_client", "memory_utils", "config", "audio_config", "conversation_memory"]`. Several of these names (`rex_assistant`, `memory_utils`, `audio_config`, `conversation_memory`) no longer exist at the repository root, so the wheel currently declares modules it cannot install. This is an active packaging defect.
- `[tool.setuptools]` uses `packages = {find = {include = ["rex*"]}}`. The `bridge/` directory, root bridge wrappers, `config/` examples, and built UI assets are NOT included by default. A `pip install .` produces a Python package that is missing major runtime resources required by both the Electron packaged app and several documented surfaces.

### 2.3 Root-level Python files
27 `.py` files live at the repository root, including 17 `rex_*_bridge.py` wrappers that mirror canonical implementations under `bridge/`. `CLAUDE.md` previously stated "9 active root-level `.py` files" — this is no longer accurate.

### 2.4 Bridge layout
Canonical bridge implementations live under `bridge/` (`bridge/rex_chat_bridge.py`, `bridge/rex_voice_bridge.py`, etc.). The repository root contains thin wrappers with the same filenames. Electron `bridgeResolver.ts` is the single source of truth for which path is resolved in dev vs packaged mode, but the relationship between root wrappers and `bridge/` canonicals is not codified by tests.

### 2.5 Security audit findings
`scripts/security_audit.py` exists (477 lines) and scans for merge markers, placeholder/incomplete code, and exposed secrets. Verified placeholder/stub markers in production-path code:
- `rex/openclaw/workflow_bridge.py` lines 151, 169, 171 — workflow executor registration is an acknowledged stub.
- `rex/replay.py` lines 10, 11, 36, 77, 78, 84, 89, 118-144 — replay is explicitly a stub that returns placeholder results.
- `rex/skills/trainer.py` line 127 — `# TODO: implement {name}`.

### 2.6 Deprecated API usage
Verified call sites:
- `rex/assistant.py` line 404: `datetime.utcnow()`
- `rex/assistant.py` line 840: `datetime.utcnow()`
- `rex/geolocation.py` line 41: `asyncio.get_event_loop()`
- `rex/openclaw/tool_executor.py` line 560: `asyncio.get_event_loop()`
- `rex/tts_voices.py` lines 192, 244: `asyncio.get_event_loop()`

### 2.7 Large files
- `rex/cli.py` — 5,326 lines.
- `rex/voice_loop.py` — 3,232 lines.
- `rex/gui_app.py` — 1,536 lines.
- `gui/src/main/index.ts` — 1,614 lines.
- `gui/src/pages/SettingsPage.tsx` — 5,360 lines.

### 2.8 CI coverage
`.github/workflows/ci.yml` runs `ruff check`, `black --check --diff rex/ tests/ bridge/ *.py`, `mypy rex --ignore-missing-imports`, pytest with coverage and integration markers, `pip-audit`, pre-commit, and `detect-secrets`. `electron-smoke.yml` runs an Electron package smoke test on tag pushes and PRs touching `gui/` or `bridge/`. There is no wheel contents smoke test, no security_audit.py CI check, no entry-point import/help smoke test, no `scripts/` black coverage, no skip-budget enforcement, and no raw-`/api/` fetch guard.

### 2.9 Docker
`Dockerfile` HEALTHCHECK is `python -c "import sys; sys.exit(0)"` — a placeholder that always succeeds.

### 2.10 Skipped tests
`grep -rn "@pytest.mark.skip" tests/` yields 125 hits. Many are legitimate `skipif(<env or dep missing>)` guards, but the set is not classified, tracked, or budgeted.

---

## 3. Production Target

When this PRD is complete, AskRex Assistant ships as:

- **Primary app artifact:** Packaged Electron desktop app under `gui/`. The Electron main process spawns the Python bridge scripts directly via stdin/stdout JSON, with no Flask backend required at runtime.
- **Primary non-GUI surface:** `rex` CLI (`python -m rex` or the `rex` console script).
- **Primary voice surface:** `rex_loop.py` plus the canonical voice loop (`rex.voice_loop`). Hold-to-Talk is the supported default voice mode for the release candidate; wake word remains beta until the reliability tests added by this PRD pass on Windows 11.
- **Primary supported platforms:** Windows 11 (primary), Windows 10, macOS, Linux (best-effort).
- **Documented experimental surfaces:** OpenClaw integration, autonomy, browser automation, computer control, phone/SMS via Twilio, and `rex-agent` remote PC control. Each is disabled by default and clearly labeled as experimental in README, GUI settings, and `SURFACE-CLASSIFICATION.md`.
- **Install paths:**
  - End users: packaged Electron installer.
  - Developers/operators: `pip install .` for the Python library and console scripts; `requirements-gpu-cu124.txt` for the Windows GPU path; documented dev setup in `INSTALL.md`.
- **CI gate:** Every PR runs the full set of checks listed in Section 10, and no story is marked complete until those checks pass.

---

## 4. Non-Goals

This PRD does NOT include:

- New product features beyond what is required to make existing surfaces production-honest.
- A redesign of the Electron GUI's information architecture.
- A new LLM provider, new STT engine, or new TTS engine.
- A move off Python 3.11 (3.12+ is explicitly deferred).
- Bringing OpenClaw to production-critical status. OpenClaw remains optional/experimental.
- Bringing autonomy, browser automation, computer control, or remote PC control to production-critical status.
- Replacing Flask, Electron, or React.
- A migration away from Whisper, openWakeWord, or Coqui XTTS.
- Adding paid telemetry, cloud sync, or remote configuration.
- Renaming the project, package, CLI, or repository.
- A full UI accessibility audit (tracked separately).

---

## 5. Release Principles

These principles bind every story in this PRD. They are enforced by review and by the global acceptance criteria in Section 9.

1. **Verification over claims.** Rex never reports an action succeeded unless the action was verified. Speech, text, log lines, GUI status messages, and tool return values must distinguish *attempted*, *completed*, *verified*, and *failed*.
2. **Docs change with code.** Any story that changes user-facing behavior, install flow, GUI behavior, commands, dependencies, file structure, configuration, integrations, or capability claims MUST update README, INSTALL, RUNNING, `docs/UI_SURFACES.md`, `SURFACE-CLASSIFICATION.md`, `INTEGRATIONS_STATUS.md`, and `CLAUDE.md` as relevant in the same story.
3. **Least privilege defaults.** Network-bound surfaces bind to localhost, require authentication, and rate-limit by default.
4. **Optional integrations degrade gracefully.** Email, calendar, SMS, MQTT, Home Assistant, web search, OpenClaw, and autonomy must fail closed and produce a clear, user-visible error rather than silent success when not configured.
5. **CI matches the shipped product.** A green CI run must mean the packaged Electron app, the wheel, the bridge scripts, the console scripts, and the documented commands all work.
6. **One story per Ralph iteration.** Stories are sized so that a fresh agent instance can read context, make the change, add tests, run validation, update docs, and commit in one focused loop.
7. **No `[x]` without proof.** A story is not checked off until code, tests, docs, validation commands, and required GitHub checks all confirm completion on the PR that delivers the story.
8. **Behavior-preserving refactors only.** Large-file decompositions in Phase 9 must not change observable behavior. They land with tests, not without.

---

## 6. Blocker Inventory

| ID | Blocker | Summary | Workstream | Priority |
|----|---------|---------|------------|----------|
| A | Packaged Electron runtime correctness | Renderer `/api/...` fetches are dead in packaged mode; no enforcement against regression. | Electron | P0 |
| B | Wheel/package install truth | `pip install .` does not produce a runnable app; declared `py_modules` reference files that no longer exist. | Packaging | P0 |
| C | `setup.py` and metadata cleanup | Stale `py_modules`, undocumented root shims, unclear console-script contract. | Packaging | P0 |
| D | Bridge layout and root file truth | Root wrappers vs `bridge/` canonicals not codified; docs claim wrong root file count. | Packaging | P0 |
| E | Security audit triage | `security_audit.py` reports actionable findings; auth gates and confirmation gates missing on several routes/tools. | Security | P0 |
| F | CI must match the shipped product | CI omits wheel smoke, security_audit, entry-point import/help smoke, scripts/ formatting, skip budget, and `/api/` guard. | CI | P0 |
| G | Skipped tests and retired surfaces | 125 skips not classified; no skip budget; tests for retired surfaces still present. | Tests | P1 |
| H | Docker healthcheck truth | Healthcheck is a no-op; Docker's support tier is undocumented. | Packaging | P1 |
| I | Runtime truth and docs consistency | README, INSTALL, RUNNING, `docs/UI_SURFACES.md`, `SURFACE-CLASSIFICATION.md`, `INTEGRATIONS_STATUS.md`, and `CLAUDE.md` disagree about what is shippable. | Docs | P0 |
| J | Voice reliability and production voice path | Wake word reliability not measured; Hold-to-Talk not defined as production path; voice pipeline lacks structured logs and latency budgets. | Voice | P0 |
| K | Home Assistant control and verification | Risky domains have no confirmation gate; post-control verification not enforced; response language mixes attempted/completed. | Home Assistant | P0 |
| L | OpenClaw production boundary | OpenClaw is on the production path despite incomplete dynamic plugin, permission, and verification work. | OpenClaw | P0 |
| M | Deprecated APIs and technical debt | `datetime.utcnow()` and `asyncio.get_event_loop()` calls remain; no regression tests. | Tech Debt | P2 |
| N | Giant file decomposition | `rex/cli.py` (5,326), `rex/voice_loop.py` (3,232), `rex/gui_app.py` (1,536), `gui/src/main/index.ts` (1,614), `gui/src/pages/SettingsPage.tsx` (5,360). Refactor only after P0 blockers are closed. | Tech Debt | P2 |

---

## 7. Workstreams / Phases

Stories execute in this phase order. Within a phase, stories execute in numeric order. Later phases assume earlier phases are complete.

- **Phase 0 — Baseline & Discovery** (US-001 to US-002)
- **Phase 1 — Electron Renderer to IPC** (US-003 to US-012)
- **Phase 2 — Packaging Truth** (US-013 to US-019)
- **Phase 3 — Security Triage and Gates** (US-020 to US-029)
- **Phase 4 — CI Hardening** (US-030 to US-037)
- **Phase 5 — Skipped Tests and Retired Surfaces** (US-038 to US-040)
- **Phase 6 — Docker Honesty** (US-041)
- **Phase 7 — Voice Reliability and Production Voice Path** (US-042 to US-046)
- **Phase 8 — Home Assistant Verification** (US-047 to US-049)
- **Phase 9 — OpenClaw Production Boundary** (US-050 to US-052)
- **Phase 10 — Documentation Truth Pass** (US-053 to US-055)
- **Phase 11 — Deprecated APIs** (US-056 to US-058)
- **Phase 12 — Giant File Decomposition (P2)** (US-059 to US-063)

---

## 8. User Stories

### US-001: Generate the security_audit triage inventory

**Priority:** P0
**Workstream:** Security / Docs
**Description:** As a maintainer, I want a triaged inventory of every finding produced by `scripts/security_audit.py` so subsequent stories can fix or document each one.

**Why it matters:** Without a current inventory, later security stories cannot tell what is a production blocker, what is a documented dev-only artifact, and what is a false positive.

**Files/areas likely involved:**
- `scripts/security_audit.py`
- `docs/security/AUDIT-INVENTORY.md` (new)

**Implementation notes:** Run the audit on a clean checkout. Capture stdout verbatim. Classify every finding as `production-blocker`, `dev-only-documented`, or `false-positive`. Do not fix findings in this story.

**Acceptance Criteria:**
- [ ] `python scripts/security_audit.py` is run and its full output is committed under `docs/security/AUDIT-INVENTORY.md`.
- [ ] Each finding has a row with file, line, marker, classification, and the User Story ID that will resolve it (or "no action — documented" with rationale).
- [ ] `docs/security/AUDIT-INVENTORY.md` is linked from `SECURITY.md` and `README.md` under a "Security baseline" section.
- [ ] `python scripts/security_audit.py` exits with its current status (no behavior change in this story).
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/security_audit.py | tee /tmp/security_audit_baseline.txt
git diff --quiet docs/security/AUDIT-INVENTORY.md || echo "inventory updated"
```

**Risk notes:** None — read-only inventory.

---

### US-002: Generate the skipped-test inventory

**Priority:** P0
**Workstream:** Tests / Docs
**Description:** As a maintainer, I want a classified inventory of every skipped or `skipif` test so later stories can remove retired tests, replace important skips, and set a skip budget.

**Why it matters:** 125 skip markers exist today. Without classification, a skip budget cannot be enforced and trust in the test suite stays low.

**Files/areas likely involved:**
- `tests/`
- `docs/testing/SKIPPED-TESTS-INVENTORY.md` (new)

**Implementation notes:** Use `pytest --collect-only -q` and a focused `grep` to enumerate every skip site. Classify each as `optional-dep-skip`, `platform-skip`, `retired-surface-skip`, or `temporary-bug-skip` and record the file, line, skip reason, and follow-up story ID if any.

**Acceptance Criteria:**
- [ ] `docs/testing/SKIPPED-TESTS-INVENTORY.md` lists every `@pytest.mark.skip`, `@pytest.mark.skipif`, and inline `pytest.skip(...)` call.
- [ ] Each row records: file, line, skip reason text, classification, and follow-up story (or "permanent" with rationale).
- [ ] Inventory is linked from `docs/TESTING_AND_QUALITY.md` if that file exists, otherwise from `README.md`'s testing section.
- [ ] `pytest --collect-only -q` exits 0.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest --collect-only -q
grep -rn "pytest.mark.skip\|pytest.skip\b" tests | wc -l
```

**Risk notes:** None — read-only inventory.

---

### US-003: Add CI guard against raw renderer `/api/` fetches

**Priority:** P0
**Workstream:** Electron / CI
**Description:** As a maintainer, I want CI to fail when a renderer file introduces a raw `fetch('/api/...')` call so future regressions are caught immediately.

**Why it matters:** Once US-004 through US-011 migrate existing calls to IPC, regressions are easy. Catching them at CI prevents the packaged app from silently breaking again.

**Files/areas likely involved:**
- `scripts/check_no_renderer_api_fetch.py` (new)
- `gui/scripts/check_no_renderer_api_fetch.cjs` (new, optional Node implementation)
- `.github/workflows/ci.yml`
- `gui/src/ALLOWED_API_FETCHES.txt` (new, empty allowlist)

**Implementation notes:** Write a script that greps `gui/src/**/*.{ts,tsx,js,jsx}` for raw `fetch('/api`, `fetch("/api`, and `fetch(\`/api` patterns. The script exits non-zero if any match is not listed in `gui/src/ALLOWED_API_FETCHES.txt` with `file:line` and a justification comment. The allowlist starts empty.

**Acceptance Criteria:**
- [ ] Script exists and exits 0 on a checkout that has zero raw `/api/` fetches (post-migration), and exits non-zero when a synthetic raw `/api/` fetch is introduced (covered by a unit test).
- [ ] Allowlist file format is documented at the top of the file.
- [ ] CI job `gui-no-raw-api` runs the script on every PR.
- [ ] Story does NOT fix existing renderer call sites — those are owned by US-004 through US-010.
- [ ] The script's allowlist permits all current renderer `/api/` call sites as a temporary baseline; each later migration story removes its line from the allowlist when complete.
- [ ] `README.md` and `docs/UI_SURFACES.md` reference the guard and the allowlist policy.
- [ ] `pytest tests/test_check_no_renderer_api_fetch.py -q` passes.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/check_no_renderer_api_fetch.py
pytest tests/test_check_no_renderer_api_fetch.py -q
```

**Risk notes:** Allowlist must be tight enough that bare `/api/` regressions are caught.

---

### US-004: Migrate `AboutPage` `/api/status` to typed IPC

**Priority:** P0
**Workstream:** Electron
**Description:** As an end user of the packaged Electron app, I want the About page to load app status via IPC so it works without a Flask backend.

**Files/areas likely involved:**
- `gui/src/pages/AboutPage.tsx`
- `gui/src/preload/index.ts`
- `gui/src/preload/index.d.ts`
- `gui/src/main/index.ts`
- `gui/src/main/handlers/usage.ts` or new `handlers/status.ts`
- `gui/src/types/ipc.ts`

**Implementation notes:** Add a `getAppStatus(): Promise<AppStatus>` IPC method. Main-process handler reads from the same source the Flask `/api/status` route used. Update the TypeScript interface. Remove the raw fetch. Remove the matching allowlist line from `gui/src/ALLOWED_API_FETCHES.txt`.

**Acceptance Criteria:**
- [ ] `gui/src/pages/AboutPage.tsx` contains no `fetch('/api/...')` call.
- [ ] Preload exposes `window.api.getAppStatus()`.
- [ ] Main-process handler returns the same shape the renderer expected from the old route.
- [ ] `gui/src/ALLOWED_API_FETCHES.txt` no longer lists `AboutPage.tsx`.
- [ ] `cd gui && npm run typecheck` passes.
- [ ] `cd gui && npm run build` passes.
- [ ] Manual: launching the packaged Electron app shows About page status without errors. Verification recorded in PR description.
- [ ] `README.md` (or `docs/UI_SURFACES.md`) notes that About status is IPC-backed.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
python scripts/check_no_renderer_api_fetch.py
```

**Risk notes:** Keep the IPC return type strict so future shape changes are caught at compile time.

---

### US-005: Migrate `CommandHistoryPage` `/api/history` to typed IPC

**Priority:** P0
**Workstream:** Electron
**Description:** As an end user, I want command history to load via IPC so it works in the packaged app.

**Files/areas likely involved:**
- `gui/src/pages/CommandHistoryPage.tsx`
- `gui/src/preload/index.ts`, `index.d.ts`
- `gui/src/main/index.ts`
- `gui/src/main/handlers/usage.ts` (or new `handlers/history.ts`)
- `gui/src/types/ipc.ts`

**Implementation notes:** Add `getCommandHistory(limit: number): Promise<CommandHistoryEntry[]>`. Preserve the `limit=50` default. Remove the raw fetch and remove the allowlist line.

**Acceptance Criteria:**
- [ ] No raw `/api/...` fetch remains in `CommandHistoryPage.tsx`.
- [ ] IPC handler returns the same shape the renderer expects.
- [ ] `gui/src/ALLOWED_API_FETCHES.txt` no longer lists `CommandHistoryPage.tsx`.
- [ ] `cd gui && npm run typecheck` passes.
- [ ] `cd gui && npm run build` passes.
- [ ] Manual: command history renders in the packaged app.
- [ ] Docs updated if user-facing behavior or wording changes.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
python scripts/check_no_renderer_api_fetch.py
```

**Risk notes:** Auth header handling must be preserved (`authHeaders()` equivalent must be encoded into the main-process handler).

---

### US-006: Migrate `DevicesPage` device-list `/api/devices` to typed IPC

**Priority:** P0
**Workstream:** Electron / Home Assistant
**Description:** As an end user, I want the Devices page to load Home Assistant entities via IPC.

**Files/areas likely involved:**
- `gui/src/pages/DevicesPage.tsx`
- `gui/src/preload/`, `gui/src/main/handlers/` (new `handlers/devices.ts`)
- `gui/src/types/ipc.ts`

**Acceptance Criteria:**
- [ ] `fetch('/api/devices')` removed from `DevicesPage.tsx`.
- [ ] IPC handler reads HA entities through the existing bridge resolver path.
- [ ] `gui/src/ALLOWED_API_FETCHES.txt` no longer lists this call.
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Manual: device list renders in packaged app.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
python scripts/check_no_renderer_api_fetch.py
```

**Risk notes:** Keep the API shape stable so US-007 can build on it.

---

### US-007: Migrate `DevicesPage` per-entity command `/api/devices/{id}/command` to typed IPC

**Priority:** P0
**Workstream:** Electron / Home Assistant
**Description:** As an end user, I want device control commands to dispatch via IPC.

**Files/areas likely involved:**
- `gui/src/pages/DevicesPage.tsx`
- `gui/src/main/handlers/devices.ts`
- `gui/src/preload/`
- `gui/src/types/ipc.ts`

**Acceptance Criteria:**
- [ ] `fetch(\`/api/devices/${entityId}/command\`, ...)` removed.
- [ ] IPC method `sendDeviceCommand(entityId, command, payload)` exists, typed.
- [ ] Allowlist line removed.
- [ ] Handler returns a discriminated `{ status: 'attempted' | 'completed' | 'verified' | 'failed', detail?: string }` shape (foundation for US-049).
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Manual: a device toggle in the packaged app reports a verified status when HA confirms state.
- [ ] `docs/home_assistant.md` notes the IPC method.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
python scripts/check_no_renderer_api_fetch.py
```

**Risk notes:** The verification gate is added by US-048; this story only establishes the response shape.

---

### US-008: Migrate `QuickActionsPage` list/create `/api/quick-actions` to typed IPC

**Priority:** P0
**Workstream:** Electron
**Description:** As an end user, I want to list and create quick actions via IPC.

**Files/areas likely involved:**
- `gui/src/pages/QuickActionsPage.tsx`
- `gui/src/preload/`, `gui/src/main/handlers/` (new `handlers/quickActions.ts`)
- `gui/src/types/ipc.ts`

**Acceptance Criteria:**
- [ ] Both `/api/quick-actions` calls (GET list, POST create) removed.
- [ ] IPC methods `listQuickActions()` and `createQuickAction(...)` exist.
- [ ] Allowlist updated.
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Manual: page renders the list and accepts a new action in the packaged app.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
python scripts/check_no_renderer_api_fetch.py
```

---

### US-009: Migrate `QuickActionsPage` delete/run `/api/quick-actions/{id}` to typed IPC

**Priority:** P0
**Workstream:** Electron
**Description:** As an end user, I want to delete and run quick actions via IPC.

**Files/areas likely involved:**
- `gui/src/pages/QuickActionsPage.tsx`
- `gui/src/main/handlers/quickActions.ts`
- `gui/src/preload/`, `gui/src/types/ipc.ts`

**Acceptance Criteria:**
- [ ] DELETE and `/run` raw fetches removed.
- [ ] IPC methods `deleteQuickAction(id)` and `runQuickAction(id)` exist.
- [ ] Allowlist updated.
- [ ] `runQuickAction` returns `{ status: 'attempted' | 'completed' | 'verified' | 'failed', detail?: string }`.
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Manual: deleting and running a quick action works in packaged app.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
python scripts/check_no_renderer_api_fetch.py
```

---

### US-010: Migrate `SetupWizardPage` and `App.tsx` `/api/setup/*` to typed IPC

**Priority:** P0
**Workstream:** Electron
**Description:** As a new user, I want the setup wizard to read and write setup state via IPC.

**Files/areas likely involved:**
- `gui/src/pages/SetupWizardPage.tsx`
- `gui/src/renderer/src/App.tsx`
- `gui/src/preload/`, `gui/src/main/handlers/` (new `handlers/setup.ts`)
- `gui/src/types/ipc.ts`

**Acceptance Criteria:**
- [ ] `/api/setup/status` and `/api/setup/complete` raw fetches removed.
- [ ] IPC methods `getSetupStatus()` and `completeSetup(payload)` exist, typed.
- [ ] Allowlist no longer lists `SetupWizardPage.tsx` or `App.tsx`.
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Manual: first-run wizard completes end-to-end in the packaged app with no network calls to `localhost`.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
python scripts/check_no_renderer_api_fetch.py
```

**Risk notes:** This is the most user-visible path. Manual sanity check is required.

---

### US-011: Drain the `/api/` allowlist and lock the guard

**Priority:** P0
**Workstream:** Electron / CI
**Description:** As a maintainer, I want the allowlist to be empty so the guard fails on any future raw `/api/` fetch.

**Files/areas likely involved:**
- `gui/src/ALLOWED_API_FETCHES.txt`
- `scripts/check_no_renderer_api_fetch.py`
- `gui/src/main/handlers/` (any leftover call sites)

**Acceptance Criteria:**
- [ ] `gui/src/ALLOWED_API_FETCHES.txt` contains only header comments — no allowed entries.
- [ ] `python scripts/check_no_renderer_api_fetch.py` exits 0 on a clean repo.
- [ ] `grep -rn "fetch('/api\\|fetch(\"/api\\|fetch(\`/api" gui/src` returns no matches.
- [ ] `README.md` documents the packaged Electron runtime as IPC-only and explicitly states a Flask backend is NOT required at runtime for end users.
- [ ] `SURFACE-CLASSIFICATION.md` is verified consistent with this state.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/check_no_renderer_api_fetch.py
grep -rn "fetch('/api\|fetch(\"/api\|fetch(\`/api" gui/src || echo "clean"
cd gui && npm run typecheck && npm run build
```

**Risk notes:** If any call site was missed in US-004 through US-010, finish it here.

---

### US-012: Packaged Electron smoke test requires no Flask backend

**Priority:** P0
**Workstream:** Electron / CI
**Description:** As a maintainer, I want the Electron smoke test to launch the packaged app and confirm the renderer renders without `rex-gui` ever being started.

**Files/areas likely involved:**
- `tests/smoke/test_electron_package.sh`
- `.github/workflows/electron-smoke.yml`

**Implementation notes:** Extend the smoke script so a step explicitly verifies no `rex-gui` process is started and no listener binds the Flask port during the smoke window. Capture the renderer console for any failed `/api/` fetch and fail if any appear.

**Acceptance Criteria:**
- [ ] Smoke test asserts no Python process bound `127.0.0.1:5000` (or equivalent Flask port) during the test.
- [ ] Smoke test asserts no renderer console error matches `/api/`.
- [ ] Smoke test runs in CI on every PR (not only on tag pushes), at least for `gui/`, `bridge/`, and renderer `/api/` allowlist changes.
- [ ] `bash tests/smoke/test_electron_package.sh` exits 0 on a clean checkout.
- [ ] `README.md` documents that the packaged app does not require running `rex-gui`.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
bash tests/smoke/test_electron_package.sh
```

**Risk notes:** Flaky GUI environments in CI. Use the existing xvfb wrapper and the documented `REQUIRE_ELECTRON_SIGNAL` knob.

---

### US-013: Decide and document the supported pip/wheel install scope

**Priority:** P0
**Workstream:** Packaging / Docs
**Description:** As an installer, I want a clear statement of what `pip install askrex-assistant` provides.

**Files/areas likely involved:**
- `README.md`
- `INSTALL.md`
- `pyproject.toml`
- `SURFACE-CLASSIFICATION.md`

**Implementation notes:** The decision for the release candidate: `pip install .` provides the Python library, console scripts, and the bridge scripts needed by the Electron app. It does NOT provide the Electron desktop installer. Document this explicitly. Update `pyproject.toml` `description` and `classifiers` to match.

**Acceptance Criteria:**
- [ ] `README.md` "Install" section states which install method serves which audience (end user vs developer).
- [ ] `INSTALL.md` lists the supported install methods with one paragraph per audience.
- [ ] `pyproject.toml` `description` reflects the package scope.
- [ ] `SURFACE-CLASSIFICATION.md` lists pip/wheel as `developer-only` with rationale.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python -c "import askrex_assistant" 2>/dev/null || true
python -m pip install --dry-run . >/dev/null
```

**Risk notes:** Avoid promising an end-user pip path that does not exist.

---

### US-014: Repair `setup.py` `py_modules` to match files on disk

**Priority:** P0
**Workstream:** Packaging
**Description:** As an installer, I want `pip install .` to not declare modules that do not exist.

**Files/areas likely involved:**
- `setup.py`
- `pyproject.toml`
- Root `.py` files referenced from `py_modules`

**Implementation notes:** Inspect each entry in `py_modules`. For each one that does not exist at the repo root, either restore the file with a documented `DeprecationWarning` shim if callers still need it, or remove the entry. Document the result.

**Acceptance Criteria:**
- [ ] Every entry in `setup.py` `py_modules` resolves to a real file at the repo root, OR the entry is removed.
- [ ] A comment block in `setup.py` documents why each surviving entry exists.
- [ ] `python -m build` produces a wheel without warnings about missing modules.
- [ ] `pip install dist/askrex_assistant-*.whl --force-reinstall` succeeds in a fresh venv.
- [ ] `README.md` and `INSTALL.md` are updated if any root file's classification changes.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
rm -rf dist
python -m build
pip install --force-reinstall dist/askrex_assistant-*.whl
python -c "import setup; print('ok')" 2>/dev/null || true
```

**Risk notes:** Removing a shim that callers still import will break those callers. Add a test under US-019.

---

### US-015: Add a wheel contents smoke test

**Priority:** P0
**Workstream:** Packaging / CI
**Description:** As a maintainer, I want CI to fail when the wheel is missing required runtime resources.

**Files/areas likely involved:**
- `scripts/check_wheel_contents.py` (new)
- `.github/workflows/ci.yml`
- `tests/test_wheel_contents.py` (new)

**Implementation notes:** Build the wheel, list its files (`zipfile`), and assert presence of: every console-script module, every required root bridge wrapper, the `bridge/` canonical scripts, `config/rex_config.example.json`, `py.typed`, and any other assets identified in US-013.

**Acceptance Criteria:**
- [ ] Script builds `dist/askrex_assistant-*.whl` and asserts the documented contents.
- [ ] CI runs the script.
- [ ] `pytest tests/test_wheel_contents.py -q` passes.
- [ ] If a required file is missing, the test names the file and the install audience that needs it.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/check_wheel_contents.py
pytest tests/test_wheel_contents.py -q
```

**Risk notes:** The required-file list is the contract. Keep it short and accurate.

---

### US-016: Ensure required runtime resources are packaged in the wheel

**Priority:** P0
**Workstream:** Packaging
**Description:** As an installer of the developer wheel, I want the bridge scripts, config example, and required assets to be present.

**Files/areas likely involved:**
- `pyproject.toml` (`[tool.setuptools.package-data]`, `[tool.setuptools.packages.find]`)
- `MANIFEST.in` (new if needed)
- `bridge/`, `config/`, `assets/`

**Implementation notes:** Configure `setuptools` so the wheel includes the resources US-015 asserts. Where appropriate, move resources into the `rex` package and update consumers, OR explicitly include top-level data via `MANIFEST.in` + `include_package_data`.

**Acceptance Criteria:**
- [ ] `scripts/check_wheel_contents.py` passes after this story.
- [ ] No new top-level package is created.
- [ ] `README.md` and `INSTALL.md` describe what `pip install` ships.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python -m build
python scripts/check_wheel_contents.py
```

**Risk notes:** Including `bridge/` files in a wheel without renaming may collide with site-packages layout; prefer placing them under `rex/bridges/` or using `include_package_data` with `MANIFEST.in`.

---

### US-017: Audit and classify every root-level `.py` file

**Priority:** P0
**Workstream:** Packaging / Docs
**Description:** As a maintainer, I want every root-level Python file classified so docs and packaging agree on what is supported.

**Files/areas likely involved:**
- `*.py` at repo root
- `SURFACE-CLASSIFICATION.md`
- `CLAUDE.md`
- `docs/UI_SURFACES.md`
- `README.md`

**Implementation notes:** Classify each root `.py` as `production-entrypoint`, `compatibility-wrapper`, `developer-utility`, `test-helper`, or `archive-candidate`. Move archive candidates under `archived/` with a rationale. Update `CLAUDE.md`'s "Active root-level `.py` files" count to match.

**Acceptance Criteria:**
- [ ] `SURFACE-CLASSIFICATION.md` lists every root `.py` file with its classification.
- [ ] `CLAUDE.md`'s root-file count and list match reality.
- [ ] Files moved to `archived/` retain history (use `git mv`).
- [ ] No production import path is broken (covered by US-018's bridge-resolver tests and US-019's entry-point smoke).
- [ ] `python scripts/check_imports.py` or equivalent passes.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python -m compileall -q $(ls *.py)
pytest -q
```

**Risk notes:** Moving a file that the Electron `bridgeResolver.ts` references will break the packaged app. Verify resolver paths first.

---

### US-018: Bridge path resolution tests for dev and packaged mode

**Priority:** P0
**Workstream:** Packaging
**Description:** As a maintainer, I want unit tests that confirm bridge resolution returns valid file paths in both dev and packaged layouts.

**Files/areas likely involved:**
- `gui/src/main/bridgeResolver.ts`
- `tests/test_bridge_resolution.py` (new) or `gui/tests/bridgeResolver.test.ts` (new)
- `bridge/`, root `rex_*_bridge.py`

**Implementation notes:** A Python test enumerates the bridge scripts referenced by `bridgeResolver.ts` (parse the TypeScript file deterministically), then asserts each path exists in the source tree. A TypeScript/Vitest test asserts that, given a faked `app.getAppPath()`, the resolver returns paths under `resources/bridge/` for the packaged layout and under `bridge/` (or repo root) for dev.

**Acceptance Criteria:**
- [ ] Python test asserts every bridge script referenced by `bridgeResolver.ts` exists in the source tree.
- [ ] TypeScript test asserts resolver behavior in both dev and packaged-path modes.
- [ ] `pytest tests/test_bridge_resolution.py -q` passes.
- [ ] `cd gui && npm test` passes (if vitest is wired) OR `cd gui && npm run typecheck && npm run build` passes (acceptable interim).
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_bridge_resolution.py -q
cd gui && npm run typecheck && npm run build
```

**Risk notes:** Resolver behavior is the contract between the wheel/Electron packaging and the renderer; lock it down with tests.

---

### US-019: Console script import/help smoke tests

**Priority:** P0
**Workstream:** Packaging / CI
**Description:** As a maintainer, I want each declared console script to import and run `--help` without crashing.

**Files/areas likely involved:**
- `tests/test_console_scripts_smoke.py` (new)
- `.github/workflows/ci.yml`

**Implementation notes:** Parametrize a pytest over `rex`, `rex-config`, `rex-speak-api`, `rex-agent`, `rex-gui`, `rex-tool-server`. For each, run `<script> --help` (or `python -c "from <module> import <fn>"` where `--help` is not safe) and assert exit code 0 and non-empty stdout.

**Acceptance Criteria:**
- [ ] One test per declared console script.
- [ ] All tests pass on a clean install of the wheel.
- [ ] CI runs these tests after `pip install -e .`.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pip install -e .
pytest tests/test_console_scripts_smoke.py -q
```

**Risk notes:** `rex-gui` may import optional GPU/ML deps; the smoke must use a lazy/short-path startup.

---

### US-020: Fix or formally stub `rex/replay.py` so production callers do not silently misreport

**Priority:** P0
**Workstream:** Security / Verification
**Description:** As a user, I want replay results to be honest. The current stub returns a placeholder result while looking like a real return value.

**Files/areas likely involved:**
- `rex/replay.py`
- Callers of `replay_tool_call` or equivalent
- `tests/test_replay.py` (new or updated)

**Implementation notes:** Either (a) remove `rex/replay.py` from the production path and emit a clear "not implemented" error when called, or (b) implement minimal honest replay that reruns the tool through the existing tool executor. Either way, the function must not return a `status: "stub"` payload to callers that present it as a result.

**Acceptance Criteria:**
- [ ] Calling `replay_tool_call(...)` either returns a real, verified result, OR raises `NotImplementedError("replay is not available in this build")` with no placeholder dict.
- [ ] `rex/replay.py` no longer contains the strings `"placeholder"`, `"status": "stub"`, or `# TODO: implement` on any execution path reachable from a console script or IPC handler.
- [ ] A test asserts that a calling code path either gets a real result or an exception — never a placeholder dict.
- [ ] `README.md` or `docs/audit.md` documents the replay capability state honestly.
- [ ] `SECURITY.md` notes the change if applicable.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_replay.py -q
python scripts/security_audit.py
```

**Risk notes:** Preserve the audit-log read path (`get_replayable_calls`) — that side is read-only and OK.

---

### US-021: Fix or formally stub `rex/openclaw/workflow_bridge.py` workflow executor registration

**Priority:** P0
**Workstream:** Security / OpenClaw
**Description:** As a user, I want OpenClaw workflow registration to either succeed or fail with a clear error.

**Files/areas likely involved:**
- `rex/openclaw/workflow_bridge.py`
- `tests/test_openclaw_workflow_bridge.py` (new or updated)

**Implementation notes:** The current code logs `"OpenClaw workflow executor registration stub — update once API is confirmed (PRD §8.6)"` and continues. Update the behavior: when `use_openclaw_tools` is False (default per US-050), registration is a no-op and the log is `info`. When `use_openclaw_tools` is True and the upstream API is not implemented, registration must raise `OpenClawConfigError("workflow executor not available")` so downstream code fails closed rather than silently bypassing OpenClaw.

**Acceptance Criteria:**
- [ ] With `use_openclaw_tools=False`, registration is a no-op; no `# TODO` or `stub` log text remains on the reachable path.
- [ ] With `use_openclaw_tools=True`, registration raises a clear error if the upstream API is not present.
- [ ] A test covers both branches.
- [ ] `python scripts/security_audit.py` no longer flags this file (or the inventory in US-001 marks it as `dev-only-documented` with a follow-up story).
- [ ] `docs/openclaw-migration-status.md` is updated.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_openclaw_workflow_bridge.py -q
python scripts/security_audit.py
```

**Risk notes:** Do not silently disable OpenClaw — that would mask configuration mistakes.

---

### US-022: Remove or guard the `rex/skills/trainer.py` placeholder

**Priority:** P0
**Workstream:** Security
**Description:** As a user, I want the skills trainer to either work or refuse cleanly.

**Files/areas likely involved:**
- `rex/skills/trainer.py`
- Callers of the trainer
- `tests/test_skills_trainer.py` (new or updated)

**Implementation notes:** The `# TODO: implement {name}` marker at line 127 must be removed. If the trainer is not on the production path, gate it behind an explicit `developer-only` flag and document it in `SURFACE-CLASSIFICATION.md`. If it is on the production path, implement the minimum honest behavior.

**Acceptance Criteria:**
- [ ] `grep -n "TODO: implement" rex/skills/trainer.py` returns nothing on reachable paths.
- [ ] A test confirms the chosen behavior (works honestly, OR raises a clear `NotImplementedError` behind a flag).
- [ ] `SURFACE-CLASSIFICATION.md` is updated.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_skills_trainer.py -q
python scripts/security_audit.py
```

---

### US-023: Authenticate the logs download endpoint

**Priority:** P0
**Workstream:** Security
**Description:** As an operator, I want `/api/logs/*` endpoints to reject unauthenticated requests.

**Files/areas likely involved:**
- `rex/gui_app.py` (or the route file)
- `tests/test_logs_auth.py` (new or updated)

**Acceptance Criteria:**
- [ ] Unauthenticated GET on `/api/logs/download` returns HTTP 401.
- [ ] Authenticated GET with a valid token still works.
- [ ] A negative test (`pytest tests/test_logs_auth.py::test_unauth_logs_returns_401 -q`) asserts the 401.
- [ ] Log output redacts home-directory paths (`/Users/<name>`, `C:\Users\<name>`) before being sent in any response.
- [ ] `docs/configuration.md` and `README.md` document the auth requirement.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_logs_auth.py -q
```

**Risk notes:** Operator dev workflows may rely on unauthenticated `localhost` access. Document the change in `INSTALL.md` and `RUNNING.md`.

---

### US-024: Authenticate Home Assistant test/save/control endpoints

**Priority:** P0
**Workstream:** Security / Home Assistant
**Description:** As an operator, I want HA admin endpoints to require authentication.

**Files/areas likely involved:**
- `rex/gui_app.py` route handlers for HA
- `gui/src/main/handlers/devices.ts`
- `tests/test_ha_auth.py` (new or updated)

**Acceptance Criteria:**
- [ ] HA test, save, and control routes return 401 without a valid token.
- [ ] IPC equivalents enforce the same auth via the main-process token store.
- [ ] Negative tests cover each route.
- [ ] `docs/home_assistant.md` documents the auth requirement.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_ha_auth.py -q
```

---

### US-025: Confirmation gates for destructive tools

**Priority:** P0
**Workstream:** Security / Verification
**Description:** As a user, I want destructive tool calls to require explicit confirmation before execution.

**Files/areas likely involved:**
- `rex/actions/dispatcher.py`
- `rex/openclaw/tool_executor.py` (read path only; do not enable OpenClaw)
- `rex/tools/*` (any destructive tool)
- `tests/test_destructive_tool_confirmation.py` (new)

**Implementation notes:** Define "destructive" as: filesystem deletion outside per-user data dirs, HA locks/garage/alarms, broad HA scripts/scenes, outbound SMS/email, financial actions, autonomy plan execution. Require an explicit `confirmation_token` or a user-visible GUI confirmation before dispatch. Default refusal must be a clear, user-visible error, not a silent skip.

**Acceptance Criteria:**
- [ ] A registry of destructive tools exists and is documented.
- [ ] Calling a destructive tool without confirmation returns a `requires_confirmation` response with a token.
- [ ] Calling with the matching token completes the action.
- [ ] A negative test asserts that the first call does not execute the side effect.
- [ ] `README.md` and `docs/tools.md` document the gate.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_destructive_tool_confirmation.py -q
```

**Risk notes:** Do not silently drop the action when refused — surface the refusal honestly.

---

### US-026: Twilio fails closed when dependencies or config are absent

**Priority:** P0
**Workstream:** Security / Integrations
**Description:** As an operator, I want SMS to refuse cleanly when Twilio is not configured rather than appear to succeed.

**Files/areas likely involved:**
- `rex/messaging_backends/twilio*.py`
- `bridge/rex_sms_bridge.py`
- `tests/test_twilio_fail_closed.py` (new or updated)

**Acceptance Criteria:**
- [ ] Importing the Twilio backend without the `twilio` package raises a clear `IntegrationUnavailable("twilio not installed")`.
- [ ] Sending without `TWILIO_*` env vars raises a clear error to the caller.
- [ ] No code path returns `success` on a missing-config send.
- [ ] A test asserts a missing-dep send fails with a user-visible error.
- [ ] `docs/messaging.md` documents the behavior.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_twilio_fail_closed.py -q
```

---

### US-027: Redact tokens from GUI settings JSON before persisting

**Priority:** P0
**Workstream:** Security
**Description:** As an operator, I want no secrets written to `config/gui_settings.json`.

**Files/areas likely involved:**
- `gui/src/main/index.ts` (`readGuiSettings` / `writeGuiSettings`)
- `gui/src/main/handlers/*`
- `tests/test_gui_settings_redaction.py` (new)
- `gui/tests/settingsRedaction.test.ts` (new, vitest)

**Acceptance Criteria:**
- [ ] Any key matching a documented secret pattern (API keys, tokens, passwords) is stored only in `.env`, never in `config/gui_settings.json`.
- [ ] A test loads `config/gui_settings.json` from a fixture and asserts no secret pattern appears.
- [ ] When the renderer needs a secret, it requests via IPC and the main process reads `.env`.
- [ ] `docs/configuration.md` and `SECURITY.md` document the rule.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_gui_settings_redaction.py -q
cd gui && npm test -- settingsRedaction || true
```

---

### US-028: Verify no tokens in tracked config

**Priority:** P0
**Workstream:** Security / CI
**Description:** As a maintainer, I want CI to fail if a token-looking string is committed under `config/`.

**Files/areas likely involved:**
- `.secrets.baseline`
- `scripts/security_audit.py`
- `.github/workflows/ci.yml`

**Acceptance Criteria:**
- [ ] `detect-secrets` scan covers `config/`.
- [ ] A test fixture confirms a known secret pattern under `config/` would fail the scan.
- [ ] The PR review checklist mentions secret-scan results.
- [ ] `SECURITY.md` documents the rule.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python -m detect_secrets scan --baseline .secrets.baseline config/
```

---

### US-029: Close out the security audit inventory

**Priority:** P0
**Workstream:** Security / Docs
**Description:** As a maintainer, I want the inventory from US-001 to show zero untriaged actionable findings on the release-candidate commit.

**Files/areas likely involved:**
- `docs/security/AUDIT-INVENTORY.md`
- `scripts/security_audit.py`

**Acceptance Criteria:**
- [ ] Every row in `docs/security/AUDIT-INVENTORY.md` is either `resolved` or `documented-and-accepted`.
- [ ] No row is `production-blocker` with status `open`.
- [ ] `python scripts/security_audit.py` exits 0 OR exits with only findings explicitly listed in an allowlist with justification.
- [ ] `README.md`'s "Security baseline" section is current.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/security_audit.py
```

---

### US-030: Run `ruff check .` over the full tree in CI

**Priority:** P0
**Workstream:** CI
**Description:** As a maintainer, I want CI to lint the full repository, not just `rex/`, `tests/`, `bridge/`, and `*.py` at the root.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`
- `pyproject.toml` `[tool.ruff]`

**Acceptance Criteria:**
- [ ] CI invokes `ruff check .` (excluding `archived/`) and fails on any error.
- [ ] `pyproject.toml` excludes are reviewed and minimized.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
ruff check .
```

---

### US-031: Run `black --check` over Python source, `bridge/`, `scripts/`, `tests/`, and root Python files in CI

**Priority:** P0
**Workstream:** CI
**Description:** As a maintainer, I want CI Black coverage to include `scripts/` too.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`

**Acceptance Criteria:**
- [ ] CI runs `black --check --diff rex/ tests/ bridge/ scripts/ *.py`.
- [ ] Any unformatted file fails the check.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
black --check --diff rex/ tests/ bridge/ scripts/ *.py
```

---

### US-032: Run `pytest` excluding only documented slow/audio/GPU markers in CI

**Priority:** P0
**Workstream:** CI
**Description:** As a maintainer, I want CI test scope to match the documented marker policy.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`
- `pyproject.toml` `[tool.pytest.ini_options]`

**Acceptance Criteria:**
- [ ] CI runs `pytest -m "not slow and not audio and not gpu"`.
- [ ] Marker docs list which markers are excluded and why.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest -m "not slow and not audio and not gpu" -q
```

---

### US-033: Add the wheel contents smoke test as a required CI check

**Priority:** P0
**Workstream:** CI / Packaging
**Description:** As a maintainer, I want CI to run `scripts/check_wheel_contents.py` as a blocking gate.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`

**Acceptance Criteria:**
- [ ] A `wheel-smoke` job runs `python -m build` and `python scripts/check_wheel_contents.py`.
- [ ] Job is required for merge.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python -m build
python scripts/check_wheel_contents.py
```

---

### US-034: Add the security audit as a required CI check

**Priority:** P0
**Workstream:** CI / Security
**Description:** As a maintainer, I want CI to fail when `scripts/security_audit.py` detects new untriaged actionable findings.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`

**Acceptance Criteria:**
- [ ] A `security-audit` job runs `python scripts/security_audit.py` and fails on a non-zero exit.
- [ ] The job is documented in `SECURITY.md`.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/security_audit.py
```

---

### US-035: Add "no generated artifacts committed" check to CI

**Priority:** P0
**Workstream:** CI
**Description:** As a maintainer, I want CI to fail if generated artifacts like `.coverage`, `coverage.xml`, `htmlcov/`, `dist/`, `build/`, or compiled Python caches are committed.

**Files/areas likely involved:**
- `scripts/check_no_generated_artifacts.py` (new)
- `.gitignore`
- `.github/workflows/ci.yml`

**Acceptance Criteria:**
- [ ] Script enumerates generated patterns and fails if any are tracked.
- [ ] CI runs the script.
- [ ] `.gitignore` covers each pattern.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/check_no_generated_artifacts.py
```

---

### US-036: Add "working tree clean after tests" check to CI

**Priority:** P0
**Workstream:** CI
**Description:** As a maintainer, I want CI to fail if a test modified a tracked file.

**Files/areas likely involved:**
- `.github/workflows/ci.yml` (the existing "Verify tests did not modify tracked files" step — promote to all relevant jobs)

**Acceptance Criteria:**
- [ ] Every job that runs tests includes the working-tree-clean check.
- [ ] The check ignores documented artifacts (`.coverage`, `coverage.xml`, `htmlcov/`).
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest -q
git status --porcelain -- ':!.coverage' ':!coverage.xml' ':!htmlcov/'
```

---

### US-037: Skip budget enforcement in CI

**Priority:** P0
**Workstream:** CI / Tests
**Description:** As a maintainer, I want CI to fail when total skipped tests exceed a documented budget.

**Files/areas likely involved:**
- `scripts/check_skip_budget.py` (new)
- `docs/testing/SKIPPED-TESTS-INVENTORY.md`
- `.github/workflows/ci.yml`

**Implementation notes:** Parse the pytest output (`-rs`) to count skipped tests. Compare against `SKIP_BUDGET` declared in `pyproject.toml` or a top-of-file constant. Default budget is the count from US-002.

**Acceptance Criteria:**
- [ ] Script enforces the budget and fails when exceeded.
- [ ] Budget is documented and matches the post-US-002 count minus removals from US-039.
- [ ] CI runs the script after the test suite.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest -rs --no-header -q | tee /tmp/pytest.out
python scripts/check_skip_budget.py /tmp/pytest.out
```

---

### US-038: Classify each skipped test and link to a follow-up if needed

**Priority:** P1
**Workstream:** Tests / Docs
**Description:** As a maintainer, I want every entry in the skip inventory tied to an action (keep, remove, replace, fix).

**Files/areas likely involved:**
- `docs/testing/SKIPPED-TESTS-INVENTORY.md`
- `tests/` (annotation only)

**Acceptance Criteria:**
- [ ] Every inventory row has an action and, where action is non-trivial, a follow-up story ID.
- [ ] Inventory is committed and current.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
grep -E "TODO|FIXME|none" docs/testing/SKIPPED-TESTS-INVENTORY.md || echo "ok"
```

---

### US-039: Remove or archive tests for retired surfaces

**Priority:** P1
**Workstream:** Tests
**Description:** As a maintainer, I want tests that target removed surfaces (Tkinter GUI, shopping PWA, retired Flask dashboard) gone from the active suite.

**Files/areas likely involved:**
- `tests/` (any file targeting `archived/` surfaces)
- `archived/`

**Acceptance Criteria:**
- [ ] Tests for retired surfaces are either deleted or moved under `archived/` with the surface they tested.
- [ ] `pytest --collect-only -q` collects fewer tests after the change AND no collection error appears.
- [ ] Skip inventory is updated.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest --collect-only -q | wc -l
pytest -q
```

**Risk notes:** Do not delete tests for surfaces classified as `deprecated` — those still need coverage.

---

### US-040: Add or restore tests for current supported surfaces that lost coverage

**Priority:** P1
**Workstream:** Tests
**Description:** As a maintainer, I want the test suite to actually cover the surfaces this PRD classifies as shippable.

**Files/areas likely involved:**
- `tests/test_cli_smoke.py` or equivalent
- `tests/test_voice_loop_smoke.py`
- `tests/test_electron_bridge_contract.py`

**Implementation notes:** Identify shippable surfaces (`rex` CLI, packaged Electron app, bridge scripts, `rex_loop.py`) with weak or missing coverage. Add minimum-viable tests that exercise the public contract.

**Acceptance Criteria:**
- [ ] At least one direct test per shippable surface.
- [ ] Coverage gate (`fail_under = 75`) still passes.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest -q --cov=rex --cov-fail-under=75
```

---

### US-041: Replace fake Docker healthcheck or classify Docker as developer-only

**Priority:** P1
**Workstream:** Packaging / Docs
**Description:** As an operator, I want the Docker healthcheck to mean something OR I want the docs to be honest that Docker is not a supported production path.

**Files/areas likely involved:**
- `Dockerfile`
- `docs/docker.md`
- `SURFACE-CLASSIFICATION.md`
- `README.md`

**Implementation notes:** Decision for this PRD: Docker stays a developer-only path. Replace the no-op healthcheck with `python -m rex doctor --healthcheck` (or equivalent) so it returns non-zero on real failure, and classify Docker as `developer-only` everywhere.

**Acceptance Criteria:**
- [ ] `Dockerfile` HEALTHCHECK invokes a real check that returns non-zero on failure.
- [ ] `docker build .` succeeds and `docker run --rm askrex-assistant python -m rex doctor` exits 0.
- [ ] `docs/docker.md`, `README.md`, and `SURFACE-CLASSIFICATION.md` describe Docker as developer-only.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
docker build -t askrex-assistant:smoke .
docker run --rm askrex-assistant:smoke python -m rex doctor
```

**Risk notes:** CI may not run Docker; local smoke is acceptable evidence with a documented log.

---

### US-042: Define Hold-to-Talk as the production voice path

**Priority:** P0
**Workstream:** Voice / Docs
**Description:** As an end user, I want the documented default voice mode to be reliable (Hold-to-Talk) instead of marketing wake word as production.

**Files/areas likely involved:**
- `rex_loop.py`
- `rex/voice_loop.py`
- `README.md`
- `docs/voice_identity.md` (and any voice docs)
- `SURFACE-CLASSIFICATION.md`

**Implementation notes:** Add a `--mode hold-to-talk` and `--mode wake-word` flag to the voice loop entry, with `hold-to-talk` as the default. Document that wake word is beta until reliability tests pass (US-046).

**Acceptance Criteria:**
- [ ] CLI default is Hold-to-Talk.
- [ ] A test confirms the default mode resolves to Hold-to-Talk when no flag is provided.
- [ ] `README.md` says Hold-to-Talk is the supported production voice mode.
- [ ] `SURFACE-CLASSIFICATION.md` classifies wake word as `beta`/`developer-only` until US-046.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_voice_loop_default_mode.py -q
python -m rex --help | grep -i "voice\|hold"
```

**Risk notes:** Existing wake word users must have a single flag to opt back in.

---

### US-043: Voice pipeline structured logs

**Priority:** P0
**Workstream:** Voice
**Description:** As an operator, I want each voice pipeline stage to emit a structured log line with timing.

**Files/areas likely involved:**
- `rex/voice_loop.py`
- `rex_loop.py`
- `rex/logging_utils.py`
- `tests/test_voice_pipeline_logs.py` (new)

**Implementation notes:** Emit JSON log records for `wake_detected`, `capture_started`, `capture_ended`, `stt_started`, `stt_completed`, `llm_started`, `llm_completed`, `tts_started`, `playback_completed`. Each record includes a `session_id`, monotonic `start_ns`, and `duration_ms` where applicable.

**Acceptance Criteria:**
- [ ] All nine events are emitted with the documented fields.
- [ ] A test captures the log stream and asserts every expected event for one happy-path session.
- [ ] `docs/voice_identity.md` (or new `docs/voice_pipeline.md`) documents the log contract.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_voice_pipeline_logs.py -q
```

---

### US-044: Voice latency budget tests

**Priority:** P0
**Workstream:** Voice
**Description:** As an operator, I want CI to fail if voice stage latencies regress beyond documented budgets.

**Files/areas likely involved:**
- `tests/test_voice_latency_budget.py` (new)
- `docs/voice_pipeline.md`

**Implementation notes:** Define stage budgets (e.g., STT < 1500 ms, LLM token-to-first < 500 ms for local provider) and assert under a synthetic input fixture. Mark as `slow` if needed.

**Acceptance Criteria:**
- [ ] Budget table documented in `docs/voice_pipeline.md`.
- [ ] At least one test enforces a budget on each stage.
- [ ] Test runs on CI under `slow` only OR under default markers if fast enough.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_voice_latency_budget.py -q
```

**Risk notes:** Avoid flakiness; use a synthetic source and a wide margin on first introduction.

---

### US-045: Mic/speaker diagnostics surfaced to the user

**Priority:** P0
**Workstream:** Voice
**Description:** As an end user, I want a clear error if my mic or speaker is not available.

**Files/areas likely involved:**
- `rex/audio_config.py`
- `rex/voice_loop.py`
- `gui/src/main/handlers/voice.ts`
- `tests/test_audio_diagnostics.py`

**Acceptance Criteria:**
- [ ] On mic init failure, voice loop emits a structured error AND the Electron GUI shows a visible error toast/banner.
- [ ] On speaker init failure, the same holds.
- [ ] A test confirms the error is surfaced to the IPC handler with a user-actionable message.
- [ ] `docs/troubleshooting.md` lists the new errors.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_audio_diagnostics.py -q
```

---

### US-046: Wake word reliability classification

**Priority:** P0
**Workstream:** Voice / Docs
**Description:** As a maintainer, I want documented wake word reliability metrics before claiming wake word as production.

**Files/areas likely involved:**
- `tests/test_wakeword_reliability.py` (new)
- `docs/voice_pipeline.md`
- `README.md`, `SURFACE-CLASSIFICATION.md`

**Implementation notes:** Run a controlled wake word fixture (positives + negatives) and record precision/recall/latency. Until both precision and recall pass a documented threshold (default 0.9), wake word remains `beta` in docs.

**Acceptance Criteria:**
- [ ] A test produces precision/recall numbers from a fixture and writes them to a tracked report file (`docs/voice/wakeword-report.md`).
- [ ] If thresholds pass, `SURFACE-CLASSIFICATION.md` may reclassify wake word.
- [ ] If thresholds fail, docs continue to label wake word as `beta`.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_wakeword_reliability.py -q
```

**Risk notes:** Fixture data must be redistributable. Use synthesized samples or licensed clips.

---

### US-047: Home Assistant confirmation gates for risky domains

**Priority:** P0
**Workstream:** Home Assistant / Security
**Description:** As a user, I want HA actions on `lock`, `cover` (garage), `alarm_control_panel`, and broad `script.*`/`scene.*` to require explicit confirmation.

**Files/areas likely involved:**
- `rex/integrations/home_assistant*.py` (or wherever HA dispatch lives)
- `gui/src/main/handlers/devices.ts`
- `tests/test_ha_confirmation_gate.py` (new)

**Acceptance Criteria:**
- [ ] Calling a risky-domain action without confirmation returns `requires_confirmation`.
- [ ] Confirmed call proceeds.
- [ ] Negative test asserts side effect did not occur on the first call.
- [ ] `docs/home_assistant.md` lists the risky domains and the gate.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_ha_confirmation_gate.py -q
```

---

### US-048: Home Assistant post-control state verification

**Priority:** P0
**Workstream:** Home Assistant / Verification
**Description:** As a user, I want Rex to verify the entity's actual state after a control call before reporting success.

**Files/areas likely involved:**
- `rex/integrations/home_assistant*.py`
- `gui/src/main/handlers/devices.ts`
- `tests/test_ha_verification.py` (new)

**Implementation notes:** After dispatching, poll the entity state up to a configurable timeout. Compare expected vs actual. Report `verified` when the state matches, `attempted` when dispatch returned but state did not yet change, `completed` when state changes were applied but verification is not applicable, `failed` on dispatch error.

**Acceptance Criteria:**
- [ ] Verification is run for switchable domains (`switch`, `light`, `lock`, `cover`).
- [ ] Return shape is `{ status, expected, actual, latency_ms }`.
- [ ] Tests cover happy path and the "state did not change" path.
- [ ] `docs/home_assistant.md` documents the verification model.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_ha_verification.py -q
```

**Risk notes:** Verification must not block the UI longer than the documented timeout.

---

### US-049: Response language distinguishes attempted/completed/verified/failed

**Priority:** P0
**Workstream:** Home Assistant / Verification / UX
**Description:** As a user, I want Rex's speech and text responses to use precise verification language.

**Files/areas likely involved:**
- `rex/response/builder.py`
- `rex/actions/dispatcher.py`
- `gui/src/pages/DevicesPage.tsx`
- `tests/test_response_verification_language.py` (new)

**Acceptance Criteria:**
- [ ] A response builder helper maps `{ status }` to user-facing text per the documented vocabulary.
- [ ] Tests assert each status produces the correct phrase ("I tried…", "I asked HA to…", "Confirmed the light is on", "That failed because…").
- [ ] No code path produces a confident success message when `status != "verified"` and verification was applicable.
- [ ] `README.md` mentions the verification language.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_response_verification_language.py -q
```

---

### US-050: OpenClaw disabled by default unless fully configured

**Priority:** P0
**Workstream:** OpenClaw / Security
**Description:** As a user, I want OpenClaw off by default and labeled experimental until its dynamic plugin, permission, and verification work is done.

**Files/areas likely involved:**
- `rex/config.py` (`use_openclaw_tools`, `use_openclaw_voice_backend`)
- `gui/src/pages/SettingsPage.tsx`
- `docs/openclaw-migration-status.md`
- `SURFACE-CLASSIFICATION.md`

**Acceptance Criteria:**
- [ ] Defaults for both OpenClaw flags are False.
- [ ] Enabling either flag without a valid gateway URL+token raises a clear error at startup.
- [ ] `SURFACE-CLASSIFICATION.md` classifies OpenClaw surfaces as `experimental`.
- [ ] GUI settings label OpenClaw as "Experimental — off by default".
- [ ] A test asserts defaults and the fail-closed startup behavior.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_openclaw_defaults.py -q
```

---

### US-051: OpenClaw gateway health, reconnect, and graceful degradation

**Priority:** P1
**Workstream:** OpenClaw
**Description:** As an operator, I want a clear health surface for the OpenClaw gateway with reconnect and graceful degradation when down.

**Files/areas likely involved:**
- `rex/openclaw/http_client.py`
- `rex/openclaw/tool_bridge.py`
- `tests/test_openclaw_health.py` (new)

**Acceptance Criteria:**
- [ ] `GET /healthz` (or equivalent) detects gateway availability.
- [ ] On gateway failure, tool dispatch falls back to local execution AND emits a structured warning; it does not silently succeed.
- [ ] Reconnect attempts are bounded by config.
- [ ] Tests cover up/down/recovery paths.
- [ ] `docs/openclaw-migration-status.md` documents the model.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_openclaw_health.py -q
```

---

### US-052: OpenClaw GUI enable/disable/status page

**Priority:** P1
**Workstream:** OpenClaw / Electron
**Description:** As an operator, I want a settings page that shows OpenClaw status, lets me enable/disable it, and warns it is experimental.

**Files/areas likely involved:**
- `gui/src/pages/SettingsPage.tsx` (or new `pages/OpenClawSettingsPage.tsx`)
- `gui/src/main/handlers/openclaw.ts` (new)
- `gui/src/preload/`

**Acceptance Criteria:**
- [ ] A page shows gateway URL, connection health, enabled flags, last error.
- [ ] Toggling either flag persists via IPC.
- [ ] Page renders an experimental warning.
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Manual: page renders in packaged app.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
```

---

### US-053: README capabilities/status table

**Priority:** P0
**Workstream:** Docs
**Description:** As a reader, I want a single table that shows which surfaces are working, partial, experimental, developer-only, or removed.

**Files/areas likely involved:**
- `README.md`
- `SURFACE-CLASSIFICATION.md`
- `docs/UI_SURFACES.md`
- `docs/claude/INTEGRATIONS_STATUS.md`

**Acceptance Criteria:**
- [ ] README has a "Capabilities & Status" table that mirrors `SURFACE-CLASSIFICATION.md`.
- [ ] Every row links to the deeper doc for that surface.
- [ ] No conflicting status claims between README, `SURFACE-CLASSIFICATION.md`, `docs/UI_SURFACES.md`, and `INTEGRATIONS_STATUS.md`.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
grep -n "Capabilities" README.md
```

---

### US-054: Cross-doc consistency audit and fixes

**Priority:** P0
**Workstream:** Docs
**Description:** As a reader, I want README, INSTALL, RUNNING, `docs/UI_SURFACES.md`, `SURFACE-CLASSIFICATION.md`, `docs/claude/INTEGRATIONS_STATUS.md`, and `CLAUDE.md` to agree.

**Files/areas likely involved:**
- All of the above

**Acceptance Criteria:**
- [ ] A `docs/AUDIT-CROSS-DOC.md` (new) lists every cross-doc claim about install methods, console scripts, root file count, voice mode default, OpenClaw status, Docker tier, and HA verification.
- [ ] Every claim is verified against the code at the audit commit.
- [ ] Conflicts are resolved in the same story.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
grep -n "rex-gui\|rex_loop\|wake word\|OpenClaw\|Docker" README.md INSTALL.md RUNNING.md docs/UI_SURFACES.md SURFACE-CLASSIFICATION.md CLAUDE.md
```

---

### US-055: `CLAUDE.md` truth pass

**Priority:** P0
**Workstream:** Docs
**Description:** As a maintainer using Claude Code, I want `CLAUDE.md` to reflect the post-refactor reality.

**Files/areas likely involved:**
- `CLAUDE.md`

**Acceptance Criteria:**
- [ ] Root `.py` file count and list are accurate.
- [ ] Console-script list matches `pyproject.toml`.
- [ ] Voice-mode default matches US-042.
- [ ] OpenClaw status matches US-050.
- [ ] Docker tier matches US-041.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
grep -nE "Active root-level|Entry points|OpenClaw|Docker|wake word" CLAUDE.md
```

---

### US-056: Replace `datetime.utcnow()` with timezone-aware UTC

**Priority:** P2
**Workstream:** Tech Debt
**Description:** As a maintainer, I want all `datetime.utcnow()` calls replaced with `datetime.now(timezone.utc)`.

**Files/areas likely involved:**
- `rex/assistant.py` lines 404, 840
- Any other call sites discovered by grep

**Acceptance Criteria:**
- [ ] `grep -rn "datetime.utcnow" rex/` returns no results.
- [ ] Tests assert the timestamps are timezone-aware (`tzinfo is not None`).
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
grep -rn "datetime.utcnow" rex/ || echo "clean"
pytest -q
```

---

### US-057: Replace deprecated `asyncio.get_event_loop()` patterns where safe

**Priority:** P2
**Workstream:** Tech Debt
**Description:** As a maintainer, I want `asyncio.get_event_loop()` replaced with `asyncio.get_running_loop()` (when inside async) or `asyncio.new_event_loop()` (when bootstrapping), per Python 3.12+ guidance.

**Files/areas likely involved:**
- `rex/geolocation.py` line 41
- `rex/openclaw/tool_executor.py` line 560
- `rex/tts_voices.py` lines 192, 244

**Acceptance Criteria:**
- [ ] `grep -rn "asyncio.get_event_loop" rex/` returns no results.
- [ ] Tests cover the replaced call sites.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
grep -rn "asyncio.get_event_loop" rex/ || echo "clean"
pytest -q
```

**Risk notes:** Some patterns may be inside sync code paths; pick the replacement that preserves behavior.

---

### US-058: Lint rule preventing reintroduction of deprecated datetime/asyncio patterns

**Priority:** P2
**Workstream:** Tech Debt / CI
**Description:** As a maintainer, I want a ruff or grep rule that fails CI if the deprecated patterns return.

**Files/areas likely involved:**
- `pyproject.toml` `[tool.ruff.lint]`
- `scripts/check_deprecated_apis.py` (new)
- `.github/workflows/ci.yml`

**Acceptance Criteria:**
- [ ] Ruff rule or check script fails on `datetime.utcnow()` and `asyncio.get_event_loop()` outside `archived/`.
- [ ] CI runs the check.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python scripts/check_deprecated_apis.py
```

---

### US-059: Split `rex/cli.py` by command domain (P2)

**Priority:** P2
**Workstream:** Tech Debt
**Description:** As a maintainer, I want `rex/cli.py` split into per-domain modules under `rex/cli/` for readability.

**Files/areas likely involved:**
- `rex/cli.py` → `rex/cli/__init__.py`, `rex/cli/voice.py`, `rex/cli/chat.py`, `rex/cli/integrations.py`, etc.

**Implementation notes:** Behavior-preserving split only. `python -m rex` continues to work. All console-script entry points keep their import paths via re-export from `rex/cli/__init__.py`.

**Acceptance Criteria:**
- [ ] Each new module is < 1,000 lines.
- [ ] All CLI subcommands continue to work (covered by `tests/test_cli_smoke.py`).
- [ ] No behavior change observable to a caller.
- [ ] Coverage holds.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
python -m rex --help
pytest tests/test_cli_smoke.py -q
pytest -q --cov=rex --cov-fail-under=75
```

**Risk notes:** Big-bang splits are dangerous. Land in a single story only if tests can prove parity; otherwise split into multiple stories.

---

### US-060: Split `rex/voice_loop.py` by concern (P2)

**Priority:** P2
**Workstream:** Tech Debt
**Description:** As a maintainer, I want `rex/voice_loop.py` split into capture, STT, LLM, TTS, and orchestration modules.

**Files/areas likely involved:**
- `rex/voice_loop.py` → `rex/voice/capture.py`, `rex/voice/stt.py`, `rex/voice/llm.py`, `rex/voice/tts.py`, `rex/voice/orchestrator.py`

**Acceptance Criteria:**
- [ ] Behavior-preserving split (covered by US-043 logs, US-044 budget).
- [ ] `python rex_loop.py` continues to work.
- [ ] All voice tests still pass.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_voice_loop_smoke.py -q
pytest tests/test_voice_pipeline_logs.py -q
```

---

### US-061: Split `rex/gui_app.py` by Flask route domain (P2)

**Priority:** P2
**Workstream:** Tech Debt
**Description:** As a maintainer, I want `rex/gui_app.py` split by route domain (status, devices, history, setup, quick actions) under `rex/web/routes/`.

**Files/areas likely involved:**
- `rex/gui_app.py` → `rex/web/app.py`, `rex/web/routes/*.py`

**Acceptance Criteria:**
- [ ] Behavior-preserving split.
- [ ] `rex-gui` console script still works.
- [ ] Route tests still pass.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
pytest tests/test_gui_app_routes.py -q
rex-gui --help
```

---

### US-062: Split `gui/src/main/index.ts` by Electron main-process concern (P2)

**Priority:** P2
**Workstream:** Tech Debt
**Description:** As a maintainer, I want the Electron main process split into IPC handlers, config helpers, bridge resolver, tray, and window lifecycle modules.

**Files/areas likely involved:**
- `gui/src/main/index.ts` → smaller modules under `gui/src/main/`

**Acceptance Criteria:**
- [ ] Behavior-preserving split.
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Electron smoke test still passes.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
bash tests/smoke/test_electron_package.sh
```

---

### US-063: Split `gui/src/pages/SettingsPage.tsx` into sections (P2)

**Priority:** P2
**Workstream:** Tech Debt
**Description:** As a maintainer, I want `SettingsPage.tsx` split into smaller section components (general, voice, AI, integrations, system).

**Files/areas likely involved:**
- `gui/src/pages/SettingsPage.tsx` → `gui/src/pages/settings/sections/*.tsx`

**Acceptance Criteria:**
- [ ] Each section module is < 1,000 lines.
- [ ] `cd gui && npm run typecheck && npm run build` passes.
- [ ] Manual: every settings section still renders and saves.
- [ ] All relevant GitHub checks pass.

**Validation commands:**
```bash
cd gui && npm run typecheck && npm run build
```

---

## 9. Global Acceptance Criteria

Every story in Section 8 inherits these criteria. They are checked in addition to the per-story criteria.

- A story is not marked `[x]` until:
  - All per-story acceptance criteria are checked.
  - All per-story validation commands exit 0 on a clean checkout.
  - All required GitHub checks (Section 10) pass on the PR that delivers the story.
  - Any user-facing behavior change is reflected in README, INSTALL, RUNNING, `docs/UI_SURFACES.md`, `SURFACE-CLASSIFICATION.md`, `docs/claude/INTEGRATIONS_STATUS.md`, and `CLAUDE.md` as relevant — in the same commit or the same PR as the implementation.
  - The story does NOT introduce a new raw `fetch('/api/...')` call in the renderer unless explicitly justified in the allowlist.
  - The story does NOT introduce a new use of `datetime.utcnow()` or `asyncio.get_event_loop()`.
  - The story does NOT introduce a new exposed secret, an unauthenticated network-facing route, or a destructive tool without a confirmation gate.
  - The story does NOT claim success without verification when a verification path is available.
  - The PR description includes the validation command output (pasted or referenced via a CI artifact).
  - Conventional Commits is used for the commit and PR title.

---

## 10. Required GitHub Checks

A story is not done until all of these CI jobs are green on the PR that delivers it. Names may evolve in CI; the required set is the substance, not the label.

| Check | Source | Notes |
|-------|--------|-------|
| `ruff` | `ruff check .` | Whole repo minus `archived/`. |
| `black` | `black --check --diff rex/ tests/ bridge/ scripts/ *.py` | Includes `scripts/`. |
| `mypy` | `mypy rex --ignore-missing-imports` | Existing CI. |
| `pytest` | `pytest -m "not slow and not audio and not gpu"` | Documented marker exclusions only. |
| `integration` | `pytest -m integration -q` | Existing CI. |
| `gui-typecheck` | `cd gui && npm run typecheck` | Existing CI. |
| `gui-build` | `cd gui && npm run build` | Existing CI. |
| `electron-smoke` | `bash tests/smoke/test_electron_package.sh` | Promoted to required for renderer/bridge changes. |
| `wheel-smoke` | `python scripts/check_wheel_contents.py` | Introduced by US-015/US-033. |
| `console-scripts-smoke` | `pytest tests/test_console_scripts_smoke.py` | Introduced by US-019. |
| `security-audit` | `python scripts/security_audit.py` | Introduced by US-034. |
| `pip-audit` | Existing CI | Vulnerability scan with the documented allowlist. |
| `secret-scan` | `detect-secrets` | Existing CI; expanded scope in US-028. |
| `pre-commit` | `pre-commit run --all-files` | Existing CI. |
| `node-audit` | `npm audit --audit-level=high` | Existing CI. |
| `no-raw-api-fetch` | `python scripts/check_no_renderer_api_fetch.py` | Introduced by US-003; enforced empty in US-011. |
| `no-generated-artifacts` | `python scripts/check_no_generated_artifacts.py` | Introduced by US-035. |
| `working-tree-clean` | `git status --porcelain` post-test | Strengthened by US-036. |
| `skip-budget` | `python scripts/check_skip_budget.py` | Introduced by US-037. |
| `deprecated-api-guard` | `python scripts/check_deprecated_apis.py` | Introduced by US-058. |

---

## 11. Documentation and README Update Policy

This policy is enforced by Section 9's global acceptance criteria. It is restated here so reviewers and Ralph iterations can check the rule directly.

1. **Update with code, not after.** A story that changes user-facing behavior, install flow, GUI behavior, commands, dependencies, file structure, configuration, integrations, or capability claims MUST update relevant docs in the same PR as the implementation.
2. **Authoritative surface list.** `SURFACE-CLASSIFICATION.md` is the source of truth for what each surface is. Other docs must reference it, not contradict it.
3. **Capabilities table.** `README.md` carries a single "Capabilities & Status" table (US-053) that mirrors `SURFACE-CLASSIFICATION.md`.
4. **Cross-doc consistency.** README, INSTALL, RUNNING, `docs/UI_SURFACES.md`, `SURFACE-CLASSIFICATION.md`, `docs/claude/INTEGRATIONS_STATUS.md`, and `CLAUDE.md` must not contradict each other. US-054 closes the current gap; future stories preserve consistency.
5. **No silent claim changes.** If a story changes whether a surface is shippable, developer-only, deprecated, experimental, or archived, that change must be reflected in `SURFACE-CLASSIFICATION.md` and propagated to all referencing docs in the same PR.
6. **Verification language.** Any doc that describes Rex saying something to the user about an action's result must use the verified vocabulary: *attempted, completed, verified, failed*.
7. **Integration docs.** Email, calendar, SMS, MQTT, Home Assistant, web search, and OpenClaw docs must state the current tier (working / partial / experimental / developer-only) and the documented fail-closed behavior.
8. **No retired surface promotion.** No doc may advertise a surface that this PRD classifies as archived or removed.

---

## 12. Definition of Production Ready

The release candidate is "Production Ready" when ALL of the following are true on a single commit on `master`:

- [ ] Every User Story US-001 through US-058 is `[x]` (P0 and P1 work).
- [ ] P2 stories US-059 through US-063 may remain open and tracked, but `master` does not contain any partial decomposition.
- [ ] `python scripts/security_audit.py` exits 0 OR all findings are documented in `docs/security/AUDIT-INVENTORY.md` and accepted with an owner and expiry.
- [ ] `python scripts/check_no_renderer_api_fetch.py` exits 0 with an empty allowlist.
- [ ] `python scripts/check_wheel_contents.py` exits 0.
- [ ] `pytest -m "not slow and not audio and not gpu"` exits 0.
- [ ] `pytest -m integration` exits 0.
- [ ] `python scripts/check_skip_budget.py` exits 0.
- [ ] `python scripts/check_deprecated_apis.py` exits 0.
- [ ] `bash tests/smoke/test_electron_package.sh` exits 0 and confirms no Flask backend is required at runtime.
- [ ] `python -m build` produces a wheel that passes `scripts/check_wheel_contents.py`.
- [ ] All console scripts run `--help` cleanly after `pip install`.
- [ ] README "Capabilities & Status" table is current.
- [ ] `SURFACE-CLASSIFICATION.md`, `CLAUDE.md`, `docs/UI_SURFACES.md`, and `docs/claude/INTEGRATIONS_STATUS.md` agree with README.
- [ ] No completed implementation commit on `master` left its story unchecked in this PRD.
- [ ] All required GitHub checks listed in Section 10 are green on the release-candidate commit.
- [ ] The core AskRex principle is preserved: every reachable code path that reports an action's success does so only when the action was verified, or it uses the precise *attempted / completed / failed* vocabulary instead.
