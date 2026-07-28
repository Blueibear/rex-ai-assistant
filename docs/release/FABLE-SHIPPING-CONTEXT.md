# Fable Shipping Context — AskRex Assistant

Prepared by the initializer pass on 2026-07-28. This document is the compact,
verified handoff for continued shipping-readiness work. Every non-obvious
claim below cites a path, SHA, PR/issue number, workflow, or test. Treat this
file as a snapshot — re-verify a claim before acting on it if the underlying
code may have moved since this SHA.

## 1. Baseline

- **Repository:** `Blueibear/AskRex-Assistant`, default branch `master`.
- **Current master / initializer-branch SHA:** `8dccbe3f8bc03e71fa957d60b19f79fb0d812013`
  — merge commit for PR [#331](https://github.com/Blueibear/AskRex-Assistant/pull/331),
  "fix(release): remediate production audit findings A-K."
- **Working branch:** `fable/shipping-readiness`, created from and currently
  identical to master tip (verified: `git merge-base HEAD origin/master` ==
  `HEAD`; zero-file diff against `origin/master`). It already existed at
  session start with no divergence — safe to build on directly.
- **Working tree:** clean at session start, no uncommitted user work found.
- **CI on master tip:** all release-relevant checks green — Lint & Format
  Check, Type Check (mypy), GUI TypeScript Typecheck, GUI Vitest Tests, GUI
  Build, Node Dependency Audit, Python 3.11 Tests & Coverage, Dependency
  Vulnerability Scan, Pre-commit Hook Validation, GUI Raw API Fetch Guard,
  Wheel Contents Smoke Test, Hardcoded Secret Scan, and
  `windows-electron-artifact / Installed Windows Electron artifact` (verified
  via `gh api repos/.../commits/8dccbe3.../check-runs`). Only unrelated
  Dependabot dependency-graph-update checks show failures (npm/pip graph
  jobs for open dependency PRs, not release gates).
- **Test/coverage evidence (from PR #331, now master truth):** 8,122 passed,
  84 skipped, 83.05% coverage against a 75% required threshold
  (`docs/audits/AUDIT-REMEDIATION-2026-07-22.md:33`). 36 integration tests
  passed. Ruff, Black, compileall, mypy (362 files) all passed. Security
  release gate scanned 1,231 files, zero actionable findings, zero exposed
  secrets. GUI: `npm ci`, typecheck, 12 Vitest tests, production build, and
  `npm audit --audit-level=high` all passed with zero vulnerabilities.
- **Release version:** latest released is `1.4.1` (2026-07-12); `CHANGELOG.md`
  top section is `## Unreleased`. Open PR [#322](https://github.com/Blueibear/AskRex-Assistant/pull/322)
  is a release-please automation PR proposing `1.5.0` — not yet merged.
- **Signing status:** Authenticode is **not configured**. Installer and
  unpacked app are `NotSigned` (PR #331 description; corroborated in
  `README.md:16,111,141,152` and `INSTALL.md:9,232`). This blocks calling the
  artifact a signed public release.
- **Runtime versions:** Python 3.11 required (3.12+ explicitly unsupported —
  `README.md:9,42`, `INSTALL.md:31`); managed Python 3.11 Voice runtime
  bundled for end users (no machine Python/Node required —
  `README.md:111`, `INSTALL.md:9`); Electron `^42.3.0` (`gui/package.json:39`);
  CPU Torch pinned `2.12.0` in `requirements-cpu.txt:6` — note this pin has a
  **stale comment** two lines above it claiming "Verified working stack uses
  torch 2.6.0" (`requirements-cpu.txt:5`), which is inconsistent with the
  actual pin and worth a one-line comment fix, not a version change.

## 2. Authoritative sources (and what's stale)

- **`PRD-production-readiness.md`** is the sole active release-readiness
  tracker (self-declared at line 3). `PRD.md` and
  `PRD-remaining-release-readiness.md` are both explicitly superseded —
  `PRD.md:1-9` states this outright ("Superseded / historical... Do not use
  unchecked boxes here"), and `PRD-remaining-release-readiness.md:12` defers
  to `PRD-production-readiness.md` as authoritative. Do not treat unchecked
  boxes in either superseded file as live work.
- **`docs/audits/AUDIT-REMEDIATION-2026-07-22.md`** is the current
  cross-cutting implementation ledger for audit findings A–K, and
  `PRD-production-readiness.md:22` explicitly says these findings *supersede*
  the corresponding older PRD prose (calendar isolation, Electron
  identity/data ownership, HA mutation verification, tool lifecycle, managed
  Electron runtime, Windows artifact CI, Hold-to-Talk, integration-state
  truth, diagnostics/security gates, GUI dependency gates). Items are
  **locally verified**, not independently re-confirmed as CI-verified in this
  ledger's own text — but the live CI check-run query above confirms the
  actual GitHub checks are now green on this exact SHA, so treat findings A–K
  as CI-verified as of this baseline.
- **`SURFACE-CLASSIFICATION.md`** (147 lines) is otherwise accurate but has a
  **real gap**: it classifies every `[project.scripts]` entry point and root
  file except the mobile API gateway. `python -m rex mobile-api` and
  `rex/mobile_api/` are absent from both the surface list and its summary
  count table (2 shippable / 28 developer-only / 5 deprecated / 15 archived,
  total 50). This should be added as a developer/backend-service surface
  (consistent with how `rex-tool-server` etc. are classified) in a small doc
  patch.
- **`INTEGRATIONS_STATUS.md`** uses the correct state vocabulary matching
  `rex/integration_state.py`, but likewise has no row for the mobile API
  gateway despite it being a shipped, authenticated surface.
- **PRD.md line-count caveat:** `PRD-production-readiness.md` is 3,129 lines
  with 372 unchecked `- [ ]` boxes remaining (grep count) across User Stories
  US-023 through US-088. Most of these were **not** re-audited by findings
  A–K (which map to specific named areas, not the whole backlog) — see
  Section 7 for which remaining boxes were spot-verified as real vs. stale.

## 3. What is already complete (do not repeat)

All of the following are **CI-verified on the current master SHA** — do not
re-implement, only extend if a genuine new gap is found.

| Area | Evidence |
|---|---|
| Identity / user isolation | `gui/src/main/sessionIdentity.ts`, Electron identity bridge + private-data handlers, ownership migration, two-user isolation tests. Commit `3dac391`. `rex/identity.py` `validate_user_id` fail-closed pattern is the canonical rule (CLAUDE.md "Learned rules"). |
| Tool execution lifecycle + confirmation/verification | `rex/tools/execution.py` — canonical typed lifecycle (availability → argument → identity → permission → risk → confirmation → execution → normalization → independent verification → truthful response → redacted audit). Commit `226741c`. Confirmation-gate infrastructure (`requires_confirmation`, confirmation tokens) also present in `rex/tools/windows_settings.py`, `rex/tools/windows_repair.py`, `rex/openclaw/tools/ha_tool.py`, `rex/local_tool_executor.py`, `rex/ha/mutation_service.py`, `rex/computers/safety.py`. |
| Home Assistant safety | `rex/ha/mutation_service.py`, Electron mutation bridge, OpenClaw HA adapter, truthful response builder. Commit `e118325`. `rex/routes/ha.py:202-206` already calls `_require_auth()` on `/api/ha/test` (this is ahead of PRD US-024's still-unchecked box — the code, not the PRD, is current truth). |
| Windows packaging / managed Python | Managed Python 3.11 embeddable distribution + AskRex wheel + pinned runtime deps + bundled FFmpeg. Commit `41a2ee4`. Voice runtime ≈880 MB, unpacked app ≈1.25 GB. Final validated installer: `gui/dist/AskRex Setup 1.0.0.exe`, 305,471,434 bytes, SHA-256 in the audit ledger. |
| Windows installer CI | Blocking reusable workflow `.github/workflows/windows-electron-artifact.yml`, install/reinstall/uninstall harness. Commit `55eee02`. Green on current master tip. |
| Voice UI (Hold-to-Talk) | Response TTS, selected-output playback, cancellation, replay, barge-in, device-loss fallback, repeated turns, structured timing events. Commit `a10577b`. 123 voice tests + 12 GUI tests passed. Wake-word mode remains beta (unchanged; see issue #304). |
| Integration-state truth | Shared vocabulary in `rex/integration_state.py` (unavailable/unconfigured/configured/reachable/authenticated/degraded/read_only/write_capable/write_tested/verified), CLI, doctor, API/capability registry, Electron UI, draft-only email UX. Commit `310bcb4`. |
| Diagnostics / security gates | Distinct PASS/WARN/ERROR, `rex doctor --release-gate`, actionable/expiring security suppressions. Commit `e55701b`. |
| GUI lint/dependencies | ESLint 9 flat config, upgraded/locked dependencies. Commit `6bb97e4`. **Caveat:** `npm run lint` (`gui/package.json:18`) is defined but **not wired into `.github/workflows/ci.yml`** as a job — only typecheck/test/build run in CI. This is a real, small, currently-open gap (see Section 7). |
| Documentation/planning truth | README, INSTALL, RUNNING, surface docs, integration contract, CLAUDE.md, active/superseded PRD reconciliation, the audit ledger itself. Commit `fa22da6`. |
| Mobile API gateway (issue #323, Session 2) | `rex/mobile_api/` is fully present on master: `app.py`, `auth.py`, `sessions.py`, `chat.py`, `voice.py`, `websocket.py`, `events.py`, `idempotency.py`, `capabilities.py`, `routes/{chat.py,voice.py,auth.py,status.py,scaffolds.py}`, plus `tests/mobile_api/`. Landed via commits `6e3afd4`, `c5c5063`, `8db4afc` (message-identical to, but different SHAs from, PR #326 — see Section 6). Docs at `docs/mobile/`. |

## 4. What must not be repeated

- Do not re-implement identity isolation, tool execution lifecycle,
  confirmation/verification, HA mutation safety, managed Python packaging,
  Windows installer CI, or Hold-to-Talk voice UX — all CI-verified per
  Section 3.
- Do not re-open or re-implement the mobile API gateway (issue #323,
  Session 2 scope) — it is already on master. See Section 6 for the one
  cosmetic delta and the recommended disposition of PR #326.
- Do not treat unchecked boxes in `PRD.md` or
  `PRD-remaining-release-readiness.md` as live requirements — both are
  explicitly superseded.
- Do not re-derive the audit findings A–K narrative; cite
  `docs/audits/AUDIT-REMEDIATION-2026-07-22.md` instead of re-auditing those
  areas from scratch.

## 5. Current open PRs and issues

**Open PRs:**
- [#330](https://github.com/Blueibear/AskRex-Assistant/pull/330) dependabot: setuptools 78.1.1→83.0.0
- [#328](https://github.com/Blueibear/AskRex-Assistant/pull/328) dependabot: pillow 12.1.1→12.3.0
- [#327](https://github.com/Blueibear/AskRex-Assistant/pull/327) dependabot: torch 2.12.0→2.13.0 — **do not merge without re-validating the CPU Torch pin story**; the packaged Voice runtime build fails if Torch drifts outside its exact supported pin (`docs/audits/AUDIT-REMEDIATION-2026-07-22.md:28`).
- [#326](https://github.com/Blueibear/AskRex-Assistant/pull/326) draft, "feat(mobile): add secure chat voice and TTS runtime" — see Section 6.
- [#322](https://github.com/Blueibear/AskRex-Assistant/pull/322) release-please: proposes `askrex-assistant` 1.5.0.
- [#288](https://github.com/Blueibear/AskRex-Assistant/pull/288) dependabot: msgpack 1.1.2→1.2.1.
- [#233](https://github.com/Blueibear/AskRex-Assistant/pull/233) dependabot: pytest 8.3.4→9.0.3 (stale, opened 2026-05-24).

**Open issues (from the requested set):**
- [#299](https://github.com/Blueibear/AskRex-Assistant/issues/299) — Windows 11 physical-hardware acceptance matrix. **User-only** (physical device/mic/HA required).
- [#302](https://github.com/Blueibear/AskRex-Assistant/issues/302) — OpenClaw production safety gate (connection/permissions/verification/GUI visibility). Secondary per task scope; OpenClaw stays optional/disabled by default already.
- [#304](https://github.com/Blueibear/AskRex-Assistant/issues/304) — Quantify wake-word reliability/latency with a real test matrix. Partially executable (diagnostics/harness code) + partially **user-only** (physical mic/speaker runs).
- [#323](https://github.com/Blueibear/AskRex-Assistant/issues/323) — Mobile API gateway. Backend (Session 2) is done on master (Section 3); issue itself is not yet closed and still references a 3-PR delivery plan (planning/contract, foundation, HTTP+WS+voice) — the companion mobile client PR `Blueibear/AskRex#8` and physical-iPhone/LAN validation remain open per PR #326's own "Risk / Notes".
- [#253](https://github.com/Blueibear/AskRex-Assistant/issues/253) — `transformers>=5.0.0` upgrade to drop CVE-2026-1839 / PYSEC-2025-217 suppressions. Blocked on upstream PyPI release; suppression expiry is 2026-08-29 (~1 month from this baseline). **Monitoring item, not yet actionable.**

## 6. PR #326 integration map

**Finding: PR #326 carries essentially zero unique value versus current
master.** Independently verified (not just agent-reported): `git fetch origin
pull/326/head` → `FETCH_HEAD` = `d1b56ff794f6e2bdddc463414604cb7a0bb78687`
(matches the PR's own description). `git diff FETCH_HEAD origin/master --
rex/mobile_api/ tests/mobile_api/` shows **only one real difference**: a
one-line docstring wording change in `rex/mobile_api/capabilities.py`
(`"safe placeholder"` → `"safe fallback value"`). Every other file under
`rex/mobile_api/` and `tests/mobile_api/`, plus `pyproject.toml` and
`rex/config.py`, is byte-identical between the PR #326 branch and master.

Master's git log confirms three commits with **message-identical but
SHA-different** counterparts to PR #326's commits: master has `6e3afd4 feat
(mobile): add secure chat voice and TTS runtime`, `c5c5063 fix(mobile):
harden runtime recovery boundaries`, `8db4afc docs(mobile): document
WebSocket 4403 as reserved and untested`, versus PR #326 branch's `d223f3b`
/ `ac8009b` / `d1b56ff` with the same messages. This matches
`docs/audits/AUDIT-REMEDIATION-2026-07-22.md:42`'s own reconciliation note:
"Five local-only commits were replayed onto `origin/master` as `6e3afd4`,
`078fdec`, `c5c5063`, `2e19c00`, and `8db4afc`." — i.e. this same mobile work
was replayed onto master directly as part of the PR #331 reconciliation,
outside of the PR #326 GitHub flow.

**Recommendation: close PR #326 as superseded**, after a one-sentence
confirmation from the repo owner that this was an intentional local
replay-and-push rather than an accident (it is unusual for an open PR's
exact commit messages to reappear on master under new SHAs). If the single
docstring wording is preferred, land it as a trivial one-line follow-up
commit directly on master — reviving PR #326 is not worth it. No rebase,
merge, or cherry-pick is needed. **Do not merge PR #326 as-is** — GitHub
would show it as an unrelated-history merge of already-landed content.

## 7. Remaining shipping blockers

**Executable software work (Fable can do directly):**
- Wire `npm run lint` (`gui/package.json:18`, ESLint flat config) into
  `.github/workflows/ci.yml` as a required job — currently only
  typecheck/test/build run in CI for the GUI. Small, mechanical, verifiable.
- Add `scripts/` to the Black CI invocation
  (`.github/workflows/ci.yml:46` currently runs `black --check --diff rex/
  tests/ bridge/ *.py`, omitting `scripts/`) — confirmed live gap matching
  PRD US-031 (`PRD-production-readiness.md:1217-1231`).
- Fix the stale comment in `requirements-cpu.txt:5` ("Verified working stack
  uses torch 2.6.0") which contradicts the actual pin on line 6
  (`torch==2.12.0`).
- Add the mobile API gateway to `SURFACE-CLASSIFICATION.md` and
  `INTEGRATIONS_STATUS.md` — both currently omit it despite it being a
  shipped surface.
- Reconcile/close stale PRD boxes once verified against code — e.g. US-024
  (HA test-endpoint auth) is functionally done in code
  (`rex/routes/ha.py:202-206`) but still shown unchecked in the PRD; a
  focused pass to re-verify and check off boxes that are actually satisfied
  (not just US-024) would shrink the 372-box backlog meaningfully without
  new implementation. **Do not assume the whole backlog is stale** — only
  spot-verify before checking anything off.
- Investigate and, if genuinely missing, implement the remaining P0 items
  from `PRD-production-readiness.md` Section US-025 through US-034
  (destructive-tool confirmation *registry + documentation*, Twilio
  fail-closed proof, GUI secret redaction, tracked-config secret scan,
  security-audit CI closeout) — note that confirmation-gate *mechanics*
  already exist in several tool modules (Section 3), so this may be
  primarily a documentation/registry/test-coverage gap rather than new
  runtime logic. Verify against current code before treating as unimplemented.
- Decide and act on PR #326 disposition (Section 6) — a GitHub action, not
  code, but blocks a clean open-PR list.

**Automated validation work:** full pytest+coverage re-run to reconfirm the
83.05% figure still holds after any of the above patches; GUI Vitest re-run
after the lint-CI wiring change.

**Physical-hardware validation (user-only):** issue #299 (Windows 11
acceptance matrix — mic, speakers, HA device control, wake-word on real
hardware), issue #304's hardware-dependent rows (mic/speaker latency and
false-positive testing).

**Live-service validation (user-only or credential-dependent):** Home
Assistant live device control (locks/alarms/covers/lights — audit finding C's
remaining external-verification column), external provider writes
(email/SMS/calendar), physical-iPhone and LAN validation for the mobile
gateway (PR #326 description "Risk / Notes"; issue #323's own DoD).

**Certificate/signing work (user-only):** Authenticode signing certificate
purchase/provisioning — blocks calling any installer a signed public release.

**Product decisions (user-only):** whether to proceed with OpenClaw's
production safety gate (issue #302) now or defer further (task instructions
say keep it secondary); confirmation on PR #326 disposition (Section 6);
whether the 1.5.0 release-please PR (#322) should be merged now or held for
more remediation.

**Optional post-release work:** OpenClaw safety gate (#302) unless core
release is otherwise blocked on it; transformers 5.0 upgrade (#253) once
upstream releases it.

## 8. Exact prioritized execution queue

1. Confirm PR #326 disposition with the repo owner, then close it (Section 6) — trivial, unblocks a clean PR list.
2. Wire GUI `npm run lint` into CI (`.github/workflows/ci.yml`) — small, closes a real gate gap.
3. Add `scripts/` to the Black CI command (`.github/workflows/ci.yml:46`) — small, matches PRD US-031, will require a `black --check scripts/` pass first to see what needs reformatting.
4. Fix the stale Torch comment in `requirements-cpu.txt:5`.
5. Add mobile API gateway rows to `SURFACE-CLASSIFICATION.md` and `INTEGRATIONS_STATUS.md`.
6. Verify and reconcile PRD-production-readiness.md US-023–US-034 (P0/P1 security items) against current code — check off what's genuinely done, scope real remaining implementation work (destructive-tool registry/docs, Twilio fail-closed test, secret redaction).
7. Re-run full validation suite (Section 9) after the above land; update `docs/release/FABLE-SHIPPING-PROGRESS.md`.
8. Evaluate dependabot PR #327 (torch 2.12.0→2.13.0) against the exact-pin requirement for the packaged Voice runtime before merging.
9. Decide on release-please PR #322 (1.5.0) timing.
10. Defer: OpenClaw safety gate (#302), transformers 5.0 (#253, upstream-blocked), physical-hardware/live-service items (user-only, Section 10).

## 9. Canonical validation commands

- Focused backend test file: `pytest -q tests/<file>.py`
- Mobile API tests: `pytest -q tests/mobile_api`
- Full pytest + coverage (CI-exact): `pytest -m "not slow and not audio and not gpu" --cov=rex --cov-fail-under=75 --cov-report=term-missing --cov-report=html --cov-report=xml` — `.github/workflows/ci.yml:235`
- Integration tests: `pytest -m integration -q` — `.github/workflows/ci.yml:242`
- Ruff: `ruff check --output-format=github .` (pinned `ruff==0.15.8`) — `.github/workflows/ci.yml:43`, `.pre-commit-config.yaml:3`
- Black: `black --check --diff rex/ tests/ bridge/ *.py` (pinned `black==26.3.1`; **missing `scripts/`**, see Section 7) — `.github/workflows/ci.yml:46`
- mypy: `mypy rex --ignore-missing-imports` — `.github/workflows/ci.yml:82`
- pre-commit: `pre-commit run --all-files --show-diff-on-failure` — `.github/workflows/ci.yml:418`
- pip-audit: `pip-audit --strict --ignore-vuln <CVE list...> .` — `.github/workflows/ci.yml:301-388`
- npm audit: `npm audit --audit-level=high` run in both `gui/` (`ci.yml:167`) and `rex/ui/` (`ci.yml:177`)
- Hardcoded secrets: `python -m detect_secrets scan --exclude-files "\.venv|__pycache__|\.git|\.egg-info" --baseline .secrets.baseline` — `.github/workflows/ci.yml:480-482`
- Security release gate: `python scripts/security_audit.py --release-gate` — flag defined `scripts/security_audit.py:488`
- GUI install: `npm ci` — invoked `ci.yml:101,123,145`
- GUI lint: `npm run lint` (→ `eslint .`, `gui/package.json:18`) — **defined but not yet wired into CI** (Section 7)
- GUI typecheck: `npm run typecheck` (`gui/package.json:17`) — `ci.yml:104`
- GUI tests: `npm test` (→ `vitest run`, `gui/package.json:19`) — `ci.yml:126`
- GUI build: `npm run build` (→ `electron-vite build`, `gui/package.json:11`) — `ci.yml:148`
- Wheel contents smoke: `python scripts/check_wheel_contents.py` — `ci.yml:457`
- Electron package contents verification: `python scripts/verify_electron_package_contents.py gui/dist/win-unpacked/resources` — `.github/workflows/windows-electron-artifact.yml:71`
- Windows managed runtime build: `npm run runtime:build` (→ PowerShell `scripts/build_managed_python_runtime.ps1 -Profile Voice`, `gui/package.json:12`)
- Windows installer build: `npm run dist` (→ `electron-builder --publish never`, `gui/package.json:15`, auto-runs `runtime:build`+`build` via `predist`)
- Installed-artifact smoke: `scripts/test_installed_electron_artifact.ps1 -Installer "gui/dist/AskRex Setup 1.0.0.exe" -BuildPython python -DiagnosticsPath "$env:RUNNER_TEMP\askrex-artifact-smoke.json"` — `.github/workflows/windows-electron-artifact.yml:76-79`
- Doctor: `python -m rex doctor` and `python -m rex doctor --release-gate` — `rex/commands/core.py:204-207` (release-gate flag), `rex/commands/core.py:36` (passthrough)
- GUI raw-fetch guard: `python scripts/check_no_renderer_api_fetch.py` — `ci.yml:435`
- Workflows present: `.github/workflows/ci.yml` (main gate, push/PR to master), `windows-electron-artifact.yml` (PR path-filtered + tag push + workflow_dispatch/call), `commitlint.yml` (PR title/commit Conventional Commits), `release-please.yml` (push to master).

## 10. User-only blockers

- Physical Windows 11 hardware acceptance matrix — mic, speakers, wake-word,
  device restart/recovery (issue #299).
- Wake-word reliability/latency recordings across noise conditions and
  multiple speakers (issue #304).
- Physical iPhone + LAN testing for the mobile API gateway (issue #323,
  PR #326 description).
- Home Assistant real device control (locks, alarms, covers, lights) and
  stale-state/timeout behavior (audit finding C's remaining column).
- Authenticode signing certificate purchase and secret provisioning.
- External provider credentials for live email/SMS/search verification.
- Product-scope decisions: PR #326 disposition confirmation, OpenClaw gate
  timing (#302), 1.5.0 release timing (#322).

## 11. Known risks and disputed facts

- **PR #326 vs master duplication** (Section 6) is unusual enough that it
  should be confirmed with the repo owner before closing the PR, even though
  the git evidence is conclusive.
- The audit ledger (`docs/audits/AUDIT-REMEDIATION-2026-07-22.md:35`) frames
  findings A–K as "Locally verified" pending GitHub checks; this handoff's
  live CI query shows those checks are now green on master tip, so treat
  A–K as CI-verified — but if a future SHA regresses, re-check before
  reusing this claim.
- `requirements-cpu.txt:5`'s comment (Torch 2.6.0) contradicts its own pin
  (Torch 2.12.0, line 6) — flagged as a doc/comment bug, not a functional
  risk, but worth fixing before a Torch dependency bump (PR #327) is merged,
  to avoid confusion about which version is "verified."
- The 372 unchecked PRD boxes are **not** all genuinely outstanding — at
  least one spot-check (US-024) shows the PRD lagging shipped code. Do not
  treat the raw unchecked count as a work-remaining estimate without
  spot-verification.
- GUI lint-not-in-CI (Section 7) is confirmed by direct grep of
  `.github/workflows/ci.yml` for `npm run lint` / `eslint` — zero matches.

## 12. Recommended first Fable action

Start with the low-risk, high-confidence, quickly-verifiable items in the
execution queue (Section 8, items 1–5) before touching the larger PRD
P0-security backlog (item 6), which needs careful current-code verification
to avoid duplicating work that may already exist (as US-024 and the
confirmation-gate infrastructure already demonstrate). Update
`docs/release/FABLE-SHIPPING-PROGRESS.md` after each queue item lands.
