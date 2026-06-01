# PRD: AskRex Remaining Release Readiness Work

> **Ralph execution rule**
> A task means one full User Story, not one checkbox.
> Choose the first User Story whose acceptance criteria contain any unchecked `[ ]` box.
> Complete exactly one User Story per iteration.
> A User Story is only complete when current code, tests, and acceptance criteria prove it.
> When a story is complete, update this PRD and its progress log in the same commit as the implementation.
> Do not commit completed implementation work while leaving that story unchecked in this PRD.
> This PRD is the authoritative task tracker. `progress-remaining-release-readiness.txt` is supporting history only.

---

## Introduction

This PRD replaces the noisy reconciled `PRD.md` for remaining work only. The original `PRD.md`
accumulated checkbox drift because several stories were implemented by the Ralph agentic loop before
the PRD-update step was consistently enforced. Two reconciliation passes corrected the original
document, but it remains a historical artifact mixing completed and incomplete work.

**This document contains only stories that are not yet done**, verified against git commits, current
file state, and the progress log as of 2026-05-31 (HEAD: `a2ca126`). Do not modify `PRD.md` or
`progress.txt`. This PRD is the forward-looking tracker; those files are historical record only.

**US-RR-028 note:** The original PRD listed `rex_speak_api.py`'s `rex.shopping_pwa` import as an
unconditional dead import. Inspection of the current file shows the import is already wrapped in
`try/except Exception` with a graceful warning log. The acceptance criteria for that story are
satisfied. It is not included here.

**US-RR-034 note:** All eight negative security tests specified in US-RR-034 were created as part of
US-RR-006 through US-RR-012 (test files `test_rr007_setup_register_protection.py`,
`test_rr008_log_auth.py`, `test_rr009_ha_test_auth.py`, `test_rr010_ha_secret_required.py`,
`test_ph001_twilio_handler.py`, `test_us047_user_auth.py`). That story's criteria are met. It is
not included here.

---

## Completed-Work Summary (US-RR-001 through US-RR-012)

| Story | Title | Evidence Commit | Branch at time |
|-------|-------|-----------------|----------------|
| US-RR-001 | Verify clean repo state and map active surfaces | `0240da5` | verify-rr-001-006-cleanup |
| US-RR-002 | Confirm CI pipeline baseline | `5f41d61` | verify-rr-001-006-cleanup |
| US-RR-003 | Fix pytest collection failure (rex_calendar_bridge) | `83fb664` | verify-rr-001-006-cleanup |
| US-RR-004 | Invert Twilio fail-open test | `2ecf967` | verify-rr-001-006-cleanup |
| US-RR-005 | Confirm full test suite collects and passes | `75b9503` | verify-rr-001-006-cleanup |
| US-RR-006 | Remove hardcoded JWT fallback secret | `970a380` | verify-rr-001-006-cleanup |
| US-RR-007 | Protect /setup and /register routes | `7e89d8c` | verify-rr-001-006-cleanup |
| US-RR-008 | Protect log streaming/download routes | `e56c28f` | verify-rr-001-006-cleanup |
| US-RR-009 | Protect HA connection-test route | `f8640fd` | verify-rr-001-006-cleanup |
| US-RR-010 | Require HA_SECRET for HA blueprint | `90e39d7` | verify-rr-001-006-cleanup |
| US-RR-011 | Twilio fail-closed when package missing | `e008340` | verify-rr-001-006-cleanup |
| US-RR-012 | Add Twilio signature validation to voicemail route | `a2ca126` | verify-rr-001-006-cleanup |
| US-RR-028 | Fix dead shopping_pwa import | *(effectively complete — import already guarded with try/except Exception at rex_speak_api.py line 146)* | — |
| US-RR-034 | Add negative security tests | *(effectively complete — all eight criteria satisfied by tests from US-RR-006 through US-RR-012)* | — |

---

## Goals

- Pass `pip-audit` and `npm audit` dependency checks at the `high` severity level.
- Produce an Electron package that includes bridge scripts and proves a clean install can start.
- Gate CI on GUI typecheck, GUI build, Node audits, and an Electron package smoke test.
- Remove all tracked personal and demo runtime data from the repo.
- Classify every runtime/UI surface as shippable, developer-only, deprecated, archived, or removed.
- Achieve a release candidate that can safely enter public beta.

## Non-Goals

- No new features.
- No broad refactors before security and test gates are green (Phase 10 only).
- No new integrations.
- No UI redesign beyond what is required to make first-run safe.

---

## Phase 3 — Dependency Audit Remediation

*Pass `pip-audit` and `npm audit` at the `high` level or have narrow, documented, expiring suppressions for any accepted-risk items.*

---

### US-REM-001: Remediate Python audit failures for `idna`, `pip`, and `urllib3`

**Description:** As a maintainer, I want `idna`, `pip`, and `urllib3` upgraded to patched versions so `pip-audit` reports no known vulnerabilities for these packages.

**Why it matters:** Codex confirmed `pip-audit` fails with known CVEs in `idna` (CVE-2026-45409, fix: 3.15), `pip` (CVE-2026-3219 and CVE-2026-6357, fix: 26.1), and `urllib3` (PYSEC-2026-141 and PYSEC-2026-142, fix: 2.7.0). These are runtime dependencies with genuine exploit risk.

**Files/areas likely involved:**
- `pyproject.toml` (dependency version pins)
- `requirements-cpu.txt`, `requirements-gpu-cu124.txt`, `requirements-gpu.txt`, `requirements-dev.txt`
- `.github/workflows/ci.yml` (audit suppression list — remove these entries)
- `docs/security/VULNERABILITY-SCAN.md`

**Acceptance Criteria:**
- [x] `idna` is pinned to `>=3.15` in `pyproject.toml` and all requirements files that pin it.
- [x] `urllib3` is pinned to `>=2.7.0` in `pyproject.toml` and all requirements files that pin it.
- [x] `pip` is upgraded in CI install steps to `>=26.1`; if `pip` is a declared project dependency it is moved to a dev-only constraint.
- [x] Any suppression entries for `idna`, `pip`, and `urllib3` CVEs are removed from the CI audit suppression list in `.github/workflows/ci.yml`.
- [x] `python -m pip_audit 2>&1 | grep -E "idna|urllib3|pip.*CVE"` returns no matches.
- [x] `docs/security/VULNERABILITY-SCAN.md` is updated to reflect the reduced suppression count.
- [x] `pytest -q 2>&1 | tail -5` shows no new failures introduced by the version bumps.

**Validation commands:**
```bash
python -m pip_audit 2>&1 | grep -E "idna|urllib3|CVE-2026"
python -m pip_audit 2>&1 | tail -5
pytest -q 2>&1 | tail -10
```

**Risk notes:** Upgrading `urllib3` may break `requests` or other HTTP clients pinned to an older version. Run the full test suite after upgrading. If a transitive dependency forces an older `urllib3`, document the conflict and add a suppression with an owner and expiry date.

---

### US-REM-002: Document the `transformers` vulnerability and torch CUDA audit gap

**Description:** As a maintainer, I want the `transformers` CVE-2026-1839 finding and the torch CUDA pip-audit gap to have documented, expiring suppression entries with a named owner, so the CI audit does not silently mask real risk.

**Why it matters:** `transformers 4.57.6` has a known vulnerability whose stable fix (`5.0.0`) is not yet released. PyTorch CUDA builds are not audited by `pip-audit` because their wheel identities are not found on PyPI. Neither can be resolved by a simple upgrade, but both must be explicitly acknowledged.

**Files/areas likely involved:**
- `.github/workflows/ci.yml` (audit suppression entries)
- `docs/security/VULNERABILITY-SCAN.md`

**Acceptance Criteria:**
- [x] A suppression entry for `transformers` CVE-2026-1839 and PYSEC-2025-217 exists in the CI audit config with: owner name, date added, risk classification (`optional-ML-dependency`), rationale (stable fix not yet released), and expiry date no more than 90 days from story closure.
- [x] The suppression entry for `transformers` is in a separate, labeled section from runtime and dev suppressions.
- [x] A comment in `docs/security/VULNERABILITY-SCAN.md` explains the torch CUDA audit gap: why `pip-audit` cannot see CUDA wheel identities, and whether any upstream torch security advisories were consulted.
- [x] `python -m pip_audit 2>&1 | grep transformers` shows the known finding; the CI suppression is applied only in the audit CI step.
- [x] A GitHub issue or calendar reminder is created (or referenced in the comment) to revisit the `transformers` suppression when `transformers >= 5.0.0` stable ships.

**Validation commands:**
```bash
python -m pip_audit 2>&1 | grep -E "transformers|PYSEC-2025-217|CVE-2026-1839"
grep -A5 "transformers" docs/security/VULNERABILITY-SCAN.md
grep -A5 "torch\|cuda" docs/security/VULNERABILITY-SCAN.md
```

**Risk notes:** Do not suppress the `transformers` finding globally. Scope the suppression to the specific CVE IDs. If `transformers 5.0.0` stable ships before this story is implemented, upgrade instead of suppressing.

---

### US-REM-003: Remediate Node audit failures in `gui/`

**Description:** As a maintainer, I want the npm audit vulnerabilities in `gui/` resolved by upgrading affected packages or applying focused suppressions with documented rationale.

**Why it matters:** Codex confirmed `npm audit --audit-level=moderate` fails in `gui/` with 5 vulnerabilities (2 high, 3 moderate), including Electron and `tmp`. High-severity Electron vulnerabilities in a desktop app are a direct attack surface.

**Files/areas likely involved:**
- `gui/package.json`
- `gui/package-lock.json`

**Acceptance Criteria:**
- [x] `npm audit --audit-level=high` in `gui/` returns 0 high-severity vulnerabilities.
- [x] `npm audit --audit-level=moderate` in `gui/` returns 0 unmitigated moderate vulnerabilities; any accepted-risk moderates have documented entries in `.nsprc` or an npm audit allowlist with rationale and expiry.
- [x] Electron is upgraded to the latest stable LTS version that resolves the reported high-severity CVEs, unless a dependency incompatibility prevents this (document the blocker if so).
- [x] `tmp` is upgraded or replaced.
- [x] `npm ci && npm run typecheck` passes after the upgrade.
- [x] `npm run build` passes after the upgrade.

**Validation commands:**
```bash
cd gui && npm audit --audit-level=high
cd gui && npm audit --audit-level=moderate
cd gui && npm ci && npm run typecheck
cd gui && npm run build
```

**Risk notes:** Electron major version upgrades can introduce breaking IPC or preload API changes. Run GUI typecheck and build after any upgrade. If Electron cannot be upgraded due to a transitive dependency blocker, document the specific conflict and accept the risk with an expiry date.

---

### US-REM-004: Remediate Node audit failures in `rex/ui/`

**Description:** As a maintainer, I want the npm audit moderate vulnerabilities in `rex/ui/` (including Vite, esbuild, and PostCSS) resolved.

**Why it matters:** Even if `rex/ui/` is classified as developer-only, vulnerable build tooling in the repo creates supply-chain risk and CI noise.

**Files/areas likely involved:**
- `rex/ui/package.json`
- `rex/ui/package-lock.json`

**Acceptance Criteria:**
- [x] `npm audit --audit-level=moderate` in `rex/ui/` returns 0 unmitigated vulnerabilities.
- [x] Vite, esbuild, and PostCSS are upgraded to patched versions if available.
- [x] If `rex/ui/` is to be deprecated (Phase 7 classification), a clear comment is added to `rex/ui/package.json`: "Developer-only surface. Not included in packaged Electron app."
- [x] `npm ci && npm run build` in `rex/ui/` still succeeds after upgrades.

**Validation commands:**
```bash
cd rex/ui && npm audit --audit-level=moderate
cd rex/ui && npm ci && npm run build
```

**Risk notes:** Do not skip this story on the assumption `rex/ui/` will be deleted. Deletion must follow Phase 7 classification, not precede it.

---

### US-REM-005: Restructure CI audit suppressions with owners, expiry, and risk tiers

**Description:** As a maintainer, I want the CI `pip-audit` suppression list restructured so each entry has a named owner, risk tier, rationale, and expiry date, so the list does not silently accumulate accepted-risk items without accountability.

**Why it matters:** CI has 85 `--ignore-vuln` entries in `.github/workflows/ci.yml` with no per-entry expiry or tier. A large flat suppression list is indistinguishable from "we stopped caring."

**Files/areas likely involved:**
- `docs/security/VULNERABILITY-SCAN.md`
- `.github/workflows/ci.yml` (audit step configuration)

**Acceptance Criteria:**
- [x] The suppression documentation in `docs/security/VULNERABILITY-SCAN.md` is restructured into clearly labeled sections: `## Runtime dependencies`, `## Dev-only dependencies`, `## Optional ML/AI dependencies`.
- [x] Each documented suppression entry includes: CVE/PYSEC ID, package name and version range, owner (GitHub handle), date added, expiry date, rationale (one sentence), and risk tier.
- [x] Suppressions older than 12 months with no expiry date are reviewed; those still valid get an explicit expiry date, those no longer needed are removed from the CI suppression list.
- [x] A comment at the top of the suppression section warns: "If your suppression has no expiry date it will be removed at next review."
- [x] The total suppression count is documented as a number in `VULNERABILITY-SCAN.md`.
- [x] `pip-audit 2>&1 | tail -5` still exits 0 after restructuring (no functional change to CI behavior, only documentation and cleanup).

**Validation commands:**
```bash
grep -c "expiry\|expires\|owner" docs/security/VULNERABILITY-SCAN.md
python -m pip_audit 2>&1 | tail -5
```

**Risk notes:** Do not use this restructuring pass to remove legitimate suppressions. The goal is accountability, not a smaller list for its own sake.

---

## Phase 4 — Electron Packaging and Bridge/Runtime Inclusion

*Produce an Electron package that includes bridge scripts and proves a clean install can start and use the bridge.*

---

### US-REM-006: Audit Electron package config against required bridge/runtime files

**Description:** As a maintainer, I want a clear documented map of every file that `bridgeResolver.ts` expects at runtime in a packaged app vs. what `gui/package.json` actually packages, so the gap is known before any fix is applied.

**Why it matters:** Current state: `gui/package.json` `extraResources` includes only `assets/brand`; `bridgeResolver.ts` resolves 20 bridge scripts relative to `app.getAppPath()/../bridge/`. This gap is confirmed — bridge scripts are not in the packaged output.

**Files/areas likely involved:**
- `gui/package.json` (`extraResources` config)
- `gui/src/main/bridgeResolver.ts` (runtime path resolution and `BRIDGE_REGISTRY`)
- `bridge/*.py` (the scripts being resolved)
- `gui/src/main/index.ts` (any other bridge spawn calls)

**Acceptance Criteria:**
- [x] `gui/src/main/bridgeResolver.ts` is read in full; every bridge script name in `BRIDGE_REGISTRY` is listed.
- [x] `gui/package.json` `extraResources` section is read; the list of files/directories currently included is documented.
- [x] The gap (bridges resolved but not packaged) is listed explicitly as a note or comment in this commit.
- [x] Whether a Python runtime (interpreter) is expected on the user's PATH or is bundled is documented.
- [x] No files are changed in this story — output is a gap list that US-REM-007 will act on.

**Validation commands:**
```bash
cat gui/package.json | grep -A20 "extraResources"
grep -n "BRIDGE_REGISTRY\|resolveBridgePath\|getAppPath\|resourcesPath" gui/src/main/bridgeResolver.ts
ls bridge/*.py | wc -l
```

**Risk notes:** Check all bridge spawn calls in `gui/src/main/index.ts`, not just `bridgeResolver.ts`. There may be additional files (config, venv) that must also be packaged.

---

### US-REM-007: Fix `gui/package.json` `extraResources` to include bridge scripts

**Description:** As a maintainer, I want `gui/package.json` to include the `bridge/*.py` scripts (and any other required runtime files identified in US-REM-006) in `extraResources`, so the packaged Electron app contains the files it needs to run.

**Why it matters:** Without this fix a packaged install cannot start the Python bridge. This is the most direct cause of Codex's deployment score of 25/100.

**Files/areas likely involved:**
- `gui/package.json` (`extraResources` and `files` config)
- `gui/electron-builder.config.js` or `gui/electron-builder.yml` if present

**Acceptance Criteria:**
- [x] `gui/package.json` `extraResources` includes a glob or explicit list covering all bridge scripts identified in US-REM-006.
- [x] `npm run build` in `gui/` completes without error.
- [x] The packaged output directory (e.g., `gui/dist` or `gui/release`) contains `bridge/*.py` at the expected `extraResources` path — verified with `find gui/dist -name "*.py"` or equivalent.
- [x] No required bridge file is absent from the packaged output.

**Validation commands:**
```bash
cd gui && npm run build
find gui/dist -name "*.py" 2>/dev/null | head -30
find gui/release -name "*.py" 2>/dev/null | head -30
```

**Risk notes:** If the packaged app requires a Python interpreter on the user's PATH, document this requirement clearly in the installer README. If a bundled Python runtime is needed for a zero-dependency install, scope that as a separate follow-up story.

---

### US-REM-008: Fix `bridgeResolver.ts` path resolution for packaged app context

**Description:** As a maintainer, I want `gui/src/main/bridgeResolver.ts` to correctly resolve bridge script paths in both dev mode (relative to source tree) and packaged mode (relative to `process.resourcesPath`), so the bridge works in both contexts.

**Why it matters:** The current `resolveBridgePath` uses `app.getAppPath()/../bridge/`. In a packaged `.asar`, `app.getAppPath()` resolves inside the archive — the `../bridge/` path does not exist there. Even after US-REM-007 puts bridge scripts in `extraResources`, the path logic must be fixed to point to `process.resourcesPath`.

**Files/areas likely involved:**
- `gui/src/main/bridgeResolver.ts` (`resolveBridgePath` function)
- `gui/src/main/index.ts` (any other bridge path construction)

**Acceptance Criteria:**
- [x] `resolveBridgePath` checks `app.isPackaged` and uses `process.resourcesPath` for the bridge path in packaged mode.
- [x] In dev mode (not packaged), the resolver continues to use a path relative to the source tree (e.g., `app.getAppPath()/../bridge/`).
- [x] A comment explains both branches of the path resolution logic.
- [x] `npm run typecheck` in `gui/` returns no errors after the change.
- [x] The resolved path is logged at bridge startup so it can be inspected in packaged app logs.
- [x] `grep -n "isPackaged\|resourcesPath" gui/src/main/bridgeResolver.ts` returns matches for both.

**Validation commands:**
```bash
cd gui && npm run typecheck
grep -n "isPackaged\|resourcesPath\|getAppPath" gui/src/main/bridgeResolver.ts
```

**Risk notes:** If `app.isPackaged` is not available at the call site, use `process.defaultApp` as a fallback. Test both code paths in US-REM-009's smoke test.

---

### US-REM-009: Add a package smoke test proving clean install can start Electron and use the bridge

**Description:** As a maintainer, I want an automated smoke test that builds the Electron app from the packaged output and confirms the Python bridge is reachable, so packaging regressions are caught automatically.

**Why it matters:** Without a smoke test, packaging regressions (like the missing bridge scripts) are only caught by manual testing. This story provides the automated verification path for all Phase 4 fixes.

**Files/areas likely involved:**
- `gui/` (package output)
- New test script: `tests/smoke/test_electron_package.sh` or `gui/tests/smoke.cjs`
- `docs/claude/TESTING_AND_QUALITY.md` (documentation)

**Acceptance Criteria:**
- [x] A smoke test script exists at a documented path (e.g., `tests/smoke/test_electron_package.sh`).
- [x] The test: (1) builds the Electron package, (2) launches the packaged app in headless or minimal mode, (3) sends a bridge health-check request or waits for a startup signal, (4) asserts the bridge responded successfully, (5) exits the app cleanly.
- [x] The test exits non-zero if the bridge is unreachable within a timeout.
- [x] The test is documented in `docs/claude/TESTING_AND_QUALITY.md` under "Package Smoke Tests."
- [x] Running the smoke test locally on a clean Python environment (without the source-tree `bridge/` on PATH) passes.

**Validation commands:**
```bash
bash tests/smoke/test_electron_package.sh
```

**Risk notes:** Headless Electron testing on Linux CI requires a virtual display (`xvfb`). Document the CI display dependency. On Windows, code-signing may be required for packaged app launch; the smoke test may need `--no-sandbox` or an unsigned build flag in CI.

---

## Phase 5 — CI/Release Gate Hardening

*Ensure CI gates every component of the shippable product.*

---

### US-REM-010: Add GUI TypeScript typecheck gate to CI

**Description:** As a maintainer, I want CI to run `npm run typecheck` in `gui/` on every PR and merge to `master`, so TypeScript type errors in the Electron app are caught before they ship.

**Why it matters:** The GUI typecheck passes locally but is not gated in CI. Type errors introduced on a branch will not be caught until a developer runs the check manually.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`
- `gui/package.json` (typecheck script)

**Acceptance Criteria:**
- [x] `.github/workflows/ci.yml` has a job or step that runs `cd gui && npm ci && npm run typecheck`.
- [x] The step runs on every push to `master` and on every pull request targeting `master`.
- [x] A failing typecheck returns a non-zero exit code and fails the CI run.
- [x] The job is clearly named (e.g., `gui-typecheck`).
- [x] The current codebase passes this gate — no pre-existing type errors are hidden by adding it.

**Validation commands:**
```bash
cd gui && npm ci && npm run typecheck
```

**Risk notes:** If pre-existing type errors are found, fix them in this story before adding the gate. Do not add a gate that immediately fails.

---

### US-REM-011: Add GUI build gate to CI

**Description:** As a maintainer, I want CI to run `npm run build` in `gui/` on every PR and merge to `master`, so build-breaking changes are caught before release.

**Why it matters:** A TypeScript typecheck can pass while the bundler still fails. The build step validates the full compile and bundling pipeline.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`
- `gui/package.json` (build script)

**Acceptance Criteria:**
- [x] `.github/workflows/ci.yml` has a step that runs `cd gui && npm ci && npm run build`.
- [x] The step runs on every push to `master` and on every pull request.
- [x] Build artifacts are not uploaded unless on a release tag.
- [x] A failing build returns a non-zero exit code and fails the CI run.

**Validation commands:**
```bash
cd gui && npm ci && npm run build
```

**Risk notes:** Document any required `VITE_*` or `ELECTRON_*` env vars needed for a successful CI build and set them as CI secrets or well-documented defaults.

---

### US-REM-012: Add Node dependency audit gates to CI for `gui/` and `rex/ui/`

**Description:** As a maintainer, I want CI to run `npm audit --audit-level=high` in both `gui/` and `rex/ui/` on every PR, so high-severity Node vulnerabilities are caught automatically.

**Why it matters:** Both Node package directories have failing audits. Without a CI gate, new vulnerabilities introduced by dependency updates will go undetected.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`

**Acceptance Criteria:**
- [x] CI runs `cd gui && npm audit --audit-level=high` and fails if any high-severity vulnerabilities are present.
- [x] CI runs `cd rex/ui && npm audit --audit-level=high` and fails if any high-severity vulnerabilities are present.
- [x] The audit steps run after US-REM-003 and US-REM-004 remediation, so CI starts green for this gate.

**Validation commands:**
```bash
cd gui && npm audit --audit-level=high
cd rex/ui && npm audit --audit-level=high
```

**Risk notes:** Use `--audit-level=high` not `--audit-level=info` to avoid false failures from informational entries.

---

### US-REM-013: Add Electron package smoke test gate to CI

**Description:** As a maintainer, I want CI to run the Electron package smoke test from US-REM-009 on release tags and on PRs touching `gui/` or `bridge/`, so packaging regressions are caught automatically.

**Why it matters:** The smoke test is only useful if it runs automatically. Without a CI gate, a packaging regression can reach a release.

**Files/areas likely involved:**
- `.github/workflows/ci.yml`
- `.github/workflows/release.yml` (if it exists)
- Smoke test script from US-REM-009

**Acceptance Criteria:**
- [x] CI runs the package smoke test when: (a) a release tag is pushed, or (b) files in `gui/` or `bridge/` are changed in a PR.
- [x] The smoke test step fails CI if the bridge is unreachable.
- [x] The CI job documents any virtual display (`xvfb`) or platform requirements.
- [x] CI passes a full green run after all Phase 4 fixes are applied.

**Validation commands:**
```bash
bash tests/smoke/test_electron_package.sh
```

**Risk notes:** Run the smoke test only on path-filtered triggers (not on every Python-only commit) to avoid slowing unrelated PRs.

---

### US-REM-014: Expand Python lint and type gate from changed-files-only to full repo

**Description:** As a maintainer, I want CI to run `ruff check` and `mypy` across all Python files in the repo, not just changed files, so type errors introduced anywhere in the codebase are caught.

**Why it matters:** CI currently lints only changed files (`.github/workflows/ci.yml` line 58). A commit that adds a type error to an unchanged file will not be caught. The mypy config also excludes 12 core modules with `ignore_errors = true`.

**Files/areas likely involved:**
- `.github/workflows/ci.yml` (lint step)
- `pyproject.toml` (mypy `exclude` list)

**Acceptance Criteria:**
- [x] CI lint step runs `ruff check .` on all Python files, not just `git diff` output.
- [x] CI format step runs `black --check .` on all Python files.
- [x] The mypy `exclude` list in `pyproject.toml` is reduced: at minimum `rex/cli.py`, `rex/voice_loop.py`, and `rex/gui_app.py` are either included in mypy checking or have their exclusion documented with a ticket reference and a remediation plan.
- [x] CI passes a full green run after this change — any pre-existing lint errors found by full-repo scan are fixed before the gate is activated.
- [x] The lint and type steps complete in under 5 minutes.

**Validation commands:**
```bash
ruff check .
black --check .
mypy rex/ --ignore-missing-imports 2>&1 | tail -20
```

**Risk notes:** Enabling full-repo lint may surface pre-existing issues. Fix them in this story or document them as follow-up items. Do not disable the gate to hide pre-existing errors.

---

## Phase 6 — Runtime/Personal Data Cleanup and Secret/Config Consolidation

*Remove all tracked personal and demo runtime data. Establish one authoritative config/secret path.*

---

### US-REM-015: Remove tracked personal and demo runtime data from the repo

**Description:** As a maintainer, I want `users.json`, `Memory/james/*`, `Memory/cole/*`, `profiles/james.json`, and any other personal data removed from Git and replaced with sanitized examples, so the repo does not leak personal information.

**Why it matters:** `git ls-files` currently shows: `Memory/cole/core.json`, `Memory/cole/history.log`, `Memory/cole/notes.md`, `Memory/james/core.json`, `Memory/james/history.log`, `Memory/james/notes.md`, `profiles/james.json`, `profiles/james.example.json`, `users.json`. These are personal runtime files that must not be in a public repo.

**Files/areas likely involved:**
- `users.json`
- `Memory/james/*`, `Memory/cole/*`
- `profiles/james.json` (keep `profiles/james.example.json` as a template)
- `profiles/default.json` (check for personal content; keep `profiles/default.example.json`)
- `.gitignore` (must be updated to exclude runtime data)

**Acceptance Criteria:**
- [x] `users.json`, `Memory/james/*`, `Memory/cole/*`, and `profiles/james.json` are removed from the repo (both from disk and from `git ls-files`).
- [x] `profiles/default.json` is inspected; if it contains personal content it is replaced with a sanitized example using clearly fictional data.
- [x] `.gitignore` is updated to exclude `users.json`, `Memory/james/`, `Memory/cole/`, and `profiles/*.json` except `profiles/*.example.json` and `profiles/profile.schema.json`.
- [x] `Memory/README.md` is updated (or created) to explain that memory profile files are generated at runtime and should not be committed.
- [x] `git ls-files Memory/ profiles/james.json users.json` returns only `Memory/README.md` and `profiles/*.example.json` and `profiles/profile.schema.json` — no personal data.
- [x] The commit message explicitly states: "Remove tracked personal/demo runtime data."

**Validation commands:**
```bash
git ls-files Memory/ profiles/ users.json
grep -r "james\|cole" Memory/ profiles/ 2>/dev/null | grep -v ".example."
cat .gitignore | grep -E "Memory|profiles|users\.json"
```

**Risk notes:** If a Git history rewrite is required to expunge previously committed personal data from history, coordinate with the repo owner and document the process. At minimum, the files must be removed from the working tree and `.gitignore` updated so they are never re-committed.

---

### US-REM-016: Consolidate secret storage to one protected authority

**Description:** As a maintainer, I want secrets (JWT secret, HA token, Twilio credentials) to be stored in one protected location per deployment, not duplicated across `.env`, `config/rex_config.json`, and GUI settings, so secrets cannot drift and the leakage surface is minimized.

**Why it matters:** The HA token is mirrored in both `.env` and GUI settings. Config sprawl means a secret rotation in one location may not propagate to another, leaving stale credentials in use.

**Files/areas likely involved:**
- `rex/config.py` (config loading)
- `rex/auth.py` (JWT secret loading)
- `rex/ha_bridge.py` (HA token loading)
- `rex/telephony/twilio_handler.py` (Twilio credential loading)
- `config/rex_config.json` (must contain no secrets after this story)
- `gui/src/main/index.ts` (GUI settings that store credentials)
- `docs/claude/CONFIG_AND_SECURITY.md`

**Acceptance Criteria:**
- [x] A single canonical secret-loading path is defined and documented: secrets come from `.env` only — never from `rex_config.json` or GUI settings files. (OS keyring is documented as a future option; `.env` is the current canonical path.)
- [x] `config/rex_config.json` contains no secrets or credentials.
- [x] GUI settings do not store or mirror secrets; when HA credentials are needed they are retrieved from the canonical secret store at runtime.
- [x] The migration path for existing users who have secrets in `rex_config.json` or GUI settings is documented in `docs/claude/CONFIG_AND_SECURITY.md`.
- [x] `grep -rn "ha_token\|HA_TOKEN\|jwt.*secret\|twilio.*auth" config/rex_config.json` returns no matches.

**Validation commands:**
```bash
grep -rn "ha_token\|HA_TOKEN\|jwt_secret\|twilio.*auth" config/ 2>/dev/null
python -c "from rex.config import AppConfig; c = AppConfig(); print('ok')"
```

**Risk notes:** OS keyring access requires `pip install keyring` and platform-specific backend libraries. If keyring is not available, fall back to `.env` only. Document the fallback clearly.

---

### US-REM-017: Audit and document the complete config authority chain

**Description:** As a maintainer, I want a documented map of every configuration key, where it is read from, and which source takes precedence, so there is one authoritative reference for config behavior.

**Why it matters:** Config spread across `.env`, `config/rex_config.json`, profiles, GUI settings, and environment variables makes it impossible to reason about effective configuration at runtime.

**Files/areas likely involved:**
- `rex/config.py`
- `docs/claude/CONFIG_AND_SECURITY.md`
- `config/rex_config.json`
- `.env.example`

**Acceptance Criteria:**
- [x] `docs/claude/CONFIG_AND_SECURITY.md` contains a table listing every `AppConfig` field, its source priority (env > config JSON > default), and whether it is a secret, a runtime setting, or an optional feature flag.
- [x] Any config key that currently has conflicting sources is resolved to one winner with documented precedence.
- [x] The `AppConfig` sub-config access pattern from `CLAUDE.md` is reflected in the doc (all seven sub-config objects: `audio`, `voice`, `llm`, `tools`, `integrations`, `ui`, `security`).
- [x] The doc warns against adding new flat top-level `AppConfig` fields (per `CLAUDE.md`).

**Validation commands:**
```bash
python -c "from rex.config import AppConfig; import json; c = AppConfig(); print([f for f in dir(c) if not f.startswith('_')])"
grep -c "config\." docs/claude/CONFIG_AND_SECURITY.md
```

**Risk notes:** This is a documentation story, not a code change. Do not refactor `AppConfig` here; document the current behavior accurately.

---

## Phase 7 — Release Surface Consolidation and Legacy Classification

*Classify every runtime/UI surface. Disable or hide non-shipping surfaces from the packaged Electron app.*

---

### US-REM-018: Classify every runtime/UI surface as shippable, developer-only, deprecated, archived, or removed

**Description:** As a product owner, I want every entry point and UI surface in the repo to have an explicit classification so that packaging, docs, CI, and support scope are clear.

**Why it matters:** Without explicit classification, every surface is implicitly treated as release-critical, multiplying the security surface, packaging complexity, and support burden.

**Files/areas likely involved:**
- `pyproject.toml` (entry points: `rex`, `rex-gui`, `rex-config`, `rex-speak-api`, `rex-agent`, `rex-tool-server`)
- `gui/` (Electron app)
- `rex/gui_app.py` (Flask GUI/API)
- `rex/ui/` (alternative UI)
- `archived/` (retired code)
- Root-level compatibility shims (`voice_loop.py`, `llm_client.py`, `config.py`)
- `CLAUDE.md` (must be updated to reference classification)

**Acceptance Criteria:**
- [ ] A `SURFACE-CLASSIFICATION.md` is created at the repo root with a table covering every entry point and UI surface.
- [ ] Each surface is assigned exactly one of: `shippable`, `developer-only`, `deprecated`, `archived`, or `removed`.
- [ ] The Electron app (`gui/`) is classified as `shippable`.
- [ ] `rex/ui/` is classified (based on code inspection — do not decide without reading the code).
- [ ] `archived/` content is classified as `archived`.
- [ ] Root-level compatibility shims are classified as `developer-only` or `deprecated`.
- [ ] `CLAUDE.md` is updated to reference `SURFACE-CLASSIFICATION.md`.

**Validation commands:**
```bash
cat SURFACE-CLASSIFICATION.md
grep -c "shippable\|developer-only\|deprecated\|archived\|removed" SURFACE-CLASSIFICATION.md
```

**Risk notes:** Classification decisions drive packaging (Phase 4), CI (Phase 5), and docs (Phase 9). Make decisions based on actual code inspection, not assumptions. If a surface's status is genuinely unclear, classify it as `developer-only` until a human owner decides.

---

### US-REM-019: Disable non-shipping Flask GUI dashboard from packaged Electron app

**Description:** As a maintainer, I want the Flask GUI/API dashboard (`rex-gui`, `rex/gui_app.py`) to not be started automatically inside the packaged Electron app unless it is classified as `shippable`, so users of the Electron app are not exposed to the Flask API surface.

**Why it matters:** If the Electron app spawns `rex-gui` as a subprocess, all Flask routes are reachable from within the packaged app. If the Flask dashboard is developer-only, it must not run in the packaged context.

**Files/areas likely involved:**
- `gui/src/main/index.ts` (any subprocess spawning of `rex-gui` or Flask)
- `bridge/*.py` (bridge scripts may start Flask)
- `rex/gui_app.py`

**Acceptance Criteria:**
- [ ] `gui/src/main/index.ts` and all bridge scripts are audited for any subprocess spawn of `rex-gui`, `flask`, or `rex/gui_app.py`.
- [ ] If the packaged app spawns the Flask GUI, a feature flag or build-time exclude is added so it does not spawn in packaged mode unless `rex-gui` is explicitly classified as `shippable`.
- [ ] The smoke test from US-REM-009 confirms the Flask GUI routes are not reachable from the packaged app unless explicitly enabled.
- [ ] `SURFACE-CLASSIFICATION.md` is updated with the final decision for `rex-gui`.

**Validation commands:**
```bash
grep -n "rex-gui\|gui_app\|flask\|subprocess" gui/src/main/index.ts bridge/*.py
```

**Risk notes:** If the Flask GUI is the primary API backend for the Electron renderer (i.e., the renderer calls Flask REST endpoints), then `rex/gui_app.py` is shippable and must remain — but all unauthenticated routes from Phase 2 must still be fixed (they were fixed in US-RR-007 through US-RR-012).

---

### US-REM-020: Add shim comments to root-level compatibility files and clean Tkinter references from docs

**Description:** As a maintainer, I want root-level compatibility shim files to have module-level deprecation comments, and all Tkinter/legacy GUI references in docs to be cleaned up.

**Why it matters:** Root-level shims (`voice_loop.py`, `llm_client.py`, `config.py`) add confusion for developers. Tkinter references in docs like `README.md`, `ARCHITECTURE.md`, `INSTRUCTION_MANUAL.md`, and `docs/dashboard.md` mislead users into expecting deprecated UI paths.

**Files/areas likely involved:**
- Root `voice_loop.py`, `llm_client.py`, `config.py`
- `README.md`, `docs/ARCHITECTURE.md`, `docs/INSTRUCTION_MANUAL.md`, `docs/dashboard.md`

**Acceptance Criteria:**
- [ ] Each root-level shim file (`voice_loop.py`, `llm_client.py`, `config.py`) has a module-level comment: "Compatibility shim. Canonical implementation: rex.<module>. Scheduled for removal — see SURFACE-CLASSIFICATION.md."
- [ ] All references to Tkinter UI that are not already marked deprecated in the text are updated to deprecation notices or removed.
- [ ] `grep -rn "tkinter\|Tkinter" docs/ README.md --include="*.md"` returns results only in the context of explicit deprecation notices, not as active instructions.
- [ ] `CLAUDE.md` is updated if the shim documentation there is stale.

**Validation commands:**
```bash
grep -rn "tkinter\|Tkinter" docs/ README.md --include="*.md"
grep -n "Compatibility shim\|Scheduled for removal" voice_loop.py llm_client.py config.py
```

**Risk notes:** Do not delete the shim files in this story. Deletion must follow after classification (US-REM-018) confirms they are no longer needed. This story only adds comments and cleans docs.

---

## Phase 8 — First-Run and Recovery Tests

*Prove the first-run experience is safe, the reset flow works, and config migration does not lose data.*

---

### US-REM-021: Add first-run setup flow test file

**Description:** As a maintainer, I want a dedicated `tests/test_first_run.py` that tests the end-to-end first-run scenario in a clean state (no pre-existing users), setup endpoint completion, and post-setup authentication.

**Why it matters:** The existing setup and auth tests (from US-RR-006, US-RR-007) test individual security boundaries, not the full first-run flow from a clean state. A dedicated first-run test catches regressions in the overall onboarding path.

**Note:** The criteria about "generated JWT secret stored locally" do not apply to the current implementation, which raises `RuntimeError` when `REX_JWT_SECRET` is unset (not generates-and-stores). Tests must use `monkeypatch` to set a test JWT secret.

**Files/areas likely involved:**
- `tests/test_first_run.py` (new file)
- `rex/gui_app.py` (setup route)
- `rex/auth.py` (JWT handling)

**Acceptance Criteria:**
- [ ] `tests/test_first_run.py` exists.
- [ ] Test: On a clean state (no users in the data store, `REX_JWT_SECRET` set via monkeypatch), the first-run setup endpoint completes successfully and creates a user.
- [ ] Test: After setup, the created user can authenticate and receive a valid JWT.
- [ ] Test: A second attempt to call the setup endpoint after a user exists returns 403 (token consumed).
- [ ] Test: Attempting to call setup with an invalid or missing setup token from a clean state returns 401 or 403.
- [ ] All tests use `tmp_path` or equivalent fixtures — no test writes to real `users.json` or `config/` paths.
- [ ] `pytest tests/test_first_run.py -q` passes.
- [ ] `pytest --collect-only -q 2>&1 | grep first_run` shows tests collected.

**Validation commands:**
```bash
pytest tests/test_first_run.py -q -v
pytest --collect-only -q 2>&1 | grep first_run
```

**Risk notes:** Tests that touch `users.json` or secret paths must use `tmp_path` fixtures to avoid polluting the real user's data. Use `monkeypatch` to redirect file paths to temporary directories.

---

### US-REM-022: Add config migration and reset/recovery tests

**Description:** As a maintainer, I want automated tests covering config migration (upgrading from an older `rex_config.json` schema) and reset/recovery (missing, corrupt, or old-version config), so config schema changes cannot silently break existing installs.

**Files/areas likely involved:**
- `rex/config.py` (migration logic — add if absent)
- `rex-config` CLI entry point (`rex.config:cli`)
- `tests/test_config_migration.py` (new file)

**Acceptance Criteria:**
- [ ] `tests/test_config_migration.py` exists.
- [ ] Test: Loading a `rex_config.json` with a missing new required field either migrates gracefully with defaults or raises a clear `ConfigError`, not an unhandled `KeyError`.
- [ ] Test: A corrupt `rex_config.json` (invalid JSON) results in a `ConfigError` with a helpful message pointing to the file path.
- [ ] Test: A missing `rex_config.json` results in defaults being applied, not a crash.
- [ ] If no migration logic exists in `rex/config.py`, this story adds it (at minimum: missing keys → safe defaults; corrupt file → clear error; missing file → defaults).
- [ ] `pytest tests/test_config_migration.py -q` passes.

**Validation commands:**
```bash
pytest tests/test_config_migration.py -q -v
python -m rex config --help 2>&1
```

**Risk notes:** If `AppConfig` already handles all these cases gracefully (Pydantic defaults cover missing keys), confirm by reading `rex/config.py` before adding new migration logic.

---

## Phase 9 — Documentation Truth Pass

*Ensure docs reflect the actual supported install/run path and do not contain misleading or dangerous instructions.*

---

### US-REM-023: Update README to declare one supported install/run path and demote others

**Description:** As a maintainer, I want the README to clearly declare the Electron + Python bridge as the one supported user-facing install path, with all other paths demoted to a clearly separated "Advanced / Developer" section.

**Why it matters:** The README lists 8+ entry points without a clear hierarchy. `change-me` placeholder secrets appear in user-facing setup instructions at README.md lines 305, 312, 321, 331, 338.

**Files/areas likely involved:**
- `README.md`
- `INSTALL.md`

**Acceptance Criteria:**
- [ ] The README has a prominent "Getting Started" section describing only the Electron app install path.
- [ ] All other runtime paths (CLI, voice loop, Flask dashboard, TTS API, Windows agent, OpenClaw tool server) are in a collapsible or clearly separated "Advanced / Developer" section.
- [ ] The `change-me` placeholder values in README.md (lines 305, 312, 321, 331, 338) are replaced with generation commands or bracketed placeholders (`<YOUR-API-KEY>`).
- [ ] The alpha warning is preserved.
- [ ] Known limitations (wake-word latency, Outlook partial, per-user isolation incomplete) are preserved.
- [ ] If `INSTALL.md` describes a different primary path, it is updated to match the README.

**Validation commands:**
```bash
grep -n "change-me\|CHANGE_ME\|your-secret-here" README.md INSTALL.md
grep -n "Getting Started\|Quick Start" README.md | head -5
```

**Risk notes:** Do not remove honest alpha/limitation notices. The goal is clarity about the supported path, not false marketing.

---

### US-REM-024: Remove or replace placeholder secrets in all documentation

**Description:** As a maintainer, I want all placeholder secrets in documentation and example configs to be replaced with clearly fictional values or generation instructions, so users do not accidentally copy placeholder values into production configs.

**Why it matters:** `docs/api.md` lines 136 and 172 contain `change-me` in user-facing setup commands. `docs/security/SECRET-SCAN.md` documents a `changeme` example value at line 56.

**Files/areas likely involved:**
- `docs/api.md` (lines 136, 172)
- `docs/security/SECRET-SCAN.md` (line 56 — review whether it needs updating)
- `.env.example`
- `config/rex_config.json` (check for any example secrets)

**Acceptance Criteria:**
- [ ] Every user-facing doc that shows a secret value either: (a) shows a generation command, or (b) uses a clearly bracketed placeholder (`<YOUR_STRONG_SECRET_HERE>`), not `change-me` or `changeme`.
- [ ] `.env.example` has generation commands for `REX_JWT_SECRET` and any other required secrets.
- [ ] `grep -rn "change-me\|CHANGE_ME\|changeme" docs/ .env.example` returns no matches in user-facing install instructions (developer-only docs may retain `change-me` with an explicit developer scope warning).
- [ ] `grep -n "REX_JWT_SECRET" .env.example` returns a match with a generation command.

**Validation commands:**
```bash
grep -rn "change-me\|CHANGE_ME\|changeme" docs/ .env.example
grep -n "REX_JWT_SECRET" .env.example
```

**Risk notes:** Do not remove example config files. Only fix the placeholder values inside them.

---

### US-REM-025: Update docs to reflect surface classification and deprecation decisions

**Description:** As a maintainer, I want all docs (README, INSTALL, integration guides, CLAUDE.md) to reflect the surface classification decisions from US-REM-018, so deprecated or developer-only surfaces are no longer presented as primary options.

**Why it matters:** Docs that contradict classification decisions create confusion and cause users to attempt unsupported paths.

**Files/areas likely involved:**
- `README.md`
- `docs/` (all files)
- `CLAUDE.md`
- `SURFACE-CLASSIFICATION.md` (created in US-REM-018)

**Acceptance Criteria:**
- [ ] Every surface classified as `deprecated` in `SURFACE-CLASSIFICATION.md` has a deprecation notice in any doc that references it.
- [ ] Every surface classified as `developer-only` is in a Developer section of the relevant doc, not in user-facing Getting Started.
- [ ] Every surface classified as `archived` is mentioned only in `archived/ARCHIVED.md`, not in primary docs.
- [ ] `CLAUDE.md` Tech Stack section matches the final surface classification.
- [ ] Docs do not contradict each other on which path is primary.

**Validation commands:**
```bash
grep -rn "rex-gui\|rex/ui\|archived" README.md docs/ | grep -v "archived/ARCHIVED.md"
cat SURFACE-CLASSIFICATION.md
```

**Risk notes:** Documentation-only story. Do not change code. Do not delete docs without confirming no other doc links to them.

---

## Phase 10 — Post-Release Technical Debt Cleanup

> **Prerequisite:** Do not begin any Phase 10 story until the full pytest suite passes, all P0/P1 security fixes are merged, CI is green on all gates, and a release candidate has been tagged or is within one sprint of tagging.

---

### US-REM-026: Decompose `rex/gui_app.py` by route domain

**Description:** As a maintainer, I want `rex/gui_app.py` (52 KB, mixed-concern) decomposed into route-domain Blueprint modules (e.g., `rex/routes/auth.py`, `rex/routes/ha.py`, `rex/routes/logs.py`, `rex/routes/setup.py`), so the file is reviewable and changes are isolated to their domain.

**Files/areas likely involved:**
- `rex/gui_app.py`
- New `rex/routes/` package

**Acceptance Criteria:**
- [ ] `rex/gui_app.py` is under 200 lines after extraction (app factory, middleware registration, blueprint registration only).
- [ ] Each route domain has its own Blueprint module in `rex/routes/`.
- [ ] All existing tests pass without modification.
- [ ] `ruff check rex/gui_app.py rex/routes/` passes.
- [ ] No behavior change: the route table before and after decomposition is identical (verify with a route snapshot before and after).

**Validation commands:**
```bash
wc -l rex/gui_app.py
pytest tests/ -q
ruff check rex/gui_app.py rex/routes/
```

**Risk notes:** Highest-risk decomposition story. Do it last, with the full test suite green. Use `git diff` to confirm no route paths, methods, or decorators changed during extraction.

---

### US-REM-027: Decompose `rex/cli.py` by command domain

**Description:** As a maintainer, I want `rex/cli.py` (182 KB) decomposed into focused command modules so individual CLI commands can be reviewed, tested, and modified in isolation.

**Files/areas likely involved:**
- `rex/cli.py`
- New `rex/commands/` package (or equivalent)

**Acceptance Criteria:**
- [ ] `rex/cli.py` is under 300 lines after extraction (Click group registration only).
- [ ] Each command domain has its own module.
- [ ] All CLI commands work identically after decomposition.
- [ ] `pytest tests/ -q` passes.
- [ ] `rex --help` and `rex <subcommand> --help` outputs are unchanged.

**Validation commands:**
```bash
wc -l rex/cli.py
rex --help
pytest tests/ -q
```

---

### US-REM-028: Decompose `rex/voice_loop.py` by concern

**Description:** As a maintainer, I want `rex/voice_loop.py` (128 KB) decomposed into focused modules (wake-word handling, STT, TTS, LLM routing, session state) so the voice pipeline can be individually tested and modified.

**Files/areas likely involved:**
- `rex/voice_loop.py`
- `rex/wakeword/` (already exists — integrate)

**Acceptance Criteria:**
- [ ] `rex/voice_loop.py` is under 200 lines after extraction.
- [ ] `pytest tests/ -q` passes.
- [ ] `python -c "from rex.voice_loop import build_voice_loop; print('ok')"` succeeds.

**Validation commands:**
```bash
wc -l rex/voice_loop.py
pytest tests/ -q
python -c "from rex.voice_loop import build_voice_loop; print('ok')"
```

**Risk notes:** Per `CLAUDE.md`: the canonical wake-word implementation is `rex/wakeword/`. Do not re-introduce root-level shim behavior during decomposition.

---

### US-REM-029: Decompose `gui/src/main/index.ts` by concern

**Description:** As a maintainer, I want `gui/src/main/index.ts` (1500+ lines) decomposed into focused modules (window management, IPC handlers, bridge lifecycle, integration setup) so Electron main process code is maintainable.

**Files/areas likely involved:**
- `gui/src/main/index.ts`
- New `gui/src/main/` submodules

**Acceptance Criteria:**
- [ ] `gui/src/main/index.ts` is under 200 lines after extraction.
- [ ] `npm run typecheck` in `gui/` passes.
- [ ] `npm run build` in `gui/` produces a valid build.
- [ ] The smoke test from US-REM-009 passes.

**Validation commands:**
```bash
wc -l gui/src/main/index.ts
cd gui && npm run typecheck && npm run build
bash tests/smoke/test_electron_package.sh
```

**Risk notes:** IPC handler decomposition must preserve all channel names and argument shapes. Any IPC channel name change will break the renderer.

---

### US-REM-030: Remove broad mypy core-module exclusions and fix resulting type errors

**Description:** As a maintainer, I want the mypy `ignore_errors = true` entries for core modules in `pyproject.toml` removed one module at a time, with newly surfaced type errors fixed, so the type coverage gate is meaningful.

**Why it matters:** `pyproject.toml` currently excludes 12 modules with `ignore_errors = true`. A type gate that excludes the most complex modules provides false assurance.

**Files/areas likely involved:**
- `pyproject.toml` (mypy exclude list)
- Core modules: `rex/cli.py`, `rex/voice_loop.py`, `rex/gui_app.py`, and the other 9 listed modules

**Acceptance Criteria:**
- [ ] Each excluded core module is re-enabled in mypy one at a time.
- [ ] All type errors surfaced by re-enabling each module are fixed (not suppressed with `type: ignore` unless a third-party library requires it).
- [ ] `mypy rex/ --ignore-missing-imports` returns 0 errors after all core modules are re-enabled.
- [ ] CI mypy step (from US-REM-014) passes with the expanded scope.

**Validation commands:**
```bash
mypy rex/ --ignore-missing-imports 2>&1 | grep "error:" | wc -l
mypy rex/cli.py rex/voice_loop.py rex/gui_app.py --ignore-missing-imports 2>&1 | tail -20
```

**Risk notes:** Giant modules must be decomposed (US-REM-026 through US-REM-029) before their type errors are tractable. Do not begin this story until decomposition is complete.

---

## Definition of Done

The following checklist must be fully satisfied before any public release is cut.

### Test Suite
- [ ] All user stories US-REM-001 through US-REM-025 are checked `[x]`.
- [ ] `pytest --collect-only -q` completes with 0 errors.
- [ ] `pytest -q` passes with 0 failures on a clean checkout with only base dependencies installed.
- [ ] First-run setup flow tests (US-REM-021) pass.
- [ ] Config migration tests (US-REM-022) pass.

### Dependency Audits
- [ ] `python -m pip_audit` returns 0 runtime vulnerabilities, or all remaining findings have narrow suppression entries with owner, rationale, risk tier, and expiry date.
- [ ] `npm audit --audit-level=high` in `gui/` returns 0 high-severity vulnerabilities.
- [ ] `npm audit --audit-level=high` in `rex/ui/` returns 0 high-severity vulnerabilities.

### Electron Packaging
- [ ] The smoke test from US-REM-009 passes on a clean machine (no source-tree `bridge/` on PATH).
- [ ] `find gui/dist -name "*.py"` returns bridge scripts in the packaged output.
- [ ] `bridgeResolver.ts` uses `process.resourcesPath` in packaged mode.

### CI Gates
- [ ] CI runs `ruff check .` on all Python files and fails on errors.
- [ ] CI runs `npm run typecheck` in `gui/` and fails on errors.
- [ ] CI runs `npm run build` in `gui/` and fails on errors.
- [ ] CI runs `npm audit --audit-level=high` in `gui/` and `rex/ui/` and fails on high-severity findings.
- [x] CI runs the Electron package smoke test on PRs touching `gui/` or `bridge/`.

### Data and Secrets
- [x] `git ls-files Memory/james/ Memory/cole/ profiles/james.json users.json` returns no results.
- [x] `.gitignore` excludes `users.json`, `Memory/james/`, `Memory/cole/`, and non-example profiles.
- [ ] `grep -rn "change-me\|CHANGE_ME\|changeme" README.md INSTALL.md .env.example docs/` returns no matches in user-facing install instructions.
- [ ] `config/rex_config.json` contains no secrets or credentials.

### Surface Consolidation
- [ ] `SURFACE-CLASSIFICATION.md` exists and classifies every entry point and UI surface.
- [ ] The packaged Electron app does not start the Flask GUI dashboard unless it is classified as `shippable`.
- [ ] README has one primary Getting Started section pointing to the Electron app.
- [ ] All deprecated surfaces have deprecation notices in their docs.

---

*This PRD covers remaining work only. The completed-work record is in `progress-remaining-release-readiness.txt` and the original `PRD.md`. Phase 10 stories (US-REM-026 through US-REM-030) must not begin until a release candidate is within one sprint.*

*Created: 2026-05-31. HEAD at creation: `a2ca126` (branch: verify-rr-001-006-cleanup).*
