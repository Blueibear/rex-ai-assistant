# PRD: AskRex Assistant — Consolidation, Trust, and Production Readiness

> **Codex/Ralph task selection rule**
> A "task" means one full User Story (US-###), not an individual checkbox line.
> Choose the first US-### that contains any unchecked acceptance criteria `[ ]`.
> Complete the full story in one iteration. If it cannot be completed in one iteration, split it first.
> Only mark acceptance criteria `[x]` when the full story is done and tests pass.
> **Phase priority: consolidation stories (WS-A through WS-H) must be completed before
> feature stories (Phase K and the legacy backlog in the appendix).**

---

## 1. Title

AskRex Assistant — Consolidation, Trust, and Production Readiness

---

## 2. Executive Summary

The AskRex Assistant repository contains working code, real integrations, and a substantial
completed implementation backlog. However, it is not production-ready. The blockers are not
missing features — they are fragmented identity, broken automation, misleading documentation,
an unclear UI story, a cluttered repository root, and accumulated architectural sediment.

This PRD covers cleanup, consolidation, and truth. It does not add new capabilities.
Every story in this document exists to make the repo easier to trust, navigate, and maintain.

---

## 3. Problem Statement

The repository currently:

- Uses at least two product names ("Rex AI Assistant" and "AskRex") inconsistently across
  docs, package metadata, and repository URLs. The canonical name is **AskRex Assistant**.
- Has CI running on `master` and release automation targeting `main` — two different branches
  that do not overlap, making automated releases silently broken.
- Has a README that markets features as "beta" that are either real working backends or
  incomplete stubs, with no reliable signal to distinguish them.
- Contains four separate UI entry points (Tkinter, web/React, Flask API, CLI) with no
  documentation about which is the primary user experience.
- Has 20+ root-level utility scripts, 500+ KB of progress log files, unresolved patch files,
  and backup `.env` files all committed to git.
- Has a CI vulnerability scan that suppresses 103 CVEs — with at least one duplicate — and
  no mechanism to re-evaluate suppressions over time.
- Has entry points in `pyproject.toml` that target modules inconsistently (some root-level,
  some package-level) with no CI verification that they actually work.

---

## 4. Why This Work Matters Now

Feature development is blocked by the above. Specifically:

- Contributors cannot be confident their code reaches a release because the release pipeline
  is disconnected from the branch where CI runs.
- Users cannot trust documentation because it mixes aspirational and real claims.
- The four UI surfaces create maintenance ambiguity — every feature must be considered for
  each surface or silently neglected.
- Root clutter signals to any reviewer that the codebase is in a transitional state, not a
  production-ready one.

This work must happen before the next feature cycle.

---

## 5. Goals

- Establish **AskRex Assistant** as the single canonical product name everywhere in the repo.
- Ensure CI and release automation target the same branch and produce verified releases.
- Ensure documentation describes what is true today, not what may be true later.
- Establish one canonical UI entry point; classify all others as keep/deprecate/archive/remove.
- Clean the root directory to contain only production-relevant files.
- Ensure every declared entry point is tested in CI.
- Remove or justify every CVE suppression in the vulnerability scan workflow.
- Ensure security and dependency docs are accurate and not placeholder-heavy.
- Produce a repo that a new contributor can clone, read, and understand in under 30 minutes.

---

## 6. Non-Goals

This PRD explicitly does not cover:

- Adding new features, integrations, or capabilities.
- Adding new UI surfaces or design changes.
- OAuth calendar backends (Google, Microsoft).
- LLM streaming (tracked in the legacy backlog appendix; must wait for consolidation).
- Multi-user or RBAC support.
- GPU-specific CI runners.
- Mobile apps.
- Any story that was already marked `[x]` complete in the previous cycle.

---

## 7. Current-State Findings

All findings below are based on direct file inspection. Uncertain items are marked
**(NEEDS VERIFICATION)**.

### 7.1 Product Naming

| Naming Variant | Locations | Status |
|---|---|---|
| "Rex AI Assistant" | `README.md:1`, `pyproject.toml:6`, `INSTALL.md:1`, all security docs | Wrong — must change to AskRex Assistant |
| "askrex-assistant" | `README.md:33` (clone URL), badge URL in `README.md:3` | Wrong — stale GitHub slug |
| "rex-ai-assistant" | `pyproject.toml:6` (project name field), `release-please.yml:15` (package-name) | Wrong — must update |
| "AskRex" | GitHub repo URL `github.com/Blueibear/AskRex-Assistant` (visible in CI run URL) | Partial — correct repo slug, wrong casing/suffix |
| "AskRex Assistant" | Nowhere currently | **Target canonical name** |

`README.md:3` badge URL points to `github.com/Blueibear/askrex-assistant` (old slug).
`pyproject.toml:146` Homepage points to `github.com/Blueibear/rex-ai-assistant` (different slug).
Neither matches the actual repo URL `github.com/Blueibear/AskRex-Assistant`.

### 7.2 Branch and Automation

- `.github/workflows/ci.yml:5` — triggers on `branches: [master, claude/**]`
- `.github/workflows/ci.yml:7` — pull_request targets `master`
- `.github/workflows/release-please.yml:4` — triggers on `branches: [ main ]`
- These branches do not overlap. CI never triggers a release. Releases never run on CI-tested code.
- `release-please.yml:15` — `package-name: rex-ai-assistant` (stale name, will generate wrong tags)
- **(NEEDS VERIFICATION):** Confirm actual GitHub default branch (`master` or `main`).

### 7.3 Documentation Drift

- `pyproject.toml:17` — `Development Status :: 3 - Alpha`
- `README.md` — features listed without "alpha" caveat; some listed as "beta stub" are real backends
- `INTEGRATIONS_STATUS.md:30` — states real IMAP/SMTP backend exists
- `README.md:93` — states email is "stub/mock data only" — contradicts INTEGRATIONS_STATUS
- `README.md:96` — lists "Autonomous workflows with planner" as a feature; `PRD.md` (previous cycle) shows this as a roadmap item, not complete
- `README.md:79` — states Python 3.12 is unsupported but does not say `pyproject.toml` blocks it
- `pyproject.toml:10` — `requires-python = ">=3.11,<3.12"` — 3.12 is hard-blocked, not just discouraged

### 7.4 UI Surfaces

| Surface | Entry Point | Framework | Classification |
|---|---|---|---|
| CLI (text chat) | `rex` → `rex.cli:main` | None | **Keep — primary non-voice surface** |
| Voice loop | `python rex_loop.py` | None | **Keep — primary voice surface** |
| Web dashboard | `rex-gui` → `rex.gui_app:main` | Flask + React (pre-built in `rex/ui/dist/`) | **Keep — canonical GUI** |
| Tkinter window | `python run_gui.py` → `gui.py` | Tkinter | **Deprecate** |
| Shopping PWA | Blueprint in `rex/shopping_pwa.py` | Flask | **Keep — optional feature surface** |
| TTS API | `rex-speak-api` → `rex_speak_api:main` | Flask | **Keep — service component** |

`run_gui.py:6` claims to be "the canonical way to launch the Rex GUI on Windows" but launches
Tkinter (`gui.py:1`). The actual canonical GUI is `rex-gui` (web/React). This is a direct contradiction.

`rex/ui/dist/` contains pre-built React assets. This is the correct modern GUI.

### 7.5 Root-Level Sediment

Files at repo root that are not production entry points:

**Utility/check scripts (should move to `scripts/` or be removed):**
- `check_gpu_status.py`, `check_imports.py`, `check_patch_status.py`, `check_tts_imports.py`
- `find_gpt2_model.py`, `generate_wake_sound.py`, `list_audio.py`, `list_voices.py`
- `manual_search_demo.py`, `manual_whisper_demo.py`, `play_test.py`, `record_wakeword.py`
- `test_imports.py`, `test_mic_open.py`, `test_transformers_patch.py`, `wake_acknowledgment.py`
- `wakeword_listener.py`, `wakeword_utils.py` (likely duplicate of `rex/wakeword/`)

**Patch files (must be applied or deleted):**
- `ci-fixes.patch` (210,156 bytes) — committed but not applied; status unknown

**Progress/audit logs (must be archived or deleted):**
- `progress.txt`, `progress-master-next-cycle.txt` (181 KB), `progress-openclaw-pivot-for-rex.txt`
- `progress-gui-autonomy-integrations.txt`, `progress-full-repo-audit.txt`
- `progress-full-test-and-fix.txt`, `progress-voice-selector-and-fixes.txt`
- `progress-openclaw-pivot.txt`, `progress-ci-fix-pr216.txt`, `progress-repo-quality.txt`
- `progress-openclaw-http-integration.txt`
- Total: ~600+ KB of progress notes committed to git

**Generated artifacts (must be in .gitignore):**
- `.coverage` — generated by `pytest --cov`; should never be committed
- `coverage.txt`, `test-audit-coverage.txt`, `test-audit-final-results.txt`

**Backup files (must be removed):**
- `.env.backup-legacy`, `.env.example.backup_before_refactor`
- `backups/` directory

**Security advisory (belongs in `docs/security/`):**
- `SECURITY_ADVISORY.md` at root (13,227 bytes)

**Backward-compatibility shims (must be evaluated):**
- `setup.py` — exists to expose root-level py_modules for backward compat (`setup.py:13–23`)
- Root-level `config.py`, `llm_client.py`, `memory_utils.py`, `logging_utils.py`
  listed in `setup.py` — unclear if any external code depends on these

### 7.6 Entry Point Alignment

Declared in `pyproject.toml:137–143`:
- `rex` → `rex.cli:main` — verified
- `rex-config` → `rex.config:cli` — **unverified**; `gui.py:35` imports `rex.config_manager`, not `rex.config`
- `rex-speak-api` → `rex_speak_api:main` — verified (root-level module in `setup.py`)
- `rex-agent` → `rex.computers.agent_server:main` — **unverified**
- `rex-gui` → `rex.gui_app:main` — verified (React/web dashboard)
- `rex-tool-server` → `rex.openclaw.tool_server:main` — **unverified**

CI only tests `python -m rex --help` (`ci.yml:156`). All other entry points are untested.

### 7.7 CVE Suppression

`.github/workflows/ci.yml:217–304` suppresses 103 CVEs with `--ignore-vuln` flags.
- `CVE-2026-4539` appears at both line 272 and line 304 (duplicate).
- `docs/security/VULNERABILITY-SCAN.md` (9,647 bytes) is unlikely to contain per-item
  justifications for all 103 entries.
- No mechanism exists to re-evaluate suppressions when packages are updated.

### 7.8 Security and Dependency Docs

- `docs/security/SECURITY_AUDIT_2026-01-08.md` — internal audit doc; reports 30 "findings"
  but all flagged as legitimate (false positives). Useful internally but potentially
  confusing for external contributors.
- `SECURITY_ADVISORY.md` at root — belongs in `docs/security/`.
- `docs/security/VULNERABILITY-SCAN.md` — last modified Mar 27 (one week before audit);
  may not cover CVEs added after that date.
- **(NEEDS VERIFICATION):** Confirm whether all 103 CVE suppressions have written justifications.

---

## 8. Scope by Workstream

| ID | Workstream | Stories |
|---|---|---|
| WS-A | Product identity (naming) | US-230 – US-234 |
| WS-B | Branch and release automation | US-235 – US-237 |
| WS-C | Documentation truth | US-238 – US-243 |
| WS-D | UI consolidation | US-244 – US-247 |
| WS-E | Root directory hygiene | US-248 – US-253 |
| WS-F | Entry point correctness | US-254 – US-256 |
| WS-G | CVE and security doc cleanup | US-257 – US-260 |
| WS-H | Active CI failures (current run) | US-261 – US-269 |
| WS-I | Brand asset integration | US-270 – US-273 |
| APPENDIX | Legacy feature backlog (previous cycle) | US-175 – US-229 |

---

## 9. Risks and Constraints

| Risk | Mitigation |
|---|---|
| Renaming package breaks existing installs | Provide migration note; keep `rex` CLI alias temporarily |
| Moving root-level py_modules breaks external imports | Usage-check before removal; keep `setup.py` until verified safe |
| Branch rename breaks open PRs or forks | Check open PRs before renaming; announce in CHANGELOG |
| Deleting progress files loses history | Archive to `docs/archive/` before deleting from root |
| Removing Tkinter GUI breaks undocumented user workflows | Mark deprecated for one cycle; do not hard-delete |
| CVE suppression removal triggers CI failures | Remove suppressions one at a time; document expected outcome |
| ci-fixes.patch content is unknown | Read and apply or discard before closing the story |

---

## 10. Phase Plan

```
Phase 1 (WS-A):  Product identity           — no dependencies; do first
Phase 2 (WS-B):  Branch/release fix          — depends on Phase 1 (name in release-please)
Phase 3 (WS-C):  Documentation truth         — depends on Phase 1 (canonical name needed)
Phase 4 (WS-D):  UI consolidation            — depends on Phase 3 (docs must be ready)
Phase 5 (WS-E):  Root hygiene               — depends on Phase 4 (classify root files last)
Phase 6 (WS-F):  Entry point correctness     — depends on Phase 5 (root modules may change)
Phase 7 (WS-G):  Security doc cleanup        — no blocking dependencies; can run after Phase 3
Phase 8 (WS-H):  Active CI failures          — no blocking dependencies; run in parallel
Phase 9 (WS-I):  Brand assets               — depends on Phase 1 (canonical name confirmed)
THEN: Legacy feature backlog (US-175–US-229) — run only after Phases 1–9 are complete
```

---

## 11. Atomic Implementation Backlog

---

# WORKSTREAM A — Product Identity (Naming)

### US-230: Define canonical product name and create naming ruleset

**Description:** As a developer, I need one authoritative document that defines the canonical
product name and where each name variant is allowed or forbidden, so that all subsequent
stories apply consistent naming.

**Acceptance Criteria:**
- [x] `docs/BRANDING.md` is created and contains:
  - Canonical product name: **AskRex Assistant**
  - Canonical package name (pip): `askrex-assistant`
  - Canonical CLI command: `askrex` (or `rex` as an alias — define which)
  - Canonical GitHub repo URL: `https://github.com/Blueibear/AskRex-Assistant`
  - Allowed legacy alias table: where `rex` CLI name may remain and where it must change
  - A "banned names" table listing `Rex AI Assistant`, `askrex-assistant` (as repo slug),
    `rex-ai-assistant` (as package name) and their replacement
- [x] `CLAUDE.md` Project Overview section updated to reference `docs/BRANDING.md`
- [x] Typecheck passes (no new errors introduced; pre-existing mypy errors are tracked separately)

---

### US-231: Update pyproject.toml and setup.py with canonical name and metadata

**Description:** As a developer, I want `pyproject.toml` to reflect the canonical product
name, package name, and correct repository URL so that pip installs, PyPI listings,
and tooling all show consistent identity.

**Acceptance Criteria:**
- [x] `pyproject.toml:6` — `name = "askrex-assistant"` (was `rex-ai-assistant`)
- [x] `pyproject.toml` — `description` field updated to reference "AskRex Assistant"
- [x] `pyproject.toml:146` — `Homepage` URL set to `https://github.com/Blueibear/AskRex-Assistant`
- [x] `pyproject.toml` — any `Repository`, `Source`, or `Bug Tracker` URLs updated to match
- [x] `setup.py` — any `name=` or `url=` fields updated to match
- [x] `pip install -e .` succeeds after the change
- [x] `pip show askrex-assistant` returns the correct metadata
- [x] Typecheck passes

---

### US-232: Update README.md, INSTALL.md, and CHANGELOG.md with canonical name

**Description:** As a user reading the repository, I want all top-level user-facing documents
to use the canonical product name so that there is no confusion about what I am installing.

**Acceptance Criteria:**
- [x] `README.md:1` — title updated to `# AskRex Assistant`
- [x] `README.md:3` — all badge URLs updated to use `https://github.com/Blueibear/AskRex-Assistant`
- [x] `README.md:33` — clone URL updated to `https://github.com/Blueibear/AskRex-Assistant.git`
- [x] `INSTALL.md:1` — title updated to reference AskRex Assistant
- [x] `CHANGELOG.md` — top entry notes the rename from Rex AI Assistant to AskRex Assistant
  with the effective date
- [x] `grep -r "Rex AI Assistant" --include="*.md" .` returns zero results
  (except historical CHANGELOG entries, which are exempt)
- [x] `grep -r "askrex-assistant" --include="*.md" .` returns zero results
  (except `docs/BRANDING.md` banned-names table)
- [x] Typecheck passes

---

### US-233: Update all docs/ subdirectory references to canonical name

**Description:** As a developer, I want all documentation under `docs/` to consistently
use AskRex Assistant so that internal docs and CLAUDE reference files do not contradict
the public docs.

**Acceptance Criteria:**
- [x] `grep -r "Rex AI Assistant" docs/` returns zero results
  (CHANGELOG and historical audit docs in `docs/archive/` are exempt)
- [x] `docs/claude/` reference files updated (COMMANDS_AND_ENTRYPOINTS.md, etc.)
- [x] `docs/security/SECURITY_AUDIT_2026-01-08.md` product name references updated
- [x] `CLAUDE.md` updated throughout to use canonical name
- [x] Typecheck passes

---

### US-234: Update release-please and CI workflow name references

**Description:** As a developer, I want the `release-please.yml` package name to match the
canonical package name so that release tags and changelog entries are correctly attributed.

**Acceptance Criteria:**
- [x] `.github/workflows/release-please.yml:15` — `package-name: askrex-assistant`
  (was `rex-ai-assistant`)
- [x] Any workflow step that echoes or logs the product name uses "AskRex Assistant"
- [x] `grep -r "rex-ai-assistant" .github/` returns zero results after change
- [x] Typecheck passes

---

# WORKSTREAM B — Branch and Release Automation

### US-235: Determine canonical branch and document branch strategy

**Description:** As a developer, I need to know the one canonical primary branch so that CI,
release automation, and developer instructions all agree.

**Acceptance Criteria:**
- [x] Inspect actual GitHub default branch **(run: `gh repo view --json defaultBranchRef`)** and
  document the result
- [x] `CONTRIBUTING.md` documents:
  - The canonical primary branch name
  - The branching model (feature branches from primary, PRs back to primary)
  - That `claude/**` branches are AI-generated and follow the same PR process
- [x] `CLAUDE.md` updated to state the canonical branch name
- [x] Typecheck passes

---

### US-236: Align CI workflow to canonical branch

**Description:** As a developer, I want `ci.yml` to trigger on the canonical primary branch
so that CI runs on every merge to the branch that matters.

**Acceptance Criteria:**
- [x] `.github/workflows/ci.yml:5` — `branches:` list contains the canonical branch name
  (verified from US-235)
- [x] `.github/workflows/ci.yml:7` — `pull_request: branches:` list updated to match
- [x] `claude/**` branch trigger is retained (AI-generated PRs should still run CI)
- [x] Push a test commit to the canonical branch and confirm CI triggers **(manual verification)**
- [x] Typecheck passes

---

### US-237: Align release-please workflow to canonical branch

**Description:** As a developer, I want `release-please.yml` to trigger on the same canonical
branch as CI so that only CI-verified code produces releases.

**Acceptance Criteria:**
- [x] `.github/workflows/release-please.yml:4` — `branches:` updated to canonical branch name
- [x] `release-please.yml` — any hardcoded branch references updated
- [x] After this change, a merge to the canonical branch triggers both CI and release-please
  (verify by inspection — do not actually publish a release)
- [x] Typecheck passes

---

# WORKSTREAM C — Documentation Truth

### US-238: Rewrite README feature list to reflect actual implementation state

**Description:** As a user, I want the README feature list to distinguish between features
that work today, features that require configuration, and features that are in progress,
so that I can set accurate expectations before installing.

**Acceptance Criteria:**
- [x] Every bullet in the README feature list is classified with one of:
  `[Works today]`, `[Requires configuration]`, or `[In progress — not production ready]`
- [x] "Autonomous workflows with planner" is marked `[In progress]` (per `PRD.md` roadmap)
- [x] "Conversation history persistence" is marked `[In progress]`
- [x] Email integration is marked `[Requires configuration — IMAP/SMTP credentials needed]`
  (not "stub/mock data only" since real backend exists per `INTEGRATIONS_STATUS.md:30`)
- [x] SMS / Twilio is marked `[Requires configuration — Twilio credentials needed]`
- [x] "Smart notifications" description reflects what the dashboard store actually does
- [x] `grep "stub/mock data only" README.md` returns zero results
- [x] `grep "Autonomous workflows" README.md` is followed by a `[In progress]` annotation
- [x] `pyproject.toml:17` classifier is `Development Status :: 3 - Alpha`; README reflects this
- [x] Typecheck passes

---

### US-239: Fix Python version documentation across all user-facing files

**Description:** As a user, I want install documentation to clearly state that Python 3.11
is required and Python 3.12+ is not supported, so that I do not waste time on a failing
install with an unhelpful error message.

**Acceptance Criteria:**
- [x] `README.md` contains an explicit "Requirements" or "Prerequisites" section stating:
  "Python 3.11 is required. Python 3.12 and above are not supported."
- [x] `INSTALL.md` contains the same explicit statement
- [x] `pyproject.toml:10` — `requires-python = ">=3.11,<3.12"` — unchanged (already correct)
- [x] Any guide or doc that previously said "Python 3.11" without the 3.12 prohibition is updated
- [x] Typecheck passes

---

### US-240: Align INSTALL.md startup commands with pyproject.toml entry points

**Description:** As a user following `INSTALL.md`, I want every startup command shown
to correspond to a real, working entry point or script, so that I can actually run
the software after installing it.

**Acceptance Criteria:**
- [x] `INSTALL.md` lists the four supported startup modes:
  1. Text chat: `askrex` (or `python -m rex` — per canonical CLI name from US-231)
  2. Voice loop: `python rex_loop.py`
  3. Web dashboard: `askrex-gui` (or whatever the canonical entry point resolves to after US-231)
  4. TTS API: `askrex-speak-api` (or equivalent)
- [x] No startup command in `INSTALL.md` references `python run_gui.py` as a primary path
  (it is deprecated per WS-D)
- [x] Each command includes a one-line description of what it launches
- [x] Typecheck passes

---

### US-241: Correct INTEGRATIONS_STATUS.md to be the single source of truth

**Description:** As a developer, I want `docs/claude/INTEGRATIONS_STATUS.md` to be the
definitive status reference for all integrations and for README to defer to it, so that
integration status is maintained in one place.

**Acceptance Criteria:**
- [x] `INTEGRATIONS_STATUS.md` covers every integration listed in README with one of:
  `REAL`, `STUB`, `PARTIAL`, or `NOT STARTED` classification and a one-line evidence note
- [x] `README.md` integration section links to `INTEGRATIONS_STATUS.md` rather than
  embedding its own status claims
- [x] `INTEGRATIONS_STATUS.md` removes advisory language like "do NOT imply"
  (that instruction belongs in `CLAUDE.md`, not in a status file)
- [x] Typecheck passes

---

### US-242: Remove or correct aspirational language from all top-level docs

**Description:** As a developer, I want all documentation under `docs/claude/` and at the
repo root to describe the current state, not desired future state, so that CLAUDE and
developers make implementation decisions based on reality.

**Acceptance Criteria:**
- [x] `grep -rn "production.ready\|production-ready" docs/` — every match is reviewed;
  any that overstates current maturity is changed to reflect the `Alpha` status
- [x] `COMMANDS_AND_ENTRYPOINTS.md` only describes commands that currently work
- [x] Any doc section marked "TODO", "TBD", or "coming soon" is either completed or
  explicitly moved to a roadmap doc
- [x] Typecheck passes

---

### US-243: Move SECURITY_ADVISORY.md to docs/security/

**Description:** As a developer browsing the root, I do not want to see a security advisory
file in the root directory — it belongs under `docs/security/`.

**Acceptance Criteria:**
- [x] `SECURITY_ADVISORY.md` is moved to `docs/security/SECURITY_ADVISORY.md`
- [x] `README.md` or `SECURITY.md` (GitHub standard) links to
  `docs/security/SECURITY_ADVISORY.md` if previously linked to root version
- [x] Root no longer contains `SECURITY_ADVISORY.md`
- [x] Git history is preserved (`git mv`, not delete+create)
- [x] Typecheck passes

---

# WORKSTREAM D — UI Consolidation

### US-244: Formally classify all UI surfaces and update documentation

**Description:** As a contributor, I want one authoritative list of UI surfaces with their
classification (keep/deprecate/archive/remove) so that maintenance expectations are clear.

**Acceptance Criteria:**
- [x] `docs/UI_SURFACES.md` is created with this table:

  | Surface | Entry point | Status | Reason |
  |---|---|---|---|
  | CLI (text chat) | `askrex` | **Primary — keep** | Core text interface |
  | Voice loop | `python rex_loop.py` | **Primary — keep** | Core voice interface |
  | Web dashboard | `askrex-gui` | **Primary GUI — keep** | React, modern, canonical |
  | Shopping PWA | served by `askrex` or `askrex-gui` | **Optional feature — keep** | Functional feature surface |
  | TTS API | `askrex-speak-api` | **Service component — keep** | Required by voice loop |
  | Tkinter window (`gui.py`) | `python run_gui.py` | **Deprecated** | Superseded by web dashboard |

- [x] `README.md` Quick Start section points users to the web dashboard (`askrex-gui`)
  as the canonical GUI, not `python run_gui.py`
- [x] Typecheck passes

---

### US-245: Deprecate run_gui.py and gui.py; update startup documentation

**Description:** As a developer, I want `run_gui.py` and `gui.py` to be clearly marked
deprecated so that no new code references them and users are not directed to them.

**Acceptance Criteria:**
- [x] `run_gui.py:1–5` — add deprecation header:
  ```python
  # DEPRECATED: Use `askrex-gui` (web dashboard) instead.
  # This Tkinter launcher will be removed in the next major release.
  # See docs/UI_SURFACES.md for the canonical GUI entry point.
  ```
- [x] `gui.py:1–5` — same deprecation header
- [x] `run_gui.py` and `gui.py` are NOT deleted (they are deprecated for one cycle)
- [x] `README.md` no longer mentions `python run_gui.py` as a setup step
- [x] `INSTALL.md` no longer mentions `python run_gui.py` as a setup step
- [x] `grep "run_gui.py" README.md INSTALL.md` returns zero results
- [x] Typecheck passes

---

### US-246: Verify web dashboard entry point works end-to-end

**Description:** As a developer, I want to confirm that `askrex-gui` (`rex.gui_app:main`)
actually launches the React dashboard and serves `rex/ui/dist/index.html` correctly, so
that users directed to this entry point get a working experience.

**Acceptance Criteria:**
- [x] `rex/gui_app.py` is read and the Flask serve path for `rex/ui/dist/` is confirmed
- [x] `rex/ui/dist/index.html` exists and is a valid HTML file
- [x] `python -c "from rex.gui_app import main; print('ok')"` exits 0
- [x] A smoke test: `timeout 5 python -m rex.gui_app &` followed by
  `curl -s http://localhost:<PORT> | grep -i html` returns a non-empty response
  (or the test is added to `tests/test_gui_app.py` as a pytest fixture)
- [x] `INSTALL.md` documents the default port and how to change it
- [x] Typecheck passes

---

### US-247: Update CLAUDE.md to state canonical UI and remove Tkinter references

**Description:** As an AI agent working in this repo, I want `CLAUDE.md` to name the
canonical GUI entry point so that I do not generate code that references the deprecated
Tkinter surface.

**Acceptance Criteria:**
- [x] `CLAUDE.md` "Core components" section states:
  "GUI: Web dashboard via `rex.gui_app` (React + Flask). `run_gui.py` / `gui.py` are deprecated."
- [x] `CLAUDE.md` entry points section lists `askrex-gui` with the correct target
- [x] `grep "run_gui\|tkinter\|Tkinter" CLAUDE.md` returns zero results
- [x] Typecheck passes

---

# WORKSTREAM E — Root Directory Hygiene

### US-248: Archive all progress-*.txt files from root

**Description:** As a developer cloning the repo, I do not want to see 600+ KB of
implementation progress logs in the root directory. These are internal history and
belong in `docs/archive/` or should not be tracked at all.

**Acceptance Criteria:**
- [x] `docs/archive/progress/` directory is created
- [x] All `progress-*.txt` files (including `progress.txt`) are moved there with `git mv`
- [x] Root contains zero `progress*.txt` files
- [x] `.gitignore` adds `progress*.txt` so future progress files are not tracked by default
  (existing archived files are already committed and exempt)
- [x] Typecheck passes

---

### US-249: Remove generated artifacts from root and update .gitignore

**Description:** As a developer, I want generated test and coverage artifacts to not be
committed to git so that `git status` stays clean after running tests.

**Acceptance Criteria:**
- [x] `.coverage` is removed from git tracking: `git rm --cached .coverage`
- [x] `coverage.txt`, `test-audit-coverage.txt`, `test-audit-final-results.txt` are removed
  from git tracking (if tracked)
- [x] `.gitignore` adds rules for:
  ```
  .coverage
  coverage.txt
  coverage.html
  coverage/
  test-audit-*.txt
  *.patch
  ```
- [x] `git status` after a `pytest --cov` run shows no new untracked coverage files
- [x] Typecheck passes

---

### US-250: Remove backup .env files and document their absence

**Description:** As a security reviewer, I want no backup or legacy `.env` files committed
to the repository, even if they contain only placeholders.

**Acceptance Criteria:**
- [x] `.env.backup-legacy` is removed from git: `git rm .env.backup-legacy`
- [x] `.env.example.backup_before_refactor` is removed from git
- [x] `.gitignore` adds `*.env.backup*` and `.env.backup*`
- [x] `ls -la | grep ".env"` shows only `.env.example` (the canonical template) at root
- [x] `backups/` directory is evaluated: if it contains no tracked files, add to `.gitignore`;
  if it contains tracked files, move them to `docs/archive/` or delete as appropriate
- [x] Typecheck passes

---

### US-251: Evaluate and resolve ci-fixes.patch

**Description:** As a developer, I want the committed `ci-fixes.patch` file to either be
applied and deleted, or documented and archived, so that its 210 KB does not sit as an
unresolved artifact in the root.

**Acceptance Criteria:**
- [x] `ci-fixes.patch` is read and its purpose is summarized in a note or commit message
- [x] If the patch has already been applied to the codebase, the file is removed from the root and no duplicate changes are introduced
- [x] If the patch has NOT been applied and is still relevant, the relevant changes are applied safely
- [x] If the patch is stale or only partially relevant, the relevant hunks are applied if safe, and the remaining artifact is moved to `docs/archive/housekeeping/` with an explanatory note
- [x] Root contains no `*.patch` files after this story
- [x] `.gitignore` contains `*.patch`
- [x] `tests/test_us251_patch_hygiene.py` exits 0
- [x] `mypy rex` exits 0

---

### US-252: Move root-level utility scripts to scripts/ or remove them

**Description:** As a developer, I want root-level one-off utility scripts moved to
`scripts/` so that the root directory only contains production entry points and standard
project files.

**Acceptance Criteria:**
- [x] Each script in the list below is evaluated:
  - `check_gpu_status.py`, `check_imports.py`, `check_patch_status.py`, `check_tts_imports.py`,
    `find_gpt2_model.py`, `generate_wake_sound.py`, `list_audio.py`, `list_voices.py`,
    `manual_search_demo.py`, `manual_whisper_demo.py`, `play_test.py`, `record_wakeword.py`,
    `test_imports.py`, `test_mic_open.py`, `test_transformers_patch.py`, `wake_acknowledgment.py`
  - Each is classified as: `move to scripts/`, `move to tests/`, or `delete`
  - Classification is documented in a single commit message
- [x] All scripts classified as "move" are moved with `git mv`
- [x] All scripts classified as "delete" are removed with `git rm`
- [x] `scripts/README.md` is updated to list each moved script and its purpose
- [x] Root `.py` files after cleanup: only `rex_loop.py`, `rex_speak_api.py`, `run_gui.py`
  (deprecated), `voice_loop.py` (legacy re-export), and `setup.py`
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-253: Evaluate wakeword_listener.py and wakeword_utils.py at root

**Description:** As a developer, I want to know whether `wakeword_listener.py` and
`wakeword_utils.py` at the root are duplicates of `rex/wakeword/` so that one copy
can be removed.

**Acceptance Criteria:**
- [x] Read `wakeword_listener.py` (root) and `rex/wakeword/` — document differences in commit
- [x] If root versions are stale re-exports: `git rm wakeword_listener.py wakeword_utils.py`
  and add a note to `CLAUDE.md` that the canonical implementation is `rex/wakeword/`
- [x] If root versions contain unique code: move unique code into `rex/wakeword/` then delete roots
- [x] `grep -r "from wakeword_listener\|import wakeword_listener" --include="*.py" .` returns
  zero results after removal (or references are updated to `rex.wakeword`)
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

# WORKSTREAM F — Entry Point Correctness

### US-254: Verify and fix rex-config entry point

**Description:** As a developer running `rex-config`, I want the command to actually work
rather than failing because it targets the wrong module.

**Acceptance Criteria:**
- [x] Read `rex/config.py` (or `rex/config_manager.py`) and confirm which file contains
  a `cli()` function
- [x] If `rex.config:cli` does not exist: update `pyproject.toml` to point to the correct
  module and function (likely `rex.config_manager:cli` or create the `cli` function)
- [x] `rex-config --help` (after `pip install -e .`) exits 0 and prints usage
- [x] CI `ci.yml` adds: `rex-config --help` to the smoke-test step
- [x] Typecheck passes

---

### US-255: Verify rex-agent and rex-tool-server entry points

**Description:** As a developer, I want the less-commonly-used entry points to be verified
working so that operators can rely on them.

**Acceptance Criteria:**
- [x] `python -c "from rex.computers.agent_server import main; print('ok')"` exits 0
- [x] `python -c "from rex.openclaw.tool_server import main; print('ok')"` exits 0
- [x] If either import fails: fix the import path in `pyproject.toml` or fix the module
- [x] CI smoke-test step adds both import checks
- [x] Typecheck passes

---

### US-256: Add CI smoke test for all six entry points

**Description:** As a developer, I want CI to verify all declared entry points are importable
so that broken entry points are caught before release.

**Acceptance Criteria:**
- [x] `ci.yml` smoke-test step runs:
  ```bash
  python -c "from rex.cli import main; print('rex ok')"
  python -c "from rex.config import cli; print('rex-config ok')"  # or corrected path
  python -c "import rex_speak_api; print('rex-speak-api ok')"
  python -c "from rex.computers.agent_server import main; print('rex-agent ok')"
  python -c "from rex.gui_app import main; print('rex-gui ok')"
  python -c "from rex.openclaw.tool_server import main; print('rex-tool-server ok')"
  ```
- [x] All six checks pass
- [x] Typecheck passes

---

# WORKSTREAM G — Security and Dependency Doc Cleanup

### US-257: Audit CVE suppression list and remove stale entries

**Description:** As a security reviewer, I want the CI vulnerability scan suppression list
to contain only current, justified entries so that the scan output is meaningful.

**Acceptance Criteria:**
- [x] Read `ci.yml:217–304` (all `--ignore-vuln` entries) and compare against current
  `pip-audit` output on the installed dependency set
- [x] Entries where the vulnerable package is no longer installed are removed
- [x] The duplicate `CVE-2026-4539` entry (appears at lines 272 and 304) is deduplicated
- [x] Each remaining `--ignore-vuln` entry has a corresponding entry in
  `docs/security/VULNERABILITY-SCAN.md` with: CVE ID, affected package, reason accepted,
  and date of last review
- [x] `pip-audit` run in CI exits 0 after cleanup (all remaining CVEs are suppressed for
  documented reasons)
- [x] Typecheck passes

---

### US-258: Update VULNERABILITY-SCAN.md to cover all remaining suppressions

**Description:** As a developer, I want every CVE suppression in `ci.yml` to have a
written justification in `docs/security/VULNERABILITY-SCAN.md` so that the suppression
list is auditable.

**Acceptance Criteria:**
- [x] `docs/security/VULNERABILITY-SCAN.md` contains one entry per `--ignore-vuln` CVE
  in `ci.yml` with: CVE ID, affected package, installed version, accepted reason, review date
- [x] Total entry count in the doc matches total `--ignore-vuln` count in `ci.yml`
- [x] No "TBD" or placeholder justifications remain
- [x] Typecheck passes

---

### US-259: Evaluate setup.py py_modules backward-compat shims

**Description:** As a developer, I want to know whether the root-level `config.py`,
`llm_client.py`, `memory_utils.py`, and `logging_utils.py` shims are still needed so
that `setup.py` can be simplified or removed.

**Acceptance Criteria:**
- [x] `grep -rn "from config import\|import config\b" --include="*.py" .` is run and
  results documented in the commit message
- [x] Same search for `llm_client`, `memory_utils`, `logging_utils`
- [x] If zero external references: delete the root-level shim files and remove them from
  `setup.py:13–23`; add them to `.gitignore` or note deletion
- [x] If references exist outside `rex/`: create a deprecation warning in each shim pointing
  to the correct package path; schedule removal in next cycle
- [x] `setup.py` is either cleaned up or documented with a comment explaining why it still exists
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-260: Move docs/security/ files to canonical location and verify accuracy

**Description:** As a developer, I want all security-related documentation to live under
`docs/security/` with accurate content so that there is one place to look for security posture.

**Acceptance Criteria:**
- [x] `docs/security/` contains exactly: `SECURITY_ADVISORY.md`, `SECURITY_AUDIT_2026-01-08.md`,
  `VULNERABILITY-SCAN.md` — no other files (INDEX.md and SECURITY_FIX_SUMMARY.md removed;
  SECRET-SCAN.md retained — required by test_us096_secret_scan.py::TestScanDocumentation)
- [x] Each file's header date is accurate (not stale from a prior draft)
- [x] `SECURITY_AUDIT_2026-01-08.md` contains no TODO or TBD sections
- [x] `SECURITY_ADVISORY.md` has been moved from root (per US-243)
- [x] `README.md` security section links to `docs/security/SECURITY_ADVISORY.md`
- [x] Typecheck passes

---

# WORKSTREAM H — Active CI Failures (Run 23946480448)

### US-261: Fix mypy no-redef error in rex/wakeword/embedding.py

**Description:** As a developer, I want the duplicate `_torch` symbol removed from
`embedding.py` so that mypy passes with zero `[no-redef]` errors in that file.

**Acceptance Criteria:**
- [x] `rex/wakeword/embedding.py` defines `_torch` exactly once
- [x] Torch import degrades gracefully when torch is not installed
- [x] `mypy rex/wakeword/embedding.py --ignore-missing-imports` exits 0
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-262: Fix mypy no-any-return in custom_voices.py and smart_speaker_output.py

**Description:** As a developer, I want functions declaring `float` or `str` return types
to return explicitly typed values so that mypy's `no-any-return` rule is satisfied.

**Acceptance Criteria:**
- [x] `rex/custom_voices.py` line ~52: return cast to `float` (e.g. `return float(info.duration)`)
- [x] `rex/audio/smart_speaker_output.py` line ~41: return cast to `float`
- [x] `rex/audio/smart_speaker_output.py` line ~52: return cast to `str`
- [x] `mypy rex/custom_voices.py rex/audio/smart_speaker_output.py --ignore-missing-imports` exits 0
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-263: Remove stale type:ignore comments in four files

**Description:** As a developer, I want all `# type: ignore` comments flagged as
`[unused-ignore]` removed so that the annotation layer is clean.

**Acceptance Criteria:**
- [x] `rex/compat/transformers_shims.py` line ~76: stale `# type: ignore` removed or
  replaced with a scoped `# type: ignore[attr-defined]` with an explanatory comment
- [x] `rex/audio/smart_speaker_output.py` line ~87: removed
- [x] `rex/shopping_pwa.py` lines ~337, 361, 369, 378, 385, 399, 413: all seven removed
- [x] `rex/voice_loop.py` line ~209: removed
- [x] `mypy rex/compat/transformers_shims.py rex/audio/smart_speaker_output.py rex/shopping_pwa.py rex/voice_loop.py --ignore-missing-imports`
  exits 0 with zero `[unused-ignore]` errors
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-264: Fix mypy return-value and call-arg errors in shopping_pwa.py and twilio_handler.py

**Description:** As a developer, I want Flask route handlers to return the correct Response
type and `Assistant` to be called with valid arguments.

**Acceptance Criteria:**
- [x] `rex/shopping_pwa.py` line ~339: return type corrected to `flask.wrappers.Response`
- [x] `rex/telephony/twilio_handler.py` line ~88: return cast to `bool`
- [x] `rex/telephony/twilio_handler.py` line ~399: `Assistant(config=...)` corrected to
  match `Assistant.__init__` actual signature (inspect `rex/assistant.py` for real params)
- [x] `mypy rex/shopping_pwa.py rex/telephony/twilio_handler.py --ignore-missing-imports` exits 0
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

### US-265: Fix psutil ModuleNotFoundError blocking CI test collection

**Description:** As a developer, I want `rex/tools/windows_diagnostics.py` to import
`psutil` conditionally so that test collection does not fail on Linux CI runners.

**Acceptance Criteria:**
- [x] `rex/tools/windows_diagnostics.py` wraps `import psutil` in `try/except ImportError`;
  when absent, functions return a `{"error": "psutil not installed"}` dict and emit
  `logger.warning`
- [x] `tests/test_windows_diagnostics.py` adds `pytest.importorskip("psutil")` at the top
- [x] `requirements-dev.txt` adds `psutil>=5.9`
- [x] `pytest -q tests/test_windows_diagnostics.py` exits 0 (skipped or passing, not erroring)
- [x] `pytest -q` full run exits 0
- [x] Typecheck passes

---

### US-266: Suppress pre-commit secret-detection false positives

**Description:** As a developer, I want lines in test fixtures and security docs that
detect-secrets flags to carry inline suppression markers so that pre-commit exits clean.

**Acceptance Criteria:**
- [x] `tests/helpers/fake_smtp.py` line ~15: `  # pragma: allowlist secret` appended
- [x] `tests/test_email_backend_imap_smtp.py` line ~602: same
- [x] `docs/ARCHITECTURE.md` line ~448: `<!-- pragma: allowlist secret -->` added
- [x] `docs/security/SECURITY_AUDIT_2026-01-08.md` lines ~40–41: both suppressed
- [x] `pre-commit run detect-secrets --all-files` exits 0
- [x] Verified that suppressed lines contain only placeholder/test values, not real credentials
- [x] Typecheck passes

---

# WORKSTREAM I — Brand Asset Integration

### US-267: Add AskRex brand logo assets to assets/brand/

**Description:** As a developer, I want all official AskRex Assistant brand logo variants
stored under `assets/brand/` with a usage README so that all UI surfaces can reference them.

**Acceptance Criteria:**
- [x] `assets/brand/` directory is created
- [x] The following variants are present:
  `icon-square.png`, `icon-circle.png`, `icon-r.png`,
  `wordmark-dark.png`, `wordmark-light.png`, `wordmark-reverse.png`,
  `primary-horizontal.png`, `stacked.png`, `favicon.ico`
- [x] Each PNG is at minimum 512 px on longest axis; favicon.ico is multi-size (16/32/48 px)
- [x] `assets/brand/README.md` documents each variant and its intended use
- [x] `assets/logo.svg` is updated or replaced if a higher-fidelity vector source is available
- [x] Typecheck passes (asset-only story; no Python changes required)

---

### US-268: Update README.md with official AskRex brand logo

**Description:** As a user visiting the repository, I want to see the official AskRex
Assistant logo so that the project presents a professional identity.

**Acceptance Criteria:**
- [x] `README.md` opens with an `<img>` tag referencing `assets/brand/primary-horizontal.png`
  or `assets/brand/stacked.png` at display width 400 px
- [x] Alt text: `"AskRex Assistant — local-first voice AI"`
- [x] Any previous placeholder logo reference is removed
- [x] README renders correctly (verify via `gh browse` or manual check)
- [x] Typecheck passes

---

### US-269: Update Electron GUI and shopping PWA with brand assets

**Description:** As a user running the desktop app or shopping PWA, I want the official
AskRex icon and wordmark to appear in application chrome.

**Acceptance Criteria:**
- [x] Electron `gui/package.json` — `"icon"` field set to `assets/brand/icon-square.png`
- [x] Shopping PWA HTML template — `<link rel="icon">` uses `assets/brand/favicon.ico`
- [x] Shopping PWA `<link rel="apple-touch-icon">` uses `assets/brand/icon-square.png`
- [x] Shopping PWA `<title>` reads `"AskRex — Shopping"`
- [x] Shopping PWA header renders `assets/brand/wordmark-dark.png` or `wordmark-light.png`
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

## 12. Acceptance Criteria by Workstream

### WS-A: Product Identity
- `grep -r "Rex AI Assistant" . --include="*.md" --include="*.toml" --include="*.py"` returns
  zero results outside `docs/archive/` and historical CHANGELOG entries
- `grep -r "rex-ai-assistant" . --include="*.yml"` returns zero results
- `pip show askrex-assistant` returns correct metadata after `pip install -e .`

### WS-B: Branch and Release
- CI and release-please `on: push: branches:` entries are identical
- A push to the canonical branch triggers both CI and release-please jobs
  (verify by inspection of job run history after the change)

### WS-C: Documentation Truth
- `grep "stub/mock data only" README.md` — zero results
- `grep "Autonomous workflows" README.md` — accompanied by `[In progress]` annotation
- `grep "requires-python" pyproject.toml` — value is `>=3.11,<3.12`
- README explicitly states "Python 3.12 and above are not supported"
- Feature list has `[Works today]`, `[Requires configuration]`, or `[In progress]` on every bullet

### WS-D: UI Consolidation
- `grep "run_gui.py" README.md INSTALL.md` — zero results
- `docs/UI_SURFACES.md` exists with complete classification table
- `askrex-gui` entry point launches without error

### WS-E: Root Hygiene
- `ls *.txt` at root — zero results (progress files archived)
- `ls progress*.txt` — zero results
- `ls *.patch` — zero results
- `ls .env.backup*` — zero results
- `.gitignore` contains: `.coverage`, `*.patch`, `progress*.txt`
- Root `.py` files: only `rex_loop.py`, `rex_speak_api.py`, `run_gui.py` (deprecated),
  `voice_loop.py` (legacy), `setup.py`

### WS-F: Entry Points
- `rex-config --help` exits 0
- `rex-agent --help` (or equivalent import check) exits 0
- `rex-tool-server --help` (or equivalent import check) exits 0
- All six entry point smoke tests in CI pass

### WS-G: Security Docs
- `--ignore-vuln` count in `ci.yml` == entry count in `VULNERABILITY-SCAN.md`
- No duplicate CVE IDs in `ci.yml`
- `docs/security/` contains exactly three files; no `SECURITY_ADVISORY.md` at root

### WS-H: Active CI Failures
- `mypy rex --ignore-missing-imports` exits 0 with zero errors
- `pytest -q` exits 0 with no collection errors
- `pre-commit run --all-files` exits 0

### WS-I: Brand Assets
- `ls assets/brand/` shows all nine expected files
- `README.md` first image tag references `assets/brand/`

---

## 13. Validation and Verification Commands

Run these after each phase to confirm it is complete.

```bash
# Phase 1 (WS-A) — Naming
grep -r "Rex AI Assistant" . --include="*.md" --include="*.toml" --include="*.py" \
  --exclude-dir=".git" --exclude-dir="docs/archive"
grep -r "rex-ai-assistant" .github/ --include="*.yml"
pip show askrex-assistant | grep -E "Name|Home-page"

# Phase 2 (WS-B) — Branch
grep "branches:" .github/workflows/ci.yml .github/workflows/release-please.yml

# Phase 3 (WS-C) — Docs truth
grep "stub/mock data only" README.md
grep "In progress\|Works today\|Requires configuration" README.md | wc -l
grep "requires-python" pyproject.toml

# Phase 4 (WS-D) — UI
grep "run_gui.py" README.md INSTALL.md
python -c "from rex.gui_app import main; print('gui ok')"

# Phase 5 (WS-E) — Root hygiene
ls *.txt *.patch 2>/dev/null
ls .env.backup* 2>/dev/null
git ls-files | grep "progress" | grep -v "docs/archive"

# Phase 6 (WS-F) — Entry points
rex-config --help
python -c "from rex.computers.agent_server import main; print('ok')"
python -c "from rex.openclaw.tool_server import main; print('ok')"

# Phase 7 (WS-G) — Security
grep -c "ignore-vuln" .github/workflows/ci.yml
grep -c "CVE-" docs/security/VULNERABILITY-SCAN.md

# Phase 8 (WS-H) — CI failures
mypy rex --ignore-missing-imports 2>&1 | tail -3
pytest -q 2>&1 | tail -5
pre-commit run --all-files 2>&1 | tail -10

# Full quality gate (run before any story marked complete)
ruff check rex/
black --check rex/ *.py
mypy rex --ignore-missing-imports
pytest -q
```

---

## 14. Archive Strategy

The following items must be archived rather than deleted outright, because they represent
legitimate project history that may need to be referenced.

| Item | Current Location | Archive Location | Action |
|---|---|---|---|
| `progress-*.txt` files | Root | `docs/archive/progress/` | `git mv` |
| `coverage.txt`, `test-audit-*.txt` | Root | Remove from git tracking | `git rm --cached` |
| `ci-fixes.patch` | Root | Apply or move to `docs/archive/` | Evaluate content first |
| `SECURITY_ADVISORY.md` | Root | `docs/security/` | `git mv` |
| `SECURITY_AUDIT_2026-01-08.md` | `docs/security/` | Keep in place | No change needed |
| `.env.backup-legacy` | Root | **Delete** — not a doc, just a leaked backup | `git rm` |
| `.env.example.backup_before_refactor` | Root | **Delete** | `git rm` |
| `gui.py` / `run_gui.py` | Root | Keep as deprecated for one cycle | Add deprecation header |
| `backups/` directory | Root | Evaluate contents; add to `.gitignore` | Check tracked files |

**Rule for archiving vs. deleting:**
- Archive if: the file contains reasoning, decisions, or investigation results that may
  be referenced to understand why current code is the way it is.
- Delete if: the file is a generated artifact, a duplicate, or an accidentally committed
  backup with no documentation value.

---

## 15. Definition of Done

The overall PRD is complete when all of the following are true:

1. `grep -r "Rex AI Assistant" . --include="*.md" --include="*.toml" --include="*.py" \`
   `--exclude-dir=".git" --exclude-dir="docs/archive"` returns zero results.

2. `grep "branches:" .github/workflows/ci.yml` and
   `grep "branches:" .github/workflows/release-please.yml` return the same branch name.

3. `README.md` feature list has an explicit status annotation on every bullet (`[Works today]`,
   `[Requires configuration]`, or `[In progress]`), and contains no claim that contradicts
   `docs/claude/INTEGRATIONS_STATUS.md`.

4. `docs/UI_SURFACES.md` exists with a complete surface classification table.

5. `python run_gui.py` prints a deprecation warning before launching.

6. `ls *.txt *.patch .env.backup* 2>/dev/null` at the repo root returns nothing.

7. All six entry point smoke tests pass in CI.

8. `--ignore-vuln` count in `ci.yml` == CVE entry count in `VULNERABILITY-SCAN.md`,
   with zero duplicates.

9. `mypy rex --ignore-missing-imports` exits 0.

10. `pytest -q` exits 0 with coverage >= 75%.

11. `pre-commit run --all-files` exits 0.

12. `assets/brand/` contains all nine logo variants.

13. `pip install -e . && pip show askrex-assistant` returns correct metadata with
    canonical name and GitHub URL.

14. A reviewer unfamiliar with the project can clone the repo, read `README.md` and
    `INSTALL.md`, and successfully run `askrex` (or `python -m rex`) within 30 minutes
    without consulting any other document.

---

## Appendix: Legacy Feature Backlog

The stories below were written in a prior cycle. They cover real integrations, bug fixes,
and feature additions. **Do not start these until all WS-A through WS-I stories above are
complete.** Starting feature work before consolidation is done will create new inconsistencies
faster than they are being resolved.

All stories below retain their original `[x]`/`[ ]` state from prior runs.

Story IDs US-175 through US-220 are the previous production-readiness cycle.
Story IDs US-221 through US-229 are CI failure fixes and brand asset stories (now
superseded by WS-H and WS-I above with renumbered IDs US-261–US-269; skip if already done).

---

# APPENDIX PHASE A — Security and Docker Hardening

### US-175: Harden .dockerignore to exclude secrets and local state

**Description:** As an operator, I want the Docker build context to exclude secrets and
local runtime state so that `docker build` never captures `.env`, credentials, or
development artifacts.

**Acceptance Criteria:**
- [x] `.dockerignore` excludes: `.env`, `.env.*`, `venv/`, `.venv/`, `config/credentials.json`,
  `data/`, `logs/`, `transcripts/`, `Memory/`, `session_summaries/`, `backups/`, `*.log`,
  `*.bundle`, `*.egg-info/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`
- [x] `.dockerignore` excludes test artifacts: `tests/`, `coverage.json`, `coverage.txt`
- [x] `docker build .` succeeds after the change
- [x] Running `docker build .` in a directory containing a `.env` file does NOT include
  `.env` in the image (verify with `docker run --rm <image> ls /app/.env || echo "not found"`)
- [x] Typecheck passes

---

### US-176: Replace broad Dockerfile COPY with allowlist

**Description:** As an operator, I want the Dockerfile runtime stage to copy only
production-required files so that the resulting image is minimal and safe.

**Acceptance Criteria:**
- [x] Dockerfile runtime stage replaces `COPY . .` with explicit allowlist covering only:
  `rex/`, `rex_speak_api.py`, `rex_loop.py`, `voice_loop.py`, `pyproject.toml`,
  `config/rex_config.example.json`, `assets/`, and entry-point scripts
- [x] Image builds successfully: `docker build -t rex-test .` exits 0
- [x] `docker run --rm rex-test python -c "import rex"` exits 0
- [x] Image does not contain `.env`, `tests/`, `venv/`, or `Memory/` directories
- [x] Typecheck passes

---

# APPENDIX PHASE B — Code Quality Restoration

### US-177: Restore Ruff lint compliance — import and unused-code violations

**Description:** As a developer, I want all Ruff import-order and unused-symbol violations
fixed so that the linter baseline is clean before enforcing it in CI.

**Acceptance Criteria:**
- [x] `ruff check rex/ --select I,F` exits 0 (import order + unused imports/variables)
- [x] No `noqa` suppressions added that were not already present
- [x] `pytest -q` exits 0 after changes (no regressions)
- [x] Typecheck passes

### US-178: Restore Ruff lint compliance — remaining rule violations

**Acceptance Criteria:**
- [x] `ruff check rex/` exits 0 with zero errors
- [x] `pytest -q` exits 0
- [x] Typecheck passes

### US-179: Restore Black formatting compliance

**Acceptance Criteria:**
- [x] `black --check rex/` exits 0
- [x] `black --check *.py` exits 0
- [x] No logic changes introduced
- [x] `pytest -q` exits 0
- [x] Typecheck passes

### US-180: Resolve mypy type errors — batch 1 (core package)

**Acceptance Criteria:**
- [x] `mypy rex/assistant.py rex/config.py rex/llm_client.py rex/voice_loop.py` exits 0
- [x] No unexplained `type: ignore` comments added
- [x] `pytest -q` exits 0
- [x] Typecheck passes

### US-181: Resolve mypy type errors — batch 2 (integrations and remaining files)

**Acceptance Criteria:**
- [x] `mypy rex/` exits 0 with zero errors
- [x] `pytest -q` exits 0
- [x] Typecheck passes

---

# APPENDIX PHASE C — Test Infrastructure

### US-182: Fix brittle repo-integrity tests

**Acceptance Criteria:**
- [x] `tests/test_repo_integrity.py` captures `git status --porcelain` baseline before any test
- [x] `pytest -q tests/test_repo_integrity.py` exits 0 even when pre-existing dirty files exist
- [x] Typecheck passes

---

# APPENDIX PHASE D — Operations Scripts

### US-183: Fix security audit script false positives

**Acceptance Criteria:**
- [x] `scripts/security_audit.py` excludes `.mypy_cache/`, `.ruff_cache/`, caches, venv
- [x] Running script reports fewer than 50 findings on clean checkout
- [x] Typecheck passes

### US-184: Rewrite deployment validation script

**Acceptance Criteria:**
- [x] `scripts/validate_deployment.py` checks `config/rex_config.json` existence and schema
- [x] Script validates torch version against `pyproject.toml` range
- [x] `python scripts/validate_deployment.py` exits 0 on properly configured install
- [x] Typecheck passes

---

# APPENDIX PHASE E — Execution Surface Correctness

### US-185: Define authoritative executable tool catalog
### US-186: Implement weather_now and web_search tool handlers
### US-187: Implement send_email and calendar_create_event tool handlers
### US-188: Add planner-to-router end-to-end integration tests

*(Full acceptance criteria in prior cycle document; stories not yet marked complete)*

---

# APPENDIX PHASE F — Documentation

### US-189: Align README runtime configuration section
### US-190: Rewrite Windows quickstart with correct entrypoints
### US-191: Archive and correct stale architecture and status documents
### US-192: Consolidate to one canonical voice loop entry point

*(Full acceptance criteria in prior cycle; US-192 partially addressed by WS-D above)*

---

# APPENDIX PHASE G — Dependency Alignment

### US-193: Define and document the canonical runtime matrix

*(Full acceptance criteria in prior cycle)*

---

# APPENDIX PHASE H — Bug Fixes

### US-194: Thread-safe TTS engine in rex_speak_api.py
### US-195: Fix _followup_injected race condition in assistant.py
### US-196: Fix inconsistent temp file cleanup in voice_loop.py
### US-197: Process OpenAI tool_calls in LLM client
### US-198: Fix Ollama error message taxonomy
### US-199: Fix sentence splitting for abbreviations in TTS pipeline
### US-200: Add request body size limit to rex_speak_api.py
### US-201: Fix suppressed JSON errors in identity.py

*(Full acceptance criteria in prior cycle)*

---

# APPENDIX PHASE I — Conversation History

### US-202 – US-204: History persistence via SQLite HistoryStore

*(Full acceptance criteria in prior cycle)*

---

# APPENDIX PHASE J — Real Integration Backends

### US-205 – US-212: Email, calendar, SMS, and offline test harnesses

*(Full acceptance criteria in prior cycle)*

---

# APPENDIX PHASE K — Features

### US-213 – US-220: Whisper language, audio validation, LLM streaming, pre-commit, test suite green

*(Full acceptance criteria in prior cycle; US-220 is the final integration gate)*

---

**Story ordering note (consolidated):**
- WS-A (US-230–234) has no dependencies; do first
- WS-B (US-235–237) depends on WS-A
- WS-C (US-238–243) depends on WS-A
- WS-D (US-244–247) depends on WS-C
- WS-E (US-248–253) depends on WS-D
- WS-F (US-254–256) depends on WS-E
- WS-G (US-257–260) depends on WS-C; can run in parallel with WS-D and WS-E
- WS-H (US-261–266) has no blocking dependencies; run in parallel with any workstream
- WS-I (US-267–269) depends on WS-A
- Appendix stories (US-175–US-220) must wait until all WS-A through WS-I are complete
