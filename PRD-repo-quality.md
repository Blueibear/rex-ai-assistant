# PRD: Rex Repo Quality & Technical Debt

IMPORTANT

Stories must remain atomic.

If a story becomes too large, it must be split into smaller stories before implementation.

A story is complete only when all acceptance criteria checkboxes are checked.

---

## Introduction

This PRD addresses four identified areas of technical debt in the Rex repository: fragmented test helper logic, missing incremental CI enforcement, widespread use of the deprecated `datetime.utcnow()` API, and use of the deprecated `asyncio.get_event_loop()` pattern in tests. None of these items block existing functionality, but left unaddressed they accumulate fragility, suppress warnings that mask real problems, and make future contributors' lives harder.

Known baseline state going in:
- Repo-wide lint/format quality gates are not globally green; only touched files are currently clean.
- `datetime.utcnow()` deprecation warnings appear across multiple modules during the full test run (warnings only, not failures).

---

## Goals

- Centralize duplicated git-status test logic into a single shared fixture so invariants stay consistent
- Add a CI job that enforces `ruff`/`black` on changed files and tracks total lint debt over time, preventing new violations from landing
- Eliminate all `datetime.utcnow()` calls across the four affected modules and replace with timezone-aware equivalents
- Eliminate all `asyncio.get_event_loop()` usage in tests and replace with `asyncio.run()` / `pytest-asyncio` idioms
- Produce a full test run with zero deprecation warnings related to datetime or asyncio

---

## Non-Goals

- Not achieving 100% lint cleanliness across the entire repo in one pass (the CI job enforces on changed files only)
- Not migrating to a different test framework
- Not changing the public API or behavior of any modified module
- Not addressing deprecation warnings beyond `datetime.utcnow()` and `asyncio.get_event_loop()`
- Not upgrading minimum Python version requirements

---

## Technical Considerations

- UTC replacement: use `datetime.now(timezone.utc)` (importable from the stdlib `datetime` module) rather than `datetime.now(datetime.UTC)`, which requires Python 3.11+. Verify the project's minimum Python version before choosing.
- Timestamp format risk: existing code may compare or serialize naive vs. aware datetimes — check callers of modified functions for format assumptions before replacing.
- pytest-asyncio must be listed in dev dependencies if not already present.
- The CI debt-tracking job should diff lint error counts between base and head branches to surface trend, not just gate on zero.

---

# PHASE 34 — Test Infrastructure & CI

### US-078: Consolidate repo integrity test helpers

**Description:** As a developer, I want git-status parsing logic centralized in a shared fixture so that repo integrity invariants are defined once and stay consistent across all test files.

**Acceptance Criteria:**
- [x] `tests/conftest.py` contains a shared fixture or utility function that encapsulates git-status parsing
- [x] `tests/test_repo_integrity.py` uses the shared fixture instead of its own git-status logic
- [x] `tests/test_repository_integrity.py` uses the shared fixture instead of its own git-status logic
- [x] no duplicated git-status parsing logic remains across those two test files
- [x] all repo integrity tests pass
- [x] Typecheck passes

---

### US-079: Add CI baseline debt mode for lint and format

**Description:** As a developer, I want CI to enforce `ruff` and `black` on changed files only so that new violations cannot land while existing debt is cleaned up incrementally.

**Acceptance Criteria:**
- [x] `.github/workflows/ci.yml` contains a job that runs `ruff check` scoped to files changed in the PR
- [x] `.github/workflows/ci.yml` contains a job that runs `black --check` scoped to files changed in the PR
- [x] CI job fails if any changed file has a lint or format violation
- [x] `pyproject.toml` contains `ruff` and `black` configuration consistent with the CI job
- [x] a passing PR with no lint violations on changed files results in a green CI run
- [x] Typecheck passes

---

# PHASE 35 — UTC API Modernization

### US-080: Replace datetime.utcnow() in rex/memory_utils.py

**Description:** As a developer, I want `datetime.utcnow()` replaced with a timezone-aware UTC call in `rex/memory_utils.py` so that deprecation warnings are eliminated and timestamp behavior is explicit.

**Acceptance Criteria:**
- [x] `datetime.utcnow()` does not appear in `rex/memory_utils.py`
- [x] timezone-aware UTC equivalent (`datetime.now(timezone.utc)` or equivalent) used in its place
- [x] `timezone` imported correctly in the module
- [x] all existing tests that exercise `memory_utils` pass
- [x] no `DeprecationWarning` for `utcnow` emitted by this module during test run
- [x] Typecheck passes

---

### US-081: Replace datetime.utcnow() in rex/identity.py

**Description:** As a developer, I want `datetime.utcnow()` replaced with a timezone-aware UTC call in `rex/identity.py` so that deprecation warnings are eliminated and timestamp behavior is explicit.

**Acceptance Criteria:**
- [x] `datetime.utcnow()` does not appear in `rex/identity.py`
- [x] timezone-aware UTC equivalent used in its place
- [x] `timezone` imported correctly in the module
- [x] all existing tests that exercise `identity` pass
- [x] no `DeprecationWarning` for `utcnow` emitted by this module during test run
- [x] Typecheck passes

---

### US-082: Replace datetime.utcnow() in rex/automation_registry.py

**Description:** As a developer, I want `datetime.utcnow()` replaced with a timezone-aware UTC call in `rex/automation_registry.py` so that deprecation warnings are eliminated and timestamp behavior is explicit.

**Acceptance Criteria:**
- [x] `datetime.utcnow()` does not appear in `rex/automation_registry.py`
- [x] timezone-aware UTC equivalent used in its place
- [x] `timezone` imported correctly in the module
- [x] all existing tests that exercise `automation_registry` pass
- [x] no `DeprecationWarning` for `utcnow` emitted by this module during test run
- [x] Typecheck passes

---

### US-083: Replace datetime.utcnow() in rex/service_supervisor.py

**Description:** As a developer, I want `datetime.utcnow()` replaced with a timezone-aware UTC call in `rex/service_supervisor.py` so that deprecation warnings are eliminated and timestamp behavior is explicit.

**Acceptance Criteria:**
- [x] `datetime.utcnow()` does not appear in `rex/service_supervisor.py`
- [x] timezone-aware UTC equivalent used in its place
- [x] `timezone` imported correctly in the module
- [x] all existing tests that exercise `service_supervisor` pass
- [x] no `DeprecationWarning` for `utcnow` emitted by this module during test run
- [x] Typecheck passes

---

# PHASE 36 — Async Test Modernization

### US-084: Replace asyncio.get_event_loop() in test_us018_speech_to_text.py

**Description:** As a developer, I want `asyncio.get_event_loop()` replaced with `asyncio.run()` or `pytest-asyncio` idioms in `tests/test_us018_speech_to_text.py` so that Python-version fragility and deprecation warnings are removed.

**Acceptance Criteria:**
- [x] `asyncio.get_event_loop()` does not appear in `tests/test_us018_speech_to_text.py`
- [x] async tests use `asyncio.run()` or `@pytest.mark.asyncio` with `pytest-asyncio` instead
- [x] `pytest-asyncio` is listed in dev dependencies if not already present
- [x] all tests in `test_us018_speech_to_text.py` pass
- [x] no `DeprecationWarning` for event loop emitted by this file during test run
- [x] Typecheck passes

---

### US-085: Sweep remaining async tests for asyncio.get_event_loop()

**Description:** As a developer, I want all remaining uses of `asyncio.get_event_loop()` in the test suite replaced so that the full test run is free of event-loop deprecation warnings.

**Acceptance Criteria:**
- [x] `grep -r "asyncio.get_event_loop" tests/` returns zero results
- [x] all modified tests use `asyncio.run()` or `@pytest.mark.asyncio` instead
- [x] full `pytest` run produces no `DeprecationWarning` related to event loop creation
- [x] all previously passing tests continue to pass
- [x] Typecheck passes

---

# PHASE 37 — Conventional Commits Enforcement

> The CI commitlint job is already catching format violations, but only after a bad commit has been pushed. These stories add local enforcement so malformed commit messages are rejected at commit time before they ever reach CI.

### US-086: Add commit-msg hook to reject non-conventional commit messages locally

**Description:** As a developer, I want a `commit-msg` git hook that validates the commit message format before the commit is recorded so that Conventional Commits violations are caught locally and never reach CI.

**Acceptance Criteria:**
- [ ] a `commit-msg` hook exists at `.git/hooks/commit-msg` (or installed via `pre-commit`)
- [ ] the hook rejects any message that does not match `^(feat|fix|test|docs|refactor|chore|perf|ci)(\(.+\))?: .+` and exits with a non-zero code and a human-readable error explaining the required format
- [ ] the hook accepts a correctly formatted message (e.g. `fix(auth): reject expired tokens`) and exits with code 0
- [ ] attempting `git commit -m "Fix something broken"` fails with a clear error message
- [ ] attempting `git commit -m "fix: resolve broken TTS playback"` succeeds
- [ ] hook installation documented in `CONTRIBUTING.md` or `AGENTS.md` with the exact command to run
- [ ] Typecheck passes

---

### US-087: Install pre-commit framework and configure conventional-commits check

**Description:** As a developer, I want `pre-commit` managing the commit-msg hook so that the hook is version-controlled, reproducible, and installable with a single command on any clone.

**Acceptance Criteria:**
- [ ] `.pre-commit-config.yaml` exists at the repo root
- [ ] config includes the `conventional-pre-commit` hook (or equivalent) targeting the `commit-msg` stage
- [ ] `pre-commit` added to dev dependencies in `pyproject.toml`
- [ ] `pre-commit install` installs the hook successfully on a clean clone
- [ ] `pre-commit run --hook-stage commit-msg --commit-msg-filename <(echo "bad message")` exits non-zero
- [ ] `pre-commit run --hook-stage commit-msg --commit-msg-filename <(echo "fix: correct the thing")` exits zero
- [ ] CI updated to run `pre-commit run --all-files` so hook config drift is caught on PRs
- [ ] Typecheck passes

---

## Non-Goals (repeated for Ralph)

- Not achieving zero lint violations repo-wide (only changed files are gated)
- Not modifying any module's public API or return types
- Not addressing deprecation warnings beyond the two patterns above
