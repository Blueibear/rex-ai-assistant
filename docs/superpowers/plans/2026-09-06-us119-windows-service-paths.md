# US-119 Windows Service Absolute Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every repository-controlled Windows Rex/RexSpeak service launch or registration path is rooted in a normalized absolute install/venv path and cannot silently depend on the caller's current working directory.

**Architecture:** Keep Windows path authority at the installer/launcher boundary. PowerShell launchers normalize their canonical root before constructing executable paths, service registration is invoked through an absolute venv interpreter, and runtime child launches normalize `sys.executable` before reuse. Cross-platform source-contract tests plus a Windows CI dry-run exercise protect the persisted registration command without changing pywin32's service-host semantics.

**Tech Stack:** PowerShell 5.1+, Python 3.11, pytest, pywin32, GitHub Actions Windows runner.

**Spec:** `PRD-production-readiness.md` — US-119.

## Global Constraints

- Preserve the existing developer/operator source install path; US-119 does not replace the packaged consumer installer.
- Do not change pywin32 service-host semantics or introduce a second Windows service framework.
- PowerShell commands must remain safe when install roots contain spaces.
- Dry-run output must expose the exact normalized registration interpreter path while real registration must fail closed if that interpreter does not exist.
- Update `CLAUDE.md` if service-registration behavior or commands change.

---

### Task 1: Add failing path-contract tests

**Files:**
- Create: `tests/test_us119_windows_service_paths.py`
- Modify: `.github/workflows/windows-electron-artifact.yml`
- Test: `tests/test_us119_windows_service_paths.py`

**Interfaces:**
- Consumes: repository PowerShell launch/install scripts.
- Produces: regression contracts for script-rooted RexSpeak startup, normalized lean-node roots, quoted absolute registration commands, and Windows dry-run execution from a non-repository working directory.

- [ ] **Step 1: Write source-contract tests** that assert `Start-RexSpeak.ps1` uses `$PSScriptRoot`, never `Resolve-Path .` or a caller-relative `.\\.venv\\Scripts\\python.exe`, and that `node_installers/install_windows.ps1` normalizes `$RexRoot` before building `venv\\Scripts\\python.exe`, quotes the dry-run registration command, and verifies the real interpreter exists before registration.
- [ ] **Step 2: Add a Windows-runner test invocation** before artifact construction so PowerShell dry-run behavior executes on Windows rather than being inferred only from source text.
- [ ] **Step 3: Run the focused tests on the test-only commit** and confirm they fail for the current relative-path behavior.
- [ ] **Step 4: Commit** with `test(windows): cover absolute service registration paths`.

### Task 2: Root RexSpeak startup at the script location

**Files:**
- Modify: `Start-RexSpeak.ps1`
- Test: `tests/test_us119_windows_service_paths.py`

**Interfaces:**
- Consumes: `$PSScriptRoot` and the repository `.venv`.
- Produces: an absolute RexSpeak Python interpreter path and absolute `PYTHONPATH` independent of caller working directory.

- [ ] **Step 1: Verify the Task 1 RexSpeak test is RED.**
- [ ] **Step 2: Normalize `$PSScriptRoot` with `[System.IO.Path]::GetFullPath`, construct `.venv\\Scripts\\python.exe` with `Join-Path`, fail if it is missing, set `PYTHONPATH` to that absolute root, and invoke the interpreter via PowerShell's call operator.**
- [ ] **Step 3: Run the focused RexSpeak tests and confirm GREEN.**
- [ ] **Step 4: Commit** with `fix(windows): root RexSpeak launcher at script path`.

### Task 3: Normalize lean-node service registration

**Files:**
- Modify: `node_installers/install_windows.ps1`
- Test: `tests/test_us119_windows_service_paths.py`

**Interfaces:**
- Consumes: optional `$RexRoot`, current working directory for relative user input, `venv\\Scripts\\python.exe`.
- Produces: normalized absolute `$RexRoot`, absolute `$python`/`$pip`, quoted dry-run registration command, and fail-closed real registration.

- [ ] **Step 1: Verify the Task 1 lean-node tests are RED.**
- [ ] **Step 2: Normalize `$RexRoot = [System.IO.Path]::GetFullPath($RexRoot)` before any directory/venv construction.**
- [ ] **Step 3: Construct `$pip`, `$python`, and `.env.node` with `Join-Path`; after real venv creation, fail if `$python` is absent.**
- [ ] **Step 4: Emit dry-run install/start commands as `& \"<absolute python>\" -m rex.windows_service ...` so paths containing spaces are represented truthfully.**
- [ ] **Step 5: Run focused tests, including a relative root containing spaces from a non-repository working directory, and confirm GREEN.**
- [ ] **Step 6: Commit** with `fix(windows): normalize lean-node service root`.

### Task 4: Harden Python service child interpreter reuse

**Files:**
- Modify: `rex/windows_service.py`
- Modify: `tests/test_windows_service.py`

**Interfaces:**
- Consumes: `sys.executable` from the interpreter that installed/hosts Rex.
- Produces: a normalized absolute existing interpreter used for the `rex.app` child process.

- [ ] **Step 1: Add a Windows test that replaces `sys.executable` with a path spelling that resolves to the expected interpreter and asserts the child command receives the normalized absolute executable; add a missing-executable test that fails closed.**
- [ ] **Step 2: Run the new tests and confirm RED.**
- [ ] **Step 3: Add a small helper using `pathlib.Path(...).resolve(strict=True)` and use it as `cmd[0]` in `SvcDoRun`.**
- [ ] **Step 4: Run `pytest tests/test_windows_service.py tests/test_us119_windows_service_paths.py -q` on Windows and confirm GREEN.**
- [ ] **Step 5: Commit** with `fix(windows): normalize service child interpreter`.

### Task 5: Documentation, tracker, and full verification

**Files:**
- Modify: `CLAUDE.md`
- Modify: `PRD-production-readiness.md`
- Modify: `docs/archive/progress/progress-production-readiness.txt`

**Interfaces:**
- Consumes: verified implementation/test evidence.
- Produces: explicit rule that persisted Windows service registration and launch wrappers use absolute canonical roots and a truthful US-119 completion record.

- [ ] **Step 1: Add the Windows service absolute-path invariant to `CLAUDE.md`.**
- [ ] **Step 2: Record the audited registration surfaces and focused validation in the progress ledger.**
- [ ] **Step 3: Run focused pytest, Ruff/Black/mypy where applicable, PowerShell parser checks, `git diff --check`, and the Windows service dry-run from a non-repo working directory.**
- [ ] **Step 4: Open a PR to `master`; run all required GitHub checks including the Windows artifact workflow.**
- [ ] **Step 5: Only after exact-head checks pass, mark every US-119 criterion complete and merge.**
