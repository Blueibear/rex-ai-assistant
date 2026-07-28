# Claude Reference: Testing and Quality

Use this when a task touches tests, CI, lint, formatting, type checking, repo
integrity, or verification rules.

## Pytest Source of Truth

- Pytest configuration lives in `[tool.pytest.ini_options]` in `pyproject.toml`.
- Do not reintroduce `pytest.ini`.
- Coverage configuration also lives in `pyproject.toml`.
- The coverage report threshold is currently `fail_under = 75`.

## Default Local Validation

```bash
pytest -q
python -m rex --help
python -m rex doctor
python scripts/security_audit.py
```

Release candidates use the stricter gates:

```bash
python -m rex doctor --release-gate
python scripts/security_audit.py --release-gate
```

Developer-mode doctor warnings remain non-blocking but render as `[WARN]`, not
`[PASS]`. The doctor release gate also exits nonzero for warnings explicitly
classified as actionable. The security release gate scans Markdown fences and
fails on merge markers, potential secrets, invalid suppressions, and actionable
placeholder markers in source or configuration files.

Security-audit suppressions are optional JSON passed with `--suppressions`.
Each entry must contain `category: "placeholder"`, a repository-relative
`path`, an exact positive `line`, a specific `reason` of at least 10 characters,
and a non-expired ISO `expires` date. Merge markers and secrets cannot be
suppressed.

For docs-only changes, `git diff --check` is usually enough unless the doc
change modifies commands that should be smoke-tested.

## Electron Validation

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

For Electron-only verification harnesses:

1. Run `npm.cmd run build` in `gui/`.
2. Put the harness under `gui/tmp_verify_*.cjs`.
3. Require `gui/dist-electron/main/index.js`.
4. Wait for the main `BrowserWindow`.
5. Drive the renderer with `webContents.executeJavaScript()`.

## Lint and Format

```bash
ruff check .
black --check .
mypy .
pre-commit run --all-files --show-diff-on-failure
```

When only a small Python file set changed, prefer targeted checks on those
files first, then broaden if the change touched shared contracts.

### Deterministic CI tool versions

- `.pre-commit-config.yaml` is the source of truth for Ruff and Black revisions.
- Any direct Ruff or Black installation in GitHub Actions must use the same
  versions as the corresponding pre-commit hooks.
- Do not install unpinned formatters in CI. A newer formatter can reject files
  accepted by the pinned pre-commit environment and create false red checks.
- Pre-commit CI must use `--show-diff-on-failure` so any file mutation is visible
  in the job log.

### Dependency-audit scope

- The Python dependency scan must audit the repository project explicitly with
  a project path, for example `pip-audit --strict .`.
- A bare `pip-audit` command audits the current runner environment and must not
  be used as the repository security gate.
- Accepted vulnerability suppressions require an owner, rationale, expiry, and
  matching documentation in `docs/security/VULNERABILITY-SCAN.md`.

## Package Smoke Tests

The Electron package smoke test builds the packaged app and verifies the Python
bridge is reachable from the packaged output — not from the source tree.

**Script:** `tests/smoke/test_electron_package.sh`

### What the test verifies

1. Electron package builds without error (`electron-builder --dir`).
2. All 20 bridge scripts registered in `bridgeResolver.ts` are present in
   `resources/bridge/` inside the packaged output.
3. Python can execute a bridge script directly from the packaged path
   (`process.resourcesPath/bridge/`) with `PYTHONPATH=""` (no source-tree
   `bridge/` on the search path). A valid JSON response proves the bridge is
   reachable.
4. The packaged Electron app launches and emits the bridge validation startup
   signal: `[bridgeResolver] All bridge scripts validated successfully.`
   (best-effort; primary gate is step 3).

### Running locally

```bash
# Full run (builds first):
bash tests/smoke/test_electron_package.sh

# Skip rebuild (reuse existing gui/dist/ output):
SKIP_BUILD=1 bash tests/smoke/test_electron_package.sh
```

Prerequisites: Node.js, npm, Python venv activated or `.venv` in repo root,
`rex` package installed (`pip install .`).

### Running in CI (Linux)

On Linux CI, Electron requires a virtual display for the launch check:

```bash
REQUIRE_ELECTRON_SIGNAL=1 xvfb-run bash tests/smoke/test_electron_package.sh
```

`REQUIRE_ELECTRON_SIGNAL=1` causes the test to fail if the Electron startup
signal is not received within `SMOKE_TIMEOUT` seconds (default: 30).

### Platform behaviour

| Platform | Python bridge check | Electron launch signal |
|----------|--------------------|-----------------------|
| Windows  | Always works       | Appears on terminal but bash cannot capture it from GUI-subsystem binary stderr; best-effort only |
| Linux CI | Always works       | Requires `xvfb-run`; use `REQUIRE_ELECTRON_SIGNAL=1` |
| macOS    | Always works       | Should work without virtual display |

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SKIP_BUILD` | `0` | Skip npm ci / build / electron-builder |
| `REQUIRE_ELECTRON_SIGNAL` | `0` | Fail if Electron startup signal not received |
| `SMOKE_TIMEOUT` | `30` | Seconds to wait for Electron startup signal |

### What "bridge unreachable" means

The test exits non-zero if the packaged bridge script produces no output or
non-JSON output when executed by Python. This detects:

- Bridge scripts missing from `extraResources` (step 2).
- `rex` package not installed in the Python environment (step 3).
- `resolveBridgePath()` returning a wrong path in packaged mode (step 3).

## Git Hygiene

- Do not revert unrelated user changes.
- Keep generated assets and built outputs untouched unless the task explicitly
  requires them.
- For docs-only tasks, the final diff should contain Markdown/text changes only.
