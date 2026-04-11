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
```

When only a small Python file set changed, prefer targeted checks on those
files first, then broaden if the change touched shared contracts.

## Git Hygiene

- Do not revert unrelated user changes.
- Keep generated assets and built outputs untouched unless the task explicitly
  requires them.
- For docs-only tasks, the final diff should contain Markdown/text changes only.
