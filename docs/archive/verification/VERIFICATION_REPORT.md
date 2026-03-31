# Verification Report: Dependabot + Pipenv Lock Audit

Date: 2026-02-20
Branch: `audit/verify-claude-dependabot`

## What Claude claimed
- `Pipfile` was made Dependabot-friendly by removing CUDA/ML lock-breakers (`openai-whisper`, `torch`, `torchvision`, `torchaudio`, `openwakeword`, `TTS`).
- `cryptography` and `pillow` were bumped to remediate active Dependabot alerts.
- `Pipfile.lock` was regenerated successfully on clean Linux.

## What was verified as true
- `Pipfile` does **not** include `openai-whisper`, `torch`, `torchvision`, `torchaudio`, `openwakeword`, or `TTS` in `[packages]`.
- `Pipfile` pins security dependencies at or above requested values:
  - `cryptography ==46.0.5` (>= 44.0.2)
  - `pillow ==12.1.1` (>= 11.2.1)
- `Pipfile` includes `werkzeug ==3.1.6`.
- Clean lock regeneration succeeds with Pipenv:
  - `PIPENV_IGNORE_VIRTUALENVS=1 pipenv lock --clear` completed successfully.
- Resulting dependency graph has no `openai-whisper`, `torch`, `triton`, or `nvidia-*` CUDA stack.
- `Pipfile.lock` hash is consistent (`b1de75d...`) and remains unchanged after lock regeneration.

## What was false or incomplete
- Could not reproduce the reported Windows `openai-whisper` build failure from current repository state.
- No lock conflict could be directly replayed against `origin/dependabot/pip/werkzeug-3.1.6` because this local clone has no configured `origin` remote.

## Root-cause analysis of "Windows tries to build openai-whisper"
Repository inspection indicates no active path to resolve `openai-whisper` during Pipenv locking:
1. `Pipfile` has no `path = "."` or `editable = true` local project entry.
2. `pyproject.toml` does **not** include `openai-whisper` in core `[project].dependencies`; it appears only in optional extras (`ml`, `full`).
3. Fresh lock and installed graph contain no `openai-whisper` transitive parent.

Most likely explanation for the maintainer's failure is a stale/local state (e.g., older checkout, cached pre-fix lock inputs, or locking from a different Pipfile context), not the current committed sources.

## Exactly what changed in this audit
- Added this report file to document command-by-command verification and outcomes.

## Commands run and outcomes
- `git grep -n -E "openai-whisper|whisper" -- .` ✅ (references found only in docs/code/optional extras; not in Pipfile packages)
- `python -m pipenv lock --clear` ✅ (succeeds in current repo)
- `PIPENV_IGNORE_VIRTUALENVS=1 pipenv lock --clear` ✅ (succeeds from clean pipenv-managed env)
- `pipenv graph | rg -n "openai-whisper|whisper|torch|triton|nvidia"` ✅ (no matches)
- `pytest -m "not slow and not audio and not gpu" --cov=rex --cov-report=term-missing --cov-report=xml` ✅ (928 passed, 29 skipped)
- `pipenv run pip-audit` ✅ (No known vulnerabilities found)

## Notes on Dependabot lock conflict resolution
The correct conflict resolution method remains: regenerate `Pipfile.lock` after applying the `werkzeug` bump in source of truth, rather than manually selecting `_meta.hash.sha256`. Current state already reflects `werkzeug==3.1.6` and a clean regenerated lock.
