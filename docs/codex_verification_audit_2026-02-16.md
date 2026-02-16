# Codex Verification + Fresh Repo Audit (2026-02-16)

This document captures verification of claimed Claude changes and a fresh full-repo review.

## Commands Executed

- `python -m pip install -e .` ✅
- `python -m rex --help` ✅
- `python scripts/security_audit.py` ✅ (exit 0; placeholder warnings remain informational)
- `pytest -q` ⚠️ initially failed because `pytest-cov` was missing from environment (addopts requires coverage options)
- `python -m pip install -e '.[dev]'` ✅ (installed pytest-cov/pytest-asyncio)
- `pytest -q` ✅ (`886 passed, 29 skipped`)
- `python -m compileall -q .` ✅
- `python -m rex doctor` ✅ (warnings for missing `.env`, API keys, ffmpeg)
- `python - <<'PY' ...` serializer check for naive + aware datetimes ✅

## Key Verification Outcomes

- HIGH-02: Verified.
- CRIT-01: Mostly verified; README/CLI messaging improved, but `docs/notifications.md` still reads as production-capable at top-level.
- HIGH-03: Verified behavior from tests and script run; regex strictness improved.
- MED-04: Verified (`pytest.ini` removed; config in `pyproject.toml`).
- MED-05: Verified (`node-ci` removed).
- LOW-06: Verified (`ConfigDict` + `field_serializer`; tests cover no deprecation warnings).

## Notable Additional Findings

1. `pytest -q` fails in non-dev environments because `pyproject.toml` hardcodes coverage addopts requiring `pytest-cov`.
2. Security audit currently skips all Markdown fenced code blocks for secrets; this can mask real leaked keys pasted in docs.
3. Documentation consistency issue: notification docs still overstate channel readiness compared with README beta/stub framing.

## Recommended Backlog

1. Move coverage options out of default `addopts` into CI command or guard by plugin presence.
2. Restrict Markdown secret-skip behavior to known docs/test-pattern files or add allowlist/flag.
3. Add explicit beta/stub disclaimer block to `docs/notifications.md` and `docs/messaging.md` intro sections.
