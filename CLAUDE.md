# CLAUDE.md

## Setup & Installation (GPU)
- Do **not** reintroduce GPU extras like `.[gpu-cu118]`, `.[gpu-cu121]`, or `.[gpu-cu124]` unless they are fully functional with the required PyTorch index behavior.
- Supported GPU installs are requirements-file based because CUDA wheels require `--extra-index-url`.
- Canonical commands:
  - `pip install -r requirements-gpu-cu124.txt` (CUDA 12.4)
  - `pip install -r requirements-gpu.txt` (CUDA 11.8)
- Keep GPU guidance aligned across `INSTALL.md`, `README.md`, and requirements files in the same PR.

## Testing
- Pytest configuration source-of-truth is `[tool.pytest.ini_options]` in `pyproject.toml`; do **not** reintroduce `pytest.ini`.
- Canonical local test setup: `python -m pip install -e '.[dev]'` before running tests, because default pytest options include coverage flags.
- Canonical test command: `pytest -q`.
- Do not make default dev commands depend on optional plugins unless those plugins are in dev extras **and** documented; otherwise keep those flags CI-only.
- Tests must not write to tracked repo fixtures/files (for example under `data/`). Use `tmp_path`/temp copies; only commit fixture updates when intentional.

## Security Audit
- Run security checks with: `python scripts/security_audit.py`.
- The audit scans common source/doc/config file types, checks merge markers/placeholders/secrets, and excludes common build/venv/log directories.
- Current behavior intentionally skips Markdown fenced code blocks for merge-marker and secret checks to reduce false positives.
- Any heuristic change in `scripts/security_audit.py` must include corresponding updates in `tests/test_security_audit.py` with both true-positive and false-positive coverage.
- Do not add silent blind spots: if skipping new content classes, document rationale in code/comments and add explicit tests that prove intended detection is still preserved.

## CI Rules
- Do not add soft-pass CI patterns that mask failures (for example `|| echo` on required checks).
- Node/JavaScript CI jobs must not be added unless a real Node project exists (for example `package.json` at repo root or explicit subpath).
- Any Node job that is added must fail on real lint/test/build errors.

## Docs Consistency
- Integration docs for `email`, `calendar`, `messaging`, and `notifications` must include a top-level **Implementation Status** block (`Beta`, `Stub`, or `Production-ready`).
- When README `Current Limitations` changes, update the corresponding integration docs in the same PR.
- Avoid capability wording that overstates readiness relative to implementation status.

## Pydantic v2 Conventions
- Use Pydantic v2 style: `ConfigDict`, `field_serializer`, and `model_dump`/`model_dump_json`.
- Do not introduce deprecated v1 patterns such as class-based `Config` with `json_encoders` in new or modified models.
- If serialization behavior changes, add or update tests to lock expected output.
