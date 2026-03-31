# Verification Report — Phase 7.1 Voice Identity MVP

## Scope
Verified the merged `master` implementation for Phase 7.1 Voice Identity MVP against the claimed deliverables and safety requirements, then applied minimal corrective fixes where behavior or safety gaps were found.

## Claims vs verified truth

### Claimed deliverables: files/modules
- ✅ `rex/voice_identity/embedding_backends.py` exists and implements:
  - `SyntheticEmbeddingBackend` (stdlib-only deterministic backend)
  - `SpeechBrainBackend` (optional + lazy model loading)
- ✅ `rex/voice_identity/calibration.py` exists and exposes `calibrate()` + `CalibrationReport`.
- ✅ `rex/voice_identity/optional_deps.py` includes backend factory `get_embedding_backend()` and optional import guards.
- ✅ `rex/cli.py` includes `rex voice-id` commands: `status`, `list`, `enroll`, `calibrate`.
- ✅ `rex/voice_loop.py` wires voice identity via `_build_voice_id_callback()` and degrades gracefully when disabled or unavailable.
- ✅ `docs/voice_identity.md`, `CLAUDE.md`, and `config/rex_config.example.json` contain voice-identity coverage.
- ✅ `tests/test_voice_id_mvp.py` exists with offline MVP tests.

### Runtime wiring + CLI UX
- ✅ `python -m rex --help` shows `voice-id` command.
- ✅ `python -m rex voice-id --help` lists expected subcommands.
- ✅ `python -m rex voice-id status` shows disabled-by-default behavior (`Enabled: no`).

### Safety / dependency expectations
- ✅ Default install dependency surfaces in `Pipfile` and `[project].dependencies` do not include heavy voice-ID ML deps (`speechbrain`, `torch`, `resemblyzer` absent from default sections).
- ✅ `speechbrain`/`resemblyzer` remain in optional extras (`[project.optional-dependencies].voice-id`).
- ✅ Optional heavy imports are lazy/guarded; importing voice identity modules does not require speechbrain/torch until selecting/using that backend.
- ✅ Enrollment writes require explicit `--yes` confirmation.
- ✅ Config writes from calibration require explicit `--write-config --yes`.

## Discrepancies found and fixed

### 1) Path traversal hardening gap in embedding storage
**Issue:**
`EmbeddingsStore._path_for(user_id)` accepted raw `user_id` path segments, allowing traversal-style values like `../...` to influence storage paths.

**Fix:**
Added strict `user_id` validation in `_path_for()`:
- must be non-empty string
- must be single path component without separators
- rejects `.` and `..`

Also made `load()` and `delete()` safely handle invalid IDs by logging and returning `None`/`False`.

### 2) Missing regression coverage for corrupted store data and path traversal
**Issue:**
MVP tests did not explicitly assert:
- path traversal rejection in store writes
- corrupted JSON store behavior in calibration path

**Fix:**
Added tests to `tests/test_voice_id_mvp.py`:
- `test_store_rejects_path_traversal_user_id`
- `test_corrupted_store_data_falls_back_to_config_defaults`

### 3) Docs mismatch on calibrate/no-users behavior
**Issue:**
`docs/voice_identity.md` stated calibrate with no users “returns an error”, but actual CLI prints a report and exits successfully.

**Fix:**
Updated docs wording to reflect shipped behavior.

## Command log and outcomes

### Required command set
1) Sanity + status
- `git status --porcelain` → started clean at verification start; dirty after intentional fixes.
- `python -m rex --help` → passed (after installing local package deps).

2) Tests
- `pytest -q` → environment-level failures remain unrelated to Phase 7.1 scope:
  - two async browser tests fail due missing async pytest plugin
  - repository-integrity tests fail when run with an intentionally dirty tree (expected during local patch authoring)
- `pytest -q tests/test_voice_id_mvp.py` → passed.

3) Lint/format on touched Python files
- `python -m ruff check rex/voice_identity/embeddings_store.py tests/test_voice_id_mvp.py` → fixed import ordering, now passes.
- `python -m black --check rex/voice_identity/embeddings_store.py tests/test_voice_id_mvp.py` → passes.

4) Security tooling
- `python scripts/security_audit.py` → passes secret/merge checks; placeholder findings are pre-existing repository heuristics output.

5) Dependency safety (best effort)
- Verified manually in `Pipfile` and `pyproject.toml` default dependency sections: no heavy voice-ID deps added.
- `pipenv` not installed in this environment, so `pipenv lock --clear` could not be run.

## Files changed in corrective patch
- `rex/voice_identity/embeddings_store.py`
  - Add `user_id` path-safety validation and safe handling on invalid IDs.
- `tests/test_voice_id_mvp.py`
  - Add regression tests for path traversal and corrupted store fallback behavior.
- `docs/voice_identity.md`
  - Correct no-users calibration behavior text.

## Risk assessment
- Low risk: narrowly scoped defensive validation + tests + docs correction.
- No changes to default dependencies.
- No changes to optional dependency loading strategy.
- No network behavior introduced.
