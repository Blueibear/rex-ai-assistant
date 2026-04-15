# PRD: Fix Failing CI Check — PR #231 (test: stabilize suite and align core behavior)

## Introduction

The CI suite on PR #231 is failing across a small, well-bounded set of tests. Every failure traces back to one of four root causes: wrong table-row format in a doc file, stale references to a deprecated launcher in two doc files, and a hard runtime crash in the wakeword embedding module when `torch` is absent from the environment. This PRD covers the minimal safe set of changes to green the suite. No architectural changes, no spec-unrelated cleanup.

---

## Goals

- All tests in the named failing files pass on the first full-suite run after these fixes.
- No currently-passing tests regress.
- Changes are confined to the exact files implicated by code evidence.

---

## Evidence Summary

| Test file | Root cause | Confirmed by |
|---|---|---|
| `test_us244_ui_surfaces.py` | `docs/UI_SURFACES.md` rows have wrong column text; `README.md` missing `"canonical GUI"` and contains `"run_gui.py"`; `INSTALL.md` contains `"run_gui.py"` | Direct string-match check against file content |
| `test_ww002_wakeword_train.py` | `save_embedding()` raises `RuntimeError("torch is required…")` when `torch` absent; base CI does not install the `ml` optional group | Traced `trainer.py → embedding.py`; torch is in optional `[ml]` group only |
| `test_us310_wakeword_sample.py` | Same root cause — `train_from_samples` calls `save_embedding` | Same trace |
| `test_us250_env_backup_hygiene.py` | **PASSES** — `.gitignore` rules already correct | Verified with `git check-ignore` |
| `test_us251_patch_hygiene.py` | **PASSES** — archived files exist in `docs/archive/housekeeping/` | `ls` confirmed |
| `test_gui_app.py` | **PASSES** — `rex/ui/dist/index.html` exists with valid HTML | `cat` confirmed |
| `test_us304_chat_stream_electron_verification.py` | **LIKELY PASSES on Windows CI** — `electron.cmd` and `tmp_verify_chat_stream.cjs` both present | Path existence confirmed |

---

## Non-Goals

- No changes to Electron, the React UI build, or Electron bridge scripts.
- No refactoring of `rex/wakeword/` beyond the embedding torch fallback.
- No changes to any test that is not demonstrably wrong.
- No changes to `AppConfig`, `Assistant`, `TextToSpeech`, or `ModelRouter` — those attributes (`_user_id`, `_router`, `_tts_output_device`) are already present in source.
- No changes to `run_gui.py` or `gui.py` — their deprecation headers are already correct.
- No changes to `.gitignore` or the `docs/archive/` directory layout.
- No addition of `torch` to base `[project.dependencies]`.

---

## User Stories

### US-001: Fix docs/UI_SURFACES.md table rows

**Description:** As a CI runner, I need the `docs/UI_SURFACES.md` table to contain the exact row strings that `test_us244_ui_surfaces.py::test_ui_surfaces_doc_exists_with_expected_rows` asserts.

**Root cause:** The current table uses different column values than what the six assertions require. The status column lacks bold `**...**` formatting and `— keep` suffixes; entry-point and notes text differ; the CLI surface row uses "CLI text chat" not "CLI (text chat)"; the Shopping PWA row cites `rex_speak_api.py` instead of `rex` or `rex-gui`; the Tkinter row's first column omits `(gui.py)`.

**What to change:** Replace the first `| Surface | Entry point | Status | Notes |` table in `docs/UI_SURFACES.md` so that it contains **at minimum** these six rows verbatim (other rows may remain):

```
| CLI (text chat) | `rex` | **Primary — keep** | Core text interface |
| Voice loop | `python rex_loop.py` | **Primary — keep** | Core voice interface |
| Web dashboard | `rex-gui` | **Primary GUI — keep** | React, modern, canonical |
| Shopping PWA | served by `rex` or `rex-gui` | **Optional feature — keep** | Functional feature surface |
| TTS API | `rex-speak-api` | **Service component — keep** | Required by voice loop |
| Tkinter window (`gui.py`) | `python run_gui.py` | **Deprecated** | Superseded by web dashboard |
```

The table header and all prose sections outside the table must be preserved unchanged.

**Acceptance Criteria:**
- [x] Run `pytest -q tests/test_us244_ui_surfaces.py::test_ui_surfaces_doc_exists_with_expected_rows` — passes.
- [x] All six verbatim strings above appear in `docs/UI_SURFACES.md`.
- [x] `grep -r "UI_SURFACES" tests/` confirms no other test file reads this doc; if any do, verify they still pass.
- [x] Typecheck passes (Markdown file — no Python typecheck needed; confirm no Python imports this doc).

---

### US-002: Fix README.md — add "canonical GUI" phrase and remove run_gui.py reference

**Description:** As a CI runner, I need `README.md` to satisfy all four assertions in `test_us244_ui_surfaces.py::test_readme_points_to_web_dashboard_as_canonical_gui`.

**Root cause:**
- `"canonical GUI"` does not appear anywhere in `README.md` (test: `assert "canonical GUI" in text` — FAILS).
- Line 105 reads `"The legacy Tkinter launchers \`python run_gui.py\` and..."` — test: `assert "run_gui.py" not in text` — FAILS.
- `` `rex-gui` `` is present — PASSES.
- `"legacy Tkinter launcher"` is a substring of `"legacy Tkinter launchers"` — PASSES.

**What to change in `README.md`:**
1. Rewrite line 105 to remove `run_gui.py` by name. For example, change `"The legacy Tkinter launchers \`python run_gui.py\` and \`python gui.py\` are deprecated."` to `"The legacy Tkinter launchers (\`gui.py\` and its entry point) are deprecated."`.
2. Add `"canonical GUI"` to the README's GUI section where `rex-gui` is introduced. For example: `` `rex-gui` is the canonical GUI for the Rex AI Assistant. ``

**Acceptance Criteria:**
- [x] Run `pytest -q tests/test_us244_ui_surfaces.py::test_readme_points_to_web_dashboard_as_canonical_gui` — passes.
- [x] `"canonical GUI"` appears in `README.md`.
- [x] `"run_gui.py"` does **not** appear anywhere in `README.md`.
- [x] `` `rex-gui` `` still appears in `README.md`.
- [x] `"legacy Tkinter launcher"` still appears (as a substring) in `README.md`.
- [x] Run `pytest -q tests/test_us244_ui_surfaces.py` — all four tests in the file pass.
- [x] Typecheck passes.

---

### US-003: Fix INSTALL.md — remove run_gui.py reference

**Description:** As a CI runner, I need `INSTALL.md` to not contain `"run_gui.py"` so that `test_us244_ui_surfaces.py::test_startup_docs_do_not_reference_run_gui_py` passes for both README and INSTALL.

**Root cause:** Line 142 of `INSTALL.md` reads: `` `python run_gui.py` and `python gui.py` are deprecated Tkinter paths... `` — test: `assert "run_gui.py" not in text` — FAILS.

**What to change in `INSTALL.md`:** Rewrite the line 142 sentence to remove `run_gui.py` while keeping the deprecation intent. For example: `"The legacy Tkinter launchers (\`gui.py\` and its entry point) are deprecated paths and should not be used for normal operation."`

**Acceptance Criteria:**
- [x] Run `pytest -q tests/test_us244_ui_surfaces.py::test_startup_docs_do_not_reference_run_gui_py` — passes.
- [x] `"run_gui.py"` does **not** appear in `INSTALL.md`.
- [x] The deprecation intent is preserved in prose.
- [x] Run `pytest -q tests/test_us244_ui_surfaces.py` — all four tests pass.
- [x] Typecheck passes.

---

### US-004: Make wakeword embedding save/load work without torch

**Description:** As a CI runner, I need `test_ww002_wakeword_train.py` and `test_us310_wakeword_sample.py` to pass in a base CI environment where the `ml` optional dependency group (including `torch`) is not installed.

**Root cause:** `rex/wakeword/embedding.py::save_embedding` and `load_embedding` unconditionally raise `RuntimeError("torch is required…")` when `_torch is None`. `trainer.py::train_from_samples` calls `save_embedding` at an unguarded call site (line 88) — the exception propagates uncaught, causing any test that calls `train_from_samples` to fail with `RuntimeError` rather than returning a dict. `test_ww002_wakeword_train.py` imports numpy directly (no `importorskip`) and expects `result["ok"] is True` from `train_from_samples`. `test_us310_wakeword_sample.py` has `pytest.importorskip("numpy")` but then also calls `train_from_samples` and expects success.

**What to change in `rex/wakeword/embedding.py`:**

Add a numpy-based fallback to both functions:

- `save_embedding(path, embedding)`: when `_torch is not None`, use the existing `_torch.save(embedding, Path(path))`. When `_torch is None`, use `np.save(path, embedding)`. Numpy serializes to the given path regardless of extension; this is sufficient for round-tripping embedding arrays in torch-free environments.

- `load_embedding(path)`: when `_torch is not None`, attempt the existing torch-load path. When `_torch is None`, use `np.load(path, allow_pickle=False)` and return the result as `np.float32`. Also add a fallback: if `_torch is not None` but the torch load raises (e.g., because the file was saved with numpy), fall back to `np.load`.

Both functions must keep their existing public signatures unchanged.

**Acceptance Criteria:**
- [x] Run `pytest -q tests/test_ww002_wakeword_train.py` in an environment **without** torch — all tests pass.
- [x] Run `pytest -q tests/test_us310_wakeword_sample.py` without torch — all tests that require numpy pass.
- [x] `train_from_samples` returns `{"ok": True, ...}` and produces a loadable embedding file when torch is absent.
- [x] `load_embedding` returns an `np.ndarray` with `ndim == 1` that round-trips correctly.
- [x] `test_train_from_samples_requires_min_positives` still returns `ok=False` with `"positive"` in the error (unchanged behavior).
- [ ] Run both test files **with** torch installed — all tests still pass (torch path must not regress).
- [x] `ruff check rex/wakeword/embedding.py` passes.
- [x] Typecheck passes on `rex/wakeword/embedding.py`.

---

### US-005: Run and verify the full failing test suite

**Description:** As the implementer, I need to confirm that all four fix clusters actually green the CI check with no new regressions.

**Workflow:**
1. After each story, run only that story's targeted test file(s) before moving on.
2. After all stories are complete, run the full pytest suite (or the exact CI command).
3. Run lint and typecheck on all changed `.py` files.

**Acceptance Criteria:**
- [x] `pytest -q tests/test_us244_ui_surfaces.py` — all 4 tests pass.
- [x] `pytest -q tests/test_us250_env_backup_hygiene.py` — all pass (no regression).
- [x] `pytest -q tests/test_us251_patch_hygiene.py` — all pass (no regression).
- [x] `pytest -q tests/test_gui_app.py` — all pass (no regression).
- [x] `pytest -q tests/test_ww002_wakeword_train.py` — all tests pass.
- [x] `pytest -q tests/test_us310_wakeword_sample.py` — all numpy-dependent tests pass.
- [x] Full `pytest -q` run has the same or fewer failures than before this fix set.
- [x] Lint passes on all changed `.py` files.
- [x] Typecheck passes on all changed `.py` files.

---

## Technical Considerations

- `docs/UI_SURFACES.md` has a prose section below the table (Electron GUI, Naming Notes). Preserve it intact; only the first table changes.
- `README.md` and `INSTALL.md` changes must be surgical — only the `run_gui.py` references and the `"canonical GUI"` addition. Do not rewrite whole sections.
- The numpy fallback in `embedding.py` must not change the public API signature or the `.pt` file extension convention.
- Use `allow_pickle=False` in `np.load` to avoid arbitrary code execution on untrusted `.npy`-format files.
- When both torch and the fallback path exist, try torch first and fall back to numpy — this preserves existing torch-saved `.pt` file compatibility.
- Do not add `torch` to the base `[project.dependencies]` in `pyproject.toml`.
