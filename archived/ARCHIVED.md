# Archived Components

This directory holds surfaces and modules that have been removed from the active codebase
but are preserved here for reference and for anyone who needs to recover them.

---

## What "archived" means

- **Not deleted** — the code still exists and can be inspected.
- **Not maintained** — archived files receive no bug fixes, security patches, or feature work.
- **Entry points removed** — no `pyproject.toml` entry point, import, or startup path references
  an archived file.  The component cannot be accidentally started by `rex`, `rex-gui`, or any
  other first-class command.
- **May be deleted** — archived items may be permanently deleted in a future major version once
  we are confident no one depends on them.  A notice will be added to the release notes.

---

## Archive log

Items are listed in the order they were archived.  Each entry explains what the component was,
why it was archived, and what (if anything) replaced it.

### Tkinter GUI (`archived/tkinter_gui/`)

**Archived in:** US-006 (2026-05-24)

**Files:**
- `archived/tkinter_gui/gui.py` — Tkinter desktop window (`AssistantGUI`)
- `archived/tkinter_gui/gui_settings_tab.py` — settings panel widget used by `gui.py`
- `archived/tkinter_gui/run_gui.py` — entry-point script that launched the Tkinter window

**Why archived:**
The Tkinter GUI was superseded by the React + Electron desktop GUI (`cd gui && npm run dev`) and
the Flask web dashboard (`rex-gui`). Tkinter is a legacy desktop toolkit that added a third UI
path with no corresponding maintenance. The entry point (`python run_gui.py`) was already
deprecated with a header comment; this story completes the removal by moving the files here.

**Replacement:**
Use `cd gui && npm.cmd run dev` for the Electron desktop GUI, or `rex-gui` for the Flask web
dashboard.

### Shopping PWA (`archived/shopping_pwa/`)

**Archived in:** US-007 (2026-05-24)

**Files:**
- `archived/shopping_pwa/shopping_pwa.py` — Flask Blueprint providing the mobile shopping-list
  PWA at `/shopping`, including PIN-auth, manifest, and REST API endpoints
- `archived/shopping_pwa/test_sl004_shopping_pwa.py` — test suite for the PWA surface

**Why archived:**
The shopping PWA was a standalone browser/PWA surface served by `rex_speak_api.py`. The core
shopping list functionality (`rex/shopping_list.py`, `rex/shopping_list_handler.py`) is retained
and still used by the assistant. Only the PWA surface layer is archived. The import in
`rex_speak_api.py` is already guarded by a `try/except` that logs a warning when the module is
absent.

**Replacement:**
Shopping list management is available through the assistant (`"add milk to shopping list"`) and
via the Electron GUI shopping routes.

### Compatibility Shims (`archived/compat_shims/`)

**Archived in:** US-020 (2026-05-25)

**Files:**
- `archived/compat_shims/patch_tts_torch_load.py` — one-shot script to patch Coqui TTS `io.py` for PyTorch 2.6 `torch.load()` compatibility
- `archived/compat_shims/patch_tts_transformers.py` — one-shot script to patch Coqui TTS `stream_generator.py` for Transformers 4.38+ import changes
- `archived/compat_shims/flask_proxy.py` — legacy Flask API / dashboard proxy application

**Why archived:**
The TTS patch scripts were one-shot compatibility fixes for PyTorch 2.6 and Transformers 4.38+. They are no longer needed because the current dependency pins avoid the problematic versions, and `rex/compat/transformers_shims.py` handles the Transformers compatibility at runtime.

`flask_proxy.py` is documented as a "Legacy compatibility proxy" in `docs/ARCHITECTURE.md`. The canonical Flask/API surface is `rex.gui_app`. The legacy service definition (`deploy/systemd/rex-api.service`) has been updated to reference the new archived path with `PYTHONPATH=/opt/rex-ai-assistant`.

**Replacement:**
- TTS patches: not needed; managed by `rex/compat/transformers_shims.py`
- Flask proxy: use `rex-gui` (entry point `rex.gui_app:main`)

### Root-level Legacy Wrappers (`archived/`)

**Archived in:** US-021 (2026-05-25)

**Files:**
- `archived/rex_assistant.py` — legacy CLI entry point mirroring old `rex_assistant.py` script; replaced by `rex.cli:main`
- `archived/conversation_memory.py` — `ConversationMemory` class; canonical copy moved to `rex/conversation_memory.py`
- `archived/plugin_loader.py` — dict-based plugin discovery utility; canonical copy moved to `rex/plugin_loader.py`
- `archived/audio_config.py` — audio device CLI utilities; canonical copy moved to `rex/audio_config.py`
- `archived/memory_utils.py` — re-export shim for `rex.memory_utils`; all callers updated to import from `rex.memory_utils` directly

**Why archived:**
These root-level files were either unused re-export shims or legacy scripts that duplicated logic
already living in the `rex/` package. Moving them here completes US-021's goal of reducing the
root-level `.py` file count to ≤12 while keeping all importers updated.

**Replacement:**
- `rex_assistant.py` → `rex.cli:main` (entry point `rex`)
- `conversation_memory.py` → `rex.conversation_memory`
- `plugin_loader.py` → `rex.plugin_loader`
- `audio_config.py` → `rex.audio_config`
- `memory_utils.py` → `rex.memory_utils`

### Root-level Deprecation Shims (`archived/`)

**Archived in:** US-023 (2026-05-25)

**Files:**
- `archived/logging_utils.py` — re-export shim for `rex.logging_utils`; all active callers already import from `rex.logging_utils` directly
- `archived/assistant_errors.py` — re-export shim for `rex.assistant_errors`; all active callers already import from `rex.assistant_errors` directly

**Why archived:**
Both files were pure re-export shims with `DeprecationWarning` added in US-020/US-021. After archiving `audio_config.py`, `conversation_memory.py`, `plugin_loader.py`, and `memory_utils.py` in earlier stories, no active code imported from either shim — only archived files referenced them. Archiving them reduces the root-level `.py` file count to 9 (≤10 target).

**Replacement:**
- `logging_utils.py` → `rex.logging_utils`
- `assistant_errors.py` → `rex.assistant_errors`

---

## Restoring an archived component

If you need to restore an archived component to the active codebase:

1. Move the file(s) back to their original location.
2. Re-add any necessary entry points in `pyproject.toml`.
3. Run `pip install -e .` to register the entry point.
4. Add or restore tests.
5. Remove the entry from this file.

---

## Questions?

Open an issue at https://github.com/Blueibear/AskRex-Assistant/issues and label it
`archived-component`.
