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
