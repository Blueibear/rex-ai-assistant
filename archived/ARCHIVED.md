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

### (empty — no items archived yet)

Items will be added here as US-006, US-007, and later stories move code into this directory.

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
