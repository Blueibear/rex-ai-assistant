# AskRex Assistant — Branding and Naming Ruleset

This document is the single authoritative source for product naming. All code, docs,
configuration, and CI must follow these rules. When in doubt, consult this file.

---

## Canonical Names

| Attribute              | Value                                               |
|------------------------|-----------------------------------------------------|
| Product name           | **AskRex Assistant**                                |
| Package name (pip)     | `askrex-assistant`                                  |
| CLI command (primary)  | `rex`                                               |
| GitHub repository URL  | `https://github.com/Blueibear/AskRex-Assistant`     |

### CLI alias policy

`rex` is the **canonical CLI command** and must remain the primary entry point.
It is registered in `pyproject.toml` under `[project.scripts]` as `rex`.
No migration to `askrex` is planned unless explicitly approved in a future story.

---

## Allowed Legacy Aliases

| Alias / old name       | Where it may remain              | Must change?                       |
|------------------------|----------------------------------|------------------------------------|
| `rex` (CLI name)       | All entry points, docs, scripts  | No — `rex` is the canonical name   |
| `rex_loop.py`          | Root module (legacy voice loop)  | No — kept for backward compat      |
| `rex/` package dir     | Python package path              | No — package directory is `rex/`   |

---

## Banned Names

These names are forbidden in new code, docs, pyproject.toml, and GitHub config.
Where they appear in existing files they must be replaced on contact.

| Banned name               | Replacement                              |
|---------------------------|------------------------------------------|
| `Rex AI Assistant`        | `AskRex Assistant`                       |
| `rex-ai-assistant`        | `askrex-assistant` (pip package name)    |
| `AskRex-Assistant` (slug) | Only allowed as the GitHub repo slug     |
| `askrex-assistant` (slug) | Not a valid repo slug — use the GitHub URL above |

---

## Usage by Context

| Context                   | Use                                      |
|---------------------------|------------------------------------------|
| Docs / README titles       | AskRex Assistant                         |
| pip install / PyPI         | `pip install askrex-assistant`           |
| GitHub clone URL           | `https://github.com/Blueibear/AskRex-Assistant.git` |
| CLI usage examples         | `rex`, `python -m rex`                   |
| Python import              | `import rex` / `from rex import ...`     |
| pyproject.toml `name`      | `askrex-assistant`                       |

---

## Enforcement

- CI lint step (`ruff`, `grep`) should flag new occurrences of banned names in `*.md` and `*.py`.
- PRs that introduce banned names must be corrected before merge.
- This file (`docs/BRANDING.md`) is the canonical reference; do not duplicate naming rules elsewhere.
