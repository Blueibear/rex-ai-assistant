# Changelog

## Unreleased

- **Rename:** Product renamed from "Rex AI Assistant" to **AskRex Assistant** (effective 2026-04-03).
  - `pyproject.toml` package name changed from `rex-ai-assistant` to the canonical pip name (see `docs/BRANDING.md`).
  - All documentation, badges, and clone URLs updated to `https://github.com/Blueibear/AskRex-Assistant`.
  - See `docs/BRANDING.md` for the full canonical naming ruleset.

## v1.0.0 - 2025-10-09
- CPU-first configuration with optional GPU path documented.
- Added wake-word, memory, proxy compatibility updates.
- Introduced rate limiter storage integration and /speak hardening.
- Added scripts/doctor.py environment checker.
- Test suite (46 tests) green under pytest + GitHub Actions.

