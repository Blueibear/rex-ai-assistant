# Surface Classification

This document classifies every runtime entry point and UI surface in the AskRex repository.
It is the authoritative reference for packaging, CI, docs, and support scope decisions.

Classifications are assigned at story US-REM-018. Any change to a classification must be made
here first, then propagated to docs, packaging config, and CI.

---

## Classification Definitions

| Classification | Meaning |
|----------------|---------|
| `shippable` | Supported, user-facing surface included in or required by a release artifact. Security fixes and regressions are P0. |
| `developer-only` | Supported for developer and operator use but not exposed to end users in the packaged app. Not in the packaged Electron installer. |
| `deprecated` | Still present for backward compatibility. Emits `DeprecationWarning` on use. Will be removed in a future release. No new features. Security fixes only. |
| `archived` | Not maintained. Entry points removed. Kept in `archived/` for reference. May be permanently deleted in a future major version. |
| `removed` | Deleted from the repo. Listed here as a historical record only. |

---

## Entry Points (`pyproject.toml [project.scripts]`)

| Entry Point | Module | Classification | Notes |
|-------------|--------|----------------|-------|
| `rex` | `rex.cli:main` | `shippable` | Primary CLI; first-class user-facing command. Text mode. |
| `rex-gui` | `rex.gui_app:main` | `shippable` | Flask API backend for the Electron desktop GUI. Required for GUI to function. Not a standalone browser app. |
| `rex-config` | `rex.config:cli` | `developer-only` | Config inspection and migration utility. Operator/developer use only. |
| `rex-speak-api` | `rex_speak_api:main` | `developer-only` | Standalone TTS API with auth and rate limiting. Backend service; not user-facing. |
| `rex-agent` | `rex.computers.agent_server:main` | `developer-only` | Optional remote PC control API. Not enabled by default; requires explicit configuration. |
| `rex-tool-server` | `rex.openclaw.tool_server:main` | `developer-only` | OpenClaw tool adapter backend at `/rex/tools/{tool_name}`. Backend service; not user-facing. |

---

## UI Surfaces

| Surface | Classification | Notes |
|---------|----------------|-------|
| `gui/` (React + Electron desktop app) | `shippable` | Primary user-facing release artifact. The supported install path for end users. |
| `rex/ui/` (Vite/React developer dashboard) | `developer-only` | Confirmed developer-only by `package.json` description: "Developer-only surface. Not included in packaged Electron app." Served by `rex-gui` at `/ui/`. Not bundled in the packaged Electron installer. |

---

## Root-level Scripts and Entry-point Files

| File | Classification | Notes |
|------|----------------|-------|
| `rex_loop.py` | `shippable` | Full voice loop entry point (`wake word -> STT -> LLM -> TTS`). Voice mode startup path (`python rex_loop.py`). Advanced/power-user surface. |
| `rex_speak_api.py` | `developer-only` | Same function as `rex-speak-api` entry point. Root-level script kept for direct invocation (`python rex_speak_api.py`). |
| `wsgi.py` | `developer-only` | WSGI deployment entry point for `rex-gui`. Operator use for production deployments (e.g., gunicorn). |
| `sitecustomize.py` | `developer-only` | Windows UTF-8 encoding fix; applied automatically at interpreter start. Not a user-facing entry point. |
| `conftest.py` | `developer-only` | pytest root conftest with shared fixtures. Test infrastructure only. |
| `setup.py` | `deprecated` | Legacy setuptools stub. Packaging is handled by `pyproject.toml`. Present only for legacy tool compatibility. |
| `voice_loop.py` | `deprecated` | Root-level re-export shim for `AsyncRexAssistant` backward compatibility. Emits `DeprecationWarning` on import. Canonical implementation: `rex.voice_loop`. Scheduled for removal — see US-REM-020. |
| `llm_client.py` | `deprecated` | Root-level re-export shim for `rex.llm_client`. Emits `DeprecationWarning` on import. Canonical implementation: `rex.llm_client`. Scheduled for removal — see US-REM-020. |
| `config.py` | `deprecated` | Root-level re-export shim for `rex.config`. Emits `DeprecationWarning` on import. Canonical implementation: `rex.config`. Scheduled for removal — see US-REM-020. |

---

## Archived Content (`archived/`)

All content under `archived/` is classified as `archived`. None of these files are referenced by
any entry point, import, or startup path in the active codebase.

| Path | Original Purpose | Archived In | Replacement |
|------|-----------------|-------------|-------------|
| `archived/tkinter_gui/gui.py` | Tkinter desktop window (`AssistantGUI`) | US-006 | `gui/` (Electron app) |
| `archived/tkinter_gui/gui_settings_tab.py` | Tkinter settings panel widget | US-006 | `gui/` (Electron app) |
| `archived/tkinter_gui/run_gui.py` | Tkinter GUI launch script | US-006 | `gui/` (Electron app) |
| `archived/shopping_pwa/shopping_pwa.py` | Flask Blueprint for mobile shopping-list PWA | US-007 | Electron GUI shopping routes |
| `archived/shopping_pwa/test_sl004_shopping_pwa.py` | Test suite for shopping PWA surface | US-007 | N/A (archived with PWA) |
| `archived/compat_shims/patch_tts_torch_load.py` | One-shot PyTorch 2.6 TTS patch | US-020 | `rex/compat/transformers_shims.py` |
| `archived/compat_shims/patch_tts_transformers.py` | One-shot Transformers 4.38+ TTS patch | US-020 | `rex/compat/transformers_shims.py` |
| `archived/compat_shims/flask_proxy.py` | Legacy Flask API compatibility proxy | US-020 | `rex-gui` (`rex.gui_app:main`) |
| `archived/rex_assistant.py` | Legacy CLI entry point | US-021 | `rex.cli:main` |
| `archived/conversation_memory.py` | `ConversationMemory` class | US-021 | `rex.conversation_memory` |
| `archived/plugin_loader.py` | Dict-based plugin discovery | US-021 | `rex.plugin_loader` |
| `archived/audio_config.py` | Audio device CLI utilities | US-021 | `rex.audio_config` |
| `archived/memory_utils.py` | Re-export shim for `rex.memory_utils` | US-021 | `rex.memory_utils` |
| `archived/logging_utils.py` | Re-export shim for `rex.logging_utils` | US-023 | `rex.logging_utils` |
| `archived/assistant_errors.py` | Re-export shim for `rex.assistant_errors` | US-023 | `rex.assistant_errors` |

---

## Summary Counts

| Classification | Count |
|----------------|-------|
| `shippable` | 4 |
| `developer-only` | 9 |
| `deprecated` | 4 |
| `archived` | 15 |
| `removed` | 0 |
| **Total** | **32** |

---

## Change Log

| Date | Story | Change |
|------|-------|--------|
| 2026-06-01 | US-REM-018 | Initial classification of all surfaces |
