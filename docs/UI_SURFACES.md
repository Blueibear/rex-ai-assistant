# UI Surfaces

This is the active inventory of UI and user-facing service surfaces that ship in this repository.

| Surface | Entry point | Status | Notes |
|---|---|---|---|
| CLI (text chat) | `rex` | **Primary — keep** | Core text interface |
| Voice loop | `python rex_loop.py` | **Primary — keep** | Core voice interface |
| Electron desktop GUI | `cd gui && npm.cmd run dev` | **Primary GUI — keep** | Current user-facing React/Electron GUI, backed by Python bridge scripts at repo root |
| Python/Flask local API and experimental web dashboard | `rex-gui` | Compatibility/API surface — keep | Starts Flask on `127.0.0.1:8765`, serves local `/api/...` routes and an incomplete `/ui/` browser dashboard; not the primary GUI |
| Shopping PWA | served by `rex` or `rex-gui` | **Archived** | Surface archived to `/archived/shopping_pwa/`; shopping list logic (`rex/shopping_list.py`) remains |
| TTS API | `rex-speak-api` | **Service component — keep** | Required by voice loop |
| OpenClaw tool server | `rex-tool-server` | Service component | Tool adapter service on `127.0.0.1:18790`; requires `REX_TOOL_API_KEY` for tool calls |
| Windows computer agent | `rex-agent` | Optional service | Remote PC control agent API |
| Flask proxy | `python flask_proxy.py` | **Deprecated** | Root-level legacy API/proxy; scheduled for removal. Use `rex-gui` instead. See SURFACE-CLASSIFICATION.md. |
| Tkinter window (`gui.py`) | `python archived/tkinter_gui/run_gui.py` | **Archived** | Superseded by the Electron desktop GUI; moved to `/archived/tkinter_gui/` |

## Electron GUI

Development:

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build/preview:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

The Electron shell currently includes routes for Home, Chat, Voice, Tasks, Calendar, Reminders, Memories, Email, SMS, Notifications, Shopping List, Logs, History, Usage, Integrations, Settings, Home Assistant, Quick Actions, and About.

Current voice state in the Electron app:

- Hold to Talk is usable in live testing.
- Wake listening is present but still being stabilized for reliability and latency.
- Long voice answers now hand off cleanly to the on-screen transcript.
- Custom wake support is wired, but a real custom asset is still required before a `Hey Rex` custom path becomes active.

Electron bridge scripts are resolved by `gui/src/main/bridgeResolver.ts` and include root-level scripts such as `rex_chat_stream_bridge.py`, `rex_tasks_bridge.py`, `rex_reminders_bridge.py`, `rex_shopping_list_bridge.py`, `rex_memories_bridge.py`, `rex_voice_bridge.py`, `rex_voice_enrollment_bridge.py`, `rex_wakeword_list_bridge.py`, and `rex_stt_bridge.py`.

For Electron-only verification, run `npm.cmd run build` first, then use harnesses under `gui/tmp_verify_*.cjs` so `gui/dist-electron/main/index.js` matches the TypeScript sources.

## IPC-backed Pages

The following Electron renderer pages communicate with the main process via typed IPC rather than raw `fetch('/api/...')` calls:

| Page | IPC method | Notes |
|------|-----------|-------|
| About (`AboutPage.tsx`) | `window.rex.getAppStatus()` → `rex:getAppStatus` | Returns `{ version, python_version, platform }` |
| Devices (`DevicesPage.tsx`) | `window.rex.getDevices()` → `rex:getDevices` | Reads `config/device_aliases.json`; returns `{ ok, devices }` |

## Python/Flask Local API and Experimental Dashboard

`rex-gui` starts `rex/gui_app.py`. It remains useful for local Flask API routes, smoke tests, and compatibility work. The browser UI at `/ui/` is incomplete in current testing and is not the primary user-facing GUI.

## Renderer IPC Policy

The Electron renderer communicates with the main process through typed IPC, not via raw `fetch('/api/...')` calls. Raw `/api/` fetches do not work in the packaged app (file:// protocol) and are guarded by CI.

**Guard:** `scripts/check_no_renderer_api_fetch.py` scans `gui/src/**/*.{ts,tsx,js,jsx}` for `fetch('/api`, `fetch("/api`, and fetch(`` `/api `` patterns. The CI job `gui-no-raw-api` runs this check on every PR.

**Allowlist:** `gui/src/ALLOWED_API_FETCHES.txt` lists the temporary baseline of call sites that have not yet been migrated (format: `rel/path:lineno  # justification`). Each migration story (US-004 through US-010) removes its entry when complete.

## Naming Notes

- The canonical CLI remains `rex`; there is no `askrex` console script.
- The package name is `askrex-assistant`.
- Historical planning text may still mention `askrex-gui`, `askrex-speak-api`, or older `Rex AI Assistant` naming; follow [BRANDING.md](BRANDING.md) for new docs and code.
