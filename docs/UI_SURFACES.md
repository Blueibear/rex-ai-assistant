# UI Surfaces

This is the active inventory of UI and user-facing service surfaces that ship in this repository.

| Surface | Entry point | Status | Notes |
|---|---|---|---|
| CLI (text chat) | `rex` | **Shippable** | Core text interface; canonical CLI entry point |
| Voice loop | `python rex_loop.py` | **Developer-only** | Source voice entry point; wake-word mode remains beta. Packaged Electron Hold-to-Talk is the supported end-user voice path |
| Electron desktop GUI | Installed AskRex app or `cd gui && npm.cmd run dev` | **Shippable** | Primary packaged end-user GUI; development command is source-only |
| Python/Flask local API and experimental web dashboard | `rex-gui` | **Developer-only** | Local Flask API plus incomplete `/ui/` browser dashboard; not the primary GUI and not required by Electron |
| Shopping PWA | served by `rex` or `rex-gui` | **Archived** | Surface archived to `/archived/shopping_pwa/`; shopping list logic (`rex/shopping_list.py`) remains |
| TTS API | `rex-speak-api` | **Developer-only** | Optional standalone authenticated TTS service; Electron voice does not require it |
| OpenClaw tool server | `rex-tool-server` | **Experimental** | Optional tool adapter; off by default and subject to explicit gateway configuration |
| Windows computer agent | `rex-agent` | **Developer-only** | Optional permission-gated remote PC control service |
| Flask proxy | `python flask_proxy.py` | **Deprecated** | Root-level legacy API/proxy; scheduled for removal. Use `rex-gui` instead. |
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

- Hold to Talk is the supported voice path and covers STT, response TTS, selected-device playback, cancellation, replay, barge-in, device-loss fallback, repeated turns, and structured timing.
- Wake listening is present but still being stabilized for reliability and latency.
- Custom wake support is wired, but a real custom asset is still required before a `Hey Rex` custom path becomes active.

Electron bridge scripts are resolved by `gui/src/main/bridgeResolver.ts`. Development uses canonical scripts under `bridge/`; packaged Windows uses `resources/bridge/`. Root-level `rex_*_bridge.py` files are compatibility wrappers for imports and tests, not the Electron runtime path.

The packaged Voice artifact uses `resources/python/python.exe`, containing the installed AskRex wheel and pinned CPU voice dependencies. Electron does not spawn Flask and does not require machine Python, Node, a source checkout, or a neighboring `.venv`.

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

No renderer `/api/` call sites are allowed. The guard fails CI on regression.

## Naming Notes

- The canonical CLI remains `rex`; there is no `askrex` console script.
- The package name is `askrex-assistant`.
- Historical planning text may still mention `askrex-gui`, `askrex-speak-api`, or older `Rex AI Assistant` naming; follow [BRANDING.md](BRANDING.md) for new docs and code.
