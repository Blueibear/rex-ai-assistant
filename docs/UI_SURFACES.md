# UI Surfaces

This is the active inventory of UI and user-facing service surfaces that ship in this repository.

| Surface | Entry point | Status | Notes |
|---|---|---|---|
| CLI (text chat) | `rex` | **Primary — keep** | Core text interface |
| Voice loop | `python rex_loop.py` | **Primary — keep** | Core voice interface |
| Electron desktop GUI | `cd gui && npm.cmd run dev` | **Primary GUI — keep** | Current user-facing React/Electron GUI, backed by Python bridge scripts at repo root |
| Python/Flask local API and experimental web dashboard | `rex-gui` | Compatibility/API surface — keep | Starts Flask on `127.0.0.1:8765`, serves local `/api/...` routes and an incomplete `/ui/` browser dashboard; not the primary GUI |
| Shopping PWA | served by `rex` or `rex-gui` | **Optional feature — keep** | Functional feature surface |
| TTS API | `rex-speak-api` | **Service component — keep** | Required by voice loop |
| OpenClaw tool server | `rex-tool-server` | Service component | Tool adapter service on `127.0.0.1:18790`; requires `REX_TOOL_API_KEY` for tool calls |
| Windows computer agent | `rex-agent` | Optional service | Remote PC control agent API |
| Flask proxy | `python flask_proxy.py` | Legacy API/proxy surface | Kept for compatibility with proxy/search/contracts paths; not the normal desktop GUI entry point |
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

## Python/Flask Local API and Experimental Dashboard

`rex-gui` starts `rex/gui_app.py`. It remains useful for local Flask API routes, smoke tests, and compatibility work. The browser UI at `/ui/` is incomplete in current testing and is not the primary user-facing GUI.

## Naming Notes

- The canonical CLI remains `rex`; there is no `askrex` console script.
- The package name is `askrex-assistant`.
- Historical planning text may still mention `askrex-gui`, `askrex-speak-api`, or older `Rex AI Assistant` naming; follow [BRANDING.md](BRANDING.md) for new docs and code.
