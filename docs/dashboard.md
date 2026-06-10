# AskRex Dashboard

AskRex currently has one primary GUI and one Python/Flask service surface:

- Electron desktop app under `gui/` - current primary GUI.
- `rex-gui` - local Flask API/runtime surface with an incomplete, experimental browser dashboard at `/ui/`.

The old Tkinter entry points (`gui.py`, `run_gui.py`) are **archived** (moved to
`archived/tkinter_gui/`) and are no longer maintained. The root-level
`flask_proxy.py` is **deprecated** — use `rex-gui` instead.

## Electron Desktop App

The Electron app lives in `gui/` and is the current user-facing GUI.

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build and verify:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

Routes in the current app include home, devices, chat, voice, tasks, calendar,
reminders, memories, email, SMS, notifications, shopping, logs, history, usage,
integrations, settings, Home Assistant, quick actions, and about.

For Electron-only verification, build first and run a `gui/tmp_verify_*.cjs`
harness that requires `gui/dist-electron/main/index.js`.

## Python/Flask API and Experimental Browser Dashboard

Start:

```bash
rex-gui
```

Open:

```text
http://127.0.0.1:8765/ui/
```

The `/ui/` browser surface is incomplete in current testing and is not the
recommended main interface. Use it for API smoke checks or dashboard
compatibility work, not as the primary GUI.

Override the port with:

```bash
REX_GUI_PORT=8770 rex-gui
```

On Windows PowerShell:

```powershell
$env:REX_GUI_PORT = "8770"
rex-gui
```

## Current API Areas

`rex/gui_app.py` serves `/ui/` and local API endpoints including:

| Area | Example endpoint |
|---|---|
| Dashboard status | `/api/dashboard/status` |
| Chat | `/api/chat/send`, `/api/chat/history` |
| Logs | `/api/logs/stream`, `/api/logs/download` |
| Usage | `/api/usage` |
| Setup/auth/user profile | `/api/setup/status`, `/api/auth/login`, `/api/user/preferences` |
| Devices/Home Assistant | `/api/devices`, `/api/ha/test`, `/api/ha/save`, `/api/ha/states` |
| Quick actions | `/api/quick-actions` |
| Status stream | `/api/status/current`, `/api/status/stream` |
| History | `/api/history` |
| Integrations | `/api/integrations` |
| Calendar/email/SMS | `/api/calendar/events`, `/api/email/inbox`, `/api/sms/threads` |
| Capabilities/tools | `/api/capabilities`, `/api/tools` |

Smoke checks:

```bash
curl http://127.0.0.1:8765/api/dashboard/status
curl http://127.0.0.1:8765/api/tools
```

## Security

- Keep the dashboard bound to localhost unless deliberately deploying behind a
  reverse proxy and authentication layer.
- Keep secrets in `.env`, not in `config/rex_config.json`.
- Treat bridge/API responses from integrations as untrusted.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/ui/` does not load | Confirm `rex-gui` is running and the port is not in use |
| API calls fail | Check the terminal running `rex-gui` and run `python -m rex doctor` |
| Electron app shows stale behavior | Run `npm.cmd run build` in `gui/` |
| Electron type errors | Run `npm.cmd run typecheck` from `gui/` |
