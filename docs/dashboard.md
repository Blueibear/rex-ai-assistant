# AskRex Dashboard

AskRex has two current GUI surfaces:

- Python web dashboard served by `rex-gui`
- Electron desktop app under `gui/`

The old Tkinter entry points (`gui.py`, `run_gui.py`) are deprecated. The legacy
`flask_proxy.py` surface is kept for compatibility and should not be treated as
the primary dashboard.

## Python Web Dashboard

Start:

```bash
rex-gui
```

Open:

```text
http://127.0.0.1:8765/ui/
```

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

## Electron Desktop App

The Electron app lives in `gui/`.

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
