# AskRex Assistant Operations Runbook

This runbook covers the current local operating surfaces for AskRex Assistant:
starting, stopping, health checks, common recovery steps, and where each surface
listens. It intentionally treats `flask_proxy.py` and Tkinter as legacy surfaces;
use `rex-gui` or the Electron app for day-to-day UI work.

For installation, see `INSTALL.md` and `docs/advanced-install.md`.
For configuration, see `CONFIGURATION.md` and `docs/environment-variables.md`.

## Process Inventory

| Surface | Entry point | Default address | Purpose |
|---|---|---:|---|
| CLI chat | `python -m rex` or `rex chat` | none | Text chat and command access |
| Voice loop | `python rex_loop.py` | none | Wake word, STT, LLM, and TTS loop |
| Python web dashboard | `rex-gui` | `http://127.0.0.1:8765/ui/` | Browser dashboard and local API |
| Electron desktop app | `npm.cmd run dev` or built app under `gui/` | app window | Desktop React UI with Electron bridge |
| Rex Speak API | `rex-speak-api` | `http://127.0.0.1:5005` | Authenticated TTS `/speak` API |
| Rex tool server | `rex-tool-server` | `http://127.0.0.1:18790` | Authenticated OpenClaw-style tool endpoint |
| Legacy Flask proxy | `python flask_proxy.py` | `http://0.0.0.0:5000` | Compatibility API/proxy surface |

Only start the processes needed for the workflow you are running.

## Start Procedures

Activate the Python 3.11 virtual environment first.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Run diagnostics:

```bash
python -m rex doctor
python -m rex --help
```

Start text chat:

```bash
python -m rex
```

Start the voice loop:

```bash
python rex_loop.py
```

Start the Python web dashboard:

```bash
rex-gui
```

Open:

```text
http://127.0.0.1:8765/ui/
```

Start the Electron app:

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build Electron production assets:

```powershell
cd gui
npm.cmd run build
npm.cmd run preview
```

For Electron-only verification harnesses, run `npm.cmd run build` in `gui/`
first, then boot `gui/dist-electron/main/index.js` from a
`gui/tmp_verify_*.cjs` harness as described in the repo instructions.

Start the TTS API:

```bash
set REX_SPEAK_API_KEY=example-secret  # pragma: allowlist secret
rex-speak-api
```

PowerShell equivalent:

```powershell
$env:REX_SPEAK_API_KEY = "example-secret  # pragma: allowlist secret"
rex-speak-api
```

Start the Rex tool server:

```bash
set REX_TOOL_API_KEY=example-secret  # pragma: allowlist secret
rex-tool-server
```

PowerShell equivalent:

```powershell
$env:REX_TOOL_API_KEY = "example-secret  # pragma: allowlist secret"
rex-tool-server
```

## Health Checks

Python web dashboard:

```bash
curl http://127.0.0.1:8765/api/dashboard/status
curl http://127.0.0.1:8765/api/tools
```

TTS API:

```bash
curl http://127.0.0.1:5005/health/live
curl http://127.0.0.1:5005/health/ready
```

Tool server:

```bash
curl http://127.0.0.1:18790/health/live
curl http://127.0.0.1:18790/health/ready
```

Legacy Flask proxy:

```bash
curl http://127.0.0.1:5000/health/live
curl http://127.0.0.1:5000/health/ready
```

## TTS API Smoke Test

PowerShell:

```powershell
$headers = @{
  "Content-Type" = "application/json"
  "X-API-Key" = $env:REX_SPEAK_API_KEY
}
$body = @{ text = "Hello from Rex"; user = "default" } | ConvertTo-Json
Invoke-WebRequest `
  -Uri http://127.0.0.1:5005/speak `
  -Method POST `
  -Headers $headers `
  -Body $body `
  -OutFile speech.wav
```

Bash:

```bash
curl -X POST http://127.0.0.1:5005/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $REX_SPEAK_API_KEY" \
  -d '{"text":"Hello from Rex","user":"default"}' \
  --output speech.wav
```

## Tool Server Smoke Test

```bash
curl -X POST http://127.0.0.1:18790/rex/tools/time_now \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $REX_TOOL_API_KEY" \
  -d '{}'
```

The available tools depend on optional integration configuration. Check the
local registry with:

```bash
python -m rex tools --all
```

## Stop and Restart

For foreground processes, press `Ctrl+C`.

On Windows, inspect Python processes:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "rex|rex_speak|tool_server|gui_app" } |
  Select-Object ProcessId, CommandLine
```

On macOS/Linux:

```bash
pgrep -af "rex|rex_speak|tool_server|gui_app"
```

Prefer a normal interrupt or `SIGTERM`. Avoid force-killing unless the process
is stuck and you have captured enough logs to diagnose the failure.

## Logs

Most surfaces write to stdout/stderr. The Python web dashboard and services also
use the repo logging utilities, so log format may be JSON when JSON logging is
enabled.

Useful local patterns:

```bash
python -m rex doctor
python -m rex tools --all
python scripts/security_audit.py
```

For the Electron app:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

## Common Incidents

| Symptom | Likely cause | Recovery |
|---|---|---|
| Unsupported Python version | Running with Python 3.12+ or system Python | Recreate `.venv` with Python 3.11 |
| `rex-speak-api` exits immediately | `REX_SPEAK_API_KEY` missing | Set a random API key in `.env` or shell |
| TTS returns 401 | Missing or wrong `X-API-Key` / bearer token | Use the same key as `REX_SPEAK_API_KEY` |
| TTS returns 429 | Rate limit hit | Increase `REX_SPEAK_RATE_LIMIT` or slow the caller |
| Dashboard opens but integrations are empty | Optional integration config missing | Check `config/rex_config.json` and relevant secrets |
| Voice loop fails before audio capture | Audio device or optional ML stack missing | Run `python audio_config.py --list` and `python -m rex doctor` |
| Electron harness shows stale behavior | Built Electron files are stale | Run `npm.cmd run build` in `gui/` before the harness |
| Legacy `flask_proxy.py` exits on migrations | Compatibility surface migration check failed | Prefer `rex-gui`; if maintaining proxy, inspect `rex/migrations.py` and existing DB state |

## Post-Restart Checklist

- `python -m rex doctor` completes with only expected optional warnings.
- `python -m rex --help` lists commands.
- `rex-gui` serves `http://127.0.0.1:8765/ui/` when the web dashboard is needed.
- `rex-speak-api` health endpoints respond on `5005` when TTS API is needed.
- `rex-tool-server` health endpoints respond on `18790` when tool serving is needed.
- Electron builds with `npm.cmd run build` before Electron-only verification.
