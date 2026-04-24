# AskRex Assistant Quick Reference

Use this as a current command card for local development and verification.
Historical stabilization copy-file instructions are archived elsewhere and are
not part of the current setup flow.

## Install

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

Optional stacks:

```bash
pip install -r requirements-cpu.txt
pip install -r requirements-gpu-cu124.txt
pip install -r requirements-dev.txt
```

## Diagnose

```bash
python -m rex doctor
python -m rex --help
python -m rex tools --all
python scripts/security_audit.py
```

The supported Python line is `>=3.11,<3.12`.

## Main Entry Points

| Task | Command |
|---|---|
| Text chat | `python -m rex` |
| CLI help | `python -m rex --help` |
| Doctor | `python -m rex doctor` |
| Voice loop | `python rex_loop.py` |
| Electron GUI | `cd gui; npm.cmd run dev` |
| Python/Flask API and experimental web dashboard | `rex-gui` |
| TTS API | `rex-speak-api` |
| Tool server | `rex-tool-server` |
| Computer agent | `rex-agent` |
| Config helper | `rex-config show` |

## GUI

Electron development:

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Electron build/verification:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

For Electron-only harnesses, build first, then use a `gui/tmp_verify_*.cjs`
harness that requires `gui/dist-electron/main/index.js`.

Python/Flask API and experimental web dashboard:

```bash
rex-gui
```

Open:

```text
http://127.0.0.1:8765/ui/
```

The `/ui/` browser dashboard is incomplete in current testing. Use the Electron
app as the primary GUI; use `rex-gui` for local API routes and compatibility
checks.

## TTS API

Default port: `5005`.

```powershell
$env:REX_SPEAK_API_KEY = "example-secret  # pragma: allowlist secret"
rex-speak-api
```

```powershell
$headers = @{
  "Content-Type" = "application/json"
  "X-API-Key" = $env:REX_SPEAK_API_KEY
}
$body = @{ text = "Hello from Rex"; user = "default" } | ConvertTo-Json
Invoke-WebRequest -Uri http://127.0.0.1:5005/speak -Method POST -Headers $headers -Body $body -OutFile speech.wav
```

## Tool Server

Default port: `18790`.

```powershell
$env:REX_TOOL_API_KEY = "example-secret  # pragma: allowlist secret"
rex-tool-server
```

```powershell
$headers = @{
  "Content-Type" = "application/json"
  "Authorization" = "Bearer $env:REX_TOOL_API_KEY"
}
Invoke-WebRequest -Uri http://127.0.0.1:18790/rex/tools/time_now -Method POST -Headers $headers -Body "{}"
```

## Configuration

- Secrets belong in `.env`.
- Runtime settings belong in `config/rex_config.json`.
- The canonical wake-word config section is `wakeword`.
- Legacy non-secret environment variables can be migrated with:

```bash
rex-config migrate-legacy-env
```

## Common CLI Commands

```bash
rex memory recent 5
rex kb search "query"
rex scheduler list
rex email unread --limit 5
rex calendar upcoming --days 14
rex reminders list
rex notify send --priority normal --title "Update" --body "Done"
rex shopping list
rex history --limit 20
rex quick-actions list
rex wc orders list --site myshop
rex ha tts test --message "Hello from Rex"
```

## Current Repo Checks

```powershell
git status --short
python -m rex doctor
python -m rex --help
cd gui
npm.cmd run typecheck
npm.cmd run build
```
