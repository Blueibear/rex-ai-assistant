# AskRex Assistant Usage Guide

Use Python 3.11 for all Python commands. If your system default `python` is newer, activate `.venv` or use `py -3.11` on Windows.

## Text Chat

```bash
rex
# or
python -m rex
```

Type `exit` or `quit` to stop.

Useful discovery commands:

```bash
rex --help
rex version --verbose
rex doctor
rex tools -v
rex usage
```

## Voice Mode

Install the ML/audio stack first:

```bash
pip install -r requirements-cpu.txt
```

Then run:

```bash
python rex_loop.py
```

Optional:

```bash
python rex_loop.py --user james
python rex_loop.py --enable-plugin web_search
```

Voice mode uses wake word detection, Whisper STT, the configured LLM provider, and TTS. Wake-word settings live under the canonical `wakeword` key in `config/rex_config.json`.

Current voice state:

- Hold to Talk is usable in current live testing.
- Wake-word mode is wired and can work end to end, but reliability and latency are still being improved.
- Long answers now use a cleaner spoken handoff to the on-screen transcript.
- Custom wake support is wired for built-in fallback, `custom_embedding`, and `custom_onnx`, but the repo does not ship a real `Hey Rex` custom asset by default.

Default custom wake asset locations:

- `config/wake_words/hey_rex/model.onnx`
- `config/wake_words/hey_rex/embedding.pt`

## Electron Desktop GUI

The Electron app under `gui/` is the current primary GUI.

Development:

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build and preview:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

Current Electron routes include Home, Chat, Voice, Tasks, Calendar, Reminders, Memories, Email, SMS, Notifications, Shopping List, Logs, History, Usage, Integrations, Settings, Home Assistant, Quick Actions, and About.

The Electron app uses Python bridge scripts at the repo root. Build before running `gui/tmp_verify_*.cjs` harnesses so `gui/dist-electron/main/index.js` is current.

## Python/Flask API and Experimental Web Dashboard

```bash
rex-gui
```

Default URL:

```text
http://127.0.0.1:8765/ui/
```

`rex-gui` starts a local Flask service used for API routes and compatibility checks. The browser dashboard at `/ui/` is incomplete in current testing and is not the primary GUI.

Override the port:

```bash
REX_GUI_PORT=9000 rex-gui
```

PowerShell:

```powershell
$env:REX_GUI_PORT=9000; rex-gui
```

## Configuration

Show resolved config:

```bash
rex-config show
```

Migrate old non-secret env vars into `config/rex_config.json`:

```bash
rex-config migrate-legacy-env --dry-run
rex-config migrate-legacy-env
```

Use:

- `config/rex_config.json` for runtime settings.
- `.env` for secrets.
- `profiles/*.json` for profile capabilities and overrides.

## Audio Devices

```bash
python audio_config.py --list
python audio_config.py --set-input 1
python audio_config.py --set-output 2
python audio_config.py --show
```

The GUI can also persist overlapping audio settings into `config/rex_config.json`.

## TTS API

Start:

```bash
REX_SPEAK_API_KEY=example-key  # pragma: allowlist secret rex-speak-api
```

PowerShell:

```powershell
$env:REX_SPEAK_API_KEY="example-key  # pragma: allowlist secret"
rex-speak-api
```

Default URL: `http://127.0.0.1:5005`

Example:

```bash
curl -X POST http://127.0.0.1:5005/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: example-key  # pragma: allowlist secret" \
  -d '{"text":"Hello from AskRex","user":"default","language":"en"}' \
  --output speech.wav
```

The endpoint also accepts `Authorization: Bearer <key>`.

## Tool Registry

```bash
rex tools
rex tools -v
rex tools --all
```

The local tool executor supports `time_now`, `weather_now`, and `web_search`. The OpenClaw-facing tool server exposes additional adapter tools.

See [tools.md](tools.md).

## OpenClaw Tool Server

Start:

```bash
REX_TOOL_API_KEY=example-key  # pragma: allowlist secret rex-tool-server
```

PowerShell:

```powershell
$env:REX_TOOL_API_KEY="example-key  # pragma: allowlist secret"
rex-tool-server
```

Example:

```bash
curl -X POST http://127.0.0.1:18790/rex/tools/time_now \
  -H "Content-Type: application/json" \
  -H "X-API-Key: example-key  # pragma: allowlist secret" \
  -d '{"args":{"location":"Dallas, TX"},"context":{}}'
```

## Workflows and Approvals

```bash
rex plan "check weather in Dallas"
rex plan "check weather in Dallas" --execute
rex workflows
rex approvals
rex approvals --approve <approval_id>
rex executor resume <workflow_id>
```

Workflow execution is policy-gated. Medium/high-risk actions may require approval before execution.

## Memory and Knowledge Base

```bash
rex memory recent 5
rex memory add facts '{"city":"Dallas"}'
rex memory search city
rex memory stats

rex kb ingest ./notes.txt --title "Notes" --tags notes
rex kb search "query"
rex kb list
```

Per-user profiles live under `Memory/<user_id>/`. Structured working/long-term memory lives under `data/memory/`. GUI chat history is backed by `data/history.db`.

## Integrations

Email:

```bash
rex email unread
rex email unread --limit 5 -v
rex email accounts
rex email test-connection
```

Calendar:

```bash
rex calendar upcoming
rex calendar upcoming --days 14 --conflicts
```

SMS:

```bash
rex msg send --to +15551234567 --body "Hello"
rex msg receive
```

GitHub:

```bash
rex gh repos
rex gh prs owner/repo
rex gh issue-create owner/repo --title "Bug" --body "Details"
rex gh pr-create owner/repo --head feature-branch --base master --title "Title" --body "Body"
```

WordPress/WooCommerce:

```bash
rex wp health --site myblog
rex wc orders list --site myshop
rex wc products list --site myshop --low-stock
rex wc orders set-status --site myshop --order-id 101 --status completed
rex wc coupons create --site myshop --code SAVE10 --amount 10 --type percent
```

Home Assistant:

```bash
rex ha tts test
rex ha tts test --message "Hello from Rex" --entity-id media_player.living_room
rex ha approve
```

Shopping list:

```bash
rex shopping list
rex shopping add milk --quantity 1
rex shopping clear
```

## Health Checks

```bash
python -m rex doctor
curl http://127.0.0.1:5005/health/live
curl http://127.0.0.1:18790/health/live
```

## Deprecated Paths

Do not use these for normal operation:

```bash
python run_gui.py
python gui.py
```

They launch the legacy Tkinter UI and are **archived** (moved to `archived/tkinter_gui/`). The Electron app (`cd gui && npm run dev`) is the current primary GUI. The `rex-gui` Flask service (developer-only) remains for local API and experimental browser-dashboard work.
