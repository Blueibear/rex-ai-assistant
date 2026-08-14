# AskRex Assistant Deployment Guide

AskRex is primarily a local-first assistant. Deploy only the surfaces you need:
CLI/voice, the Electron desktop app, the Python/Flask local API service, TTS API,
tool server, or computer agent. The `rex-gui` browser dashboard at `/ui/` is
incomplete in current testing; use Electron as the primary GUI. The legacy Flask
proxy remains in the repo for compatibility but is not the recommended
dashboard entry point.

## Prerequisites

| Requirement | Version / note |
|---|---|
| Python | `>=3.11,<3.12` |
| Git | Required for source installs |
| pip | Use current pip/setuptools/wheel |
| ffmpeg | Required for audio workflows |
| Node/npm | Required only for Electron app under `gui/` |
| CUDA | Optional; use `requirements-gpu-cu124.txt` or `requirements-gpu.txt` |

## Install From Source

Windows PowerShell:

```powershell
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
Copy-Item .env.example .env
```

macOS/Linux:

```bash
git clone https://github.com/Blueibear/AskRex-Assistant.git
cd AskRex-Assistant
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
cp .env.example .env
```

Optional runtime stacks:

```bash
pip install -r requirements-cpu.txt
pip install -r requirements-gpu-cu124.txt
pip install -r requirements-dev.txt
```

Run diagnostics:

```bash
python -m rex doctor
python -m rex --help
```

## Configuration

- Put secrets in `.env`.
- Put runtime settings in `config/rex_config.json`.
- Use `rex-config show` to inspect resolved config.
- Use `rex-config migrate-legacy-env` to migrate old non-secret env vars.

Common service secrets:

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM backend |
| `ANTHROPIC_API_KEY` | Anthropic LLM backend |
| `REX_SPEAK_API_KEY` | TTS API |
| `REX_TOOL_API_KEY` | Rex tool server |
| `REX_AGENT_API_KEY` or configured `REX_AGENT_TOKEN_ENV` | Computer agent |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway |
| `HA_TOKEN` | Home Assistant |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | SMS/telephony |

## Start Surfaces

Text CLI:

```bash
python -m rex
```

Voice loop:

```bash
python rex_loop.py
```

Electron desktop app:

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Python/Flask API and experimental web dashboard:

```bash
rex-gui
```

Open:

```text
http://127.0.0.1:8765/ui/
```

Use this for local API routes and compatibility checks. The browser dashboard is
not the primary GUI.

TTS API:

```bash
REX_SPEAK_API_KEY=replace-with-random-secret rex-speak-api
```

Default:

```text
http://127.0.0.1:5005
```

Tool server:

```bash
REX_TOOL_API_KEY=replace-with-random-secret rex-tool-server
```

Default:

```text
http://127.0.0.1:18790
```

Computer agent:

```bash
REX_AGENT_API_KEY=replace-with-random-secret rex-agent
```

Deprecated legacy compatibility proxy (`flask_proxy.py`; see `SURFACE-CLASSIFICATION.md`).
This is not an active recommended runtime surface; use `rex-gui` for new work.

```bash
python flask_proxy.py  # deprecated legacy compatibility only
```

Default:

```text
http://127.0.0.1:5000
```

## Electron Desktop App

```powershell
cd gui
npm.cmd install
npm.cmd run dev
```

Build/verify:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
npm.cmd run preview
```

For Electron-only verification harnesses, build first, then run a
`gui/tmp_verify_*.cjs` harness that requires `gui/dist-electron/main/index.js`.

## Health Checks

```bash
curl http://127.0.0.1:8765/api/dashboard/status
curl http://127.0.0.1:5005/health/live
curl http://127.0.0.1:5005/health/ready
curl http://127.0.0.1:18790/health/live
curl http://127.0.0.1:18790/health/ready
```

Deprecated legacy proxy health checks:

```bash
curl http://127.0.0.1:5000/health/live
curl http://127.0.0.1:5000/health/ready
```

## systemd Units

The repo includes systemd examples under `deploy/systemd/`:

| Unit | Current entry point |
|---|---|
| `rex-api.service` | `python flask_proxy.py` (deprecated legacy compatibility only; see `SURFACE-CLASSIFICATION.md`) |
| `rex-tts.service` | `rex-speak-api` |
| `rex-voice.service` | `python rex_loop.py` |
| `rex-agent.service` | `rex-agent` |

Treat these as templates. Review paths, environment files, user/group,
ports, auth, and whether you actually want the deprecated legacy compatibility proxy before installing.

Example install shape:

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin rex
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rex-tts rex-voice rex-agent
sudo systemctl start rex-tts rex-voice rex-agent
```

Use `rex-gui` manually or create a dedicated service for it if the Flask API or
experimental browser dashboard should run under systemd.

### Restart policy

Each unit sets `Restart=on-failure` in the `[Service]` section so that systemd
automatically restarts the process when it exits with a non-zero status.
`RestartSec=5` adds a short back-off delay between attempts.

### Burst / backoff limit

To prevent runaway restart loops, units set:

```ini
StartLimitBurst=5
StartLimitIntervalSec=60
```

This allows at most 5 restarts in 60 seconds before systemd stops trying.

### Viewing logs

Use `journalctl` to inspect service output:

```bash
journalctl -u rex-api.service -f
journalctl -u rex-tts.service --since "1 hour ago"
```

## Docker (developer-only)

Docker is retained for developer/operator smoke testing only. It is not a
supported AskRex production deployment path or end-user artifact.

```bash
docker build -t askrex-assistant .
docker run --rm --env-file .env -it askrex-assistant
```

TTS API:

```bash
docker run --rm --env-file .env -p 5005:5005 \
  -it askrex-assistant rex-speak-api
```

## External mobile access (`askrex.app`)

Do **not** expose `rex-gui`, `rex-agent`, `rex-speak-api`, or `rex-tool-server` as the mobile backend. The planned public hostname `https://askrex.app` may front only a loopback-bound `rex.mobile_api` origin through an explicit `/mobile/*` path allowlist.

A Cloudflare Tunnel reference is documented in `docs/mobile/CLOUDFLARE_TUNNEL.md`. It uses placeholder tunnel identifiers/credential-file paths only; generated Cloudflare credentials stay outside this repository. The public deployment gate remains closed until the versioned WebPKI transport-binding, trusted-proxy/rate-limit, and public-topology tests in `docs/mobile/ASKREX_APP_GATEWAY.md` are implemented.

## Security Notes

- Keep services bound to localhost unless you have a reverse proxy and auth.
- Use HTTPS and an external access layer if exposing anything outside localhost.
- Use Redis-backed limiter storage for multi-worker API deployments.
- Do not put secrets in `config/rex_config.json`.
- Do not expose `rex-agent` or `rex-tool-server` without a strict network and
  token boundary.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Unsupported Python | Wrong interpreter | Recreate `.venv` with Python 3.11 |
| `rex-speak-api` exits | Missing `REX_SPEAK_API_KEY` | Set it in `.env` or the shell |
| TTS 401 | Missing/wrong API key | Send `X-API-Key` or bearer token |
| Tool server 401 | Missing/wrong `REX_TOOL_API_KEY` | Set token and send bearer auth |
| Electron GUI unavailable | Electron dev server or built app not running | Start the Electron app from `gui/`; rebuild with `npm.cmd run build` if needed |
| Flask API or experimental browser dashboard unavailable | Wrong port or service not started | Start `rex-gui`; check `REX_GUI_PORT` |
| Electron stale behavior | Built files stale | Run `npm.cmd run build` in `gui/` |
| Legacy proxy startup fails on migrations | Compatibility DB state issue | Prefer `rex-gui`; inspect `rex/migrations.py` before maintaining proxy |
| Migration check blocks startup in an emergency | DB migration table unavailable | Set `skip_migration_check=true` in `config/rex_config.json` to bypass the migration check; apply migrations manually before re-enabling |
