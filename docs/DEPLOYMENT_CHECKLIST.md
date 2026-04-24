# AskRex Assistant Deployment Checklist

Use this checklist for a current local or single-host deployment. It replaces
older stabilization checklists that instructed copying files from `outputs/`.

## 1. Environment

- [ ] Confirm Python 3.11 is installed.
- [ ] Create a fresh virtual environment with Python 3.11.
- [ ] Activate the virtual environment.
- [ ] Install the package with `pip install -e .`.
- [ ] Install the needed optional stack:
  - [ ] `requirements-cpu.txt`
  - [ ] `requirements-gpu-cu124.txt`
  - [ ] `requirements-gpu.txt`
  - [ ] `requirements-dev.txt`

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

## 2. Configuration

- [ ] Copy `.env.example` to `.env`.
- [ ] Put secrets in `.env` only.
- [ ] Put runtime settings in `config/rex_config.json`.
- [ ] Use `wakeword`, not legacy `wake_word`, for wake-word config.
- [ ] Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or local Ollama config if needed.
- [ ] Set `REX_SPEAK_API_KEY` if running the TTS API.
- [ ] Set `REX_TOOL_API_KEY` if running the tool server.
- [ ] Set integration-specific secrets only for integrations you will use.

Useful commands:

```bash
rex-config show
rex-config migrate-legacy-env --dry-run
```

## 3. Baseline Validation

```bash
python -m rex --help
python -m rex doctor
python -m rex tools --all
```

For dev/test deployments:

```bash
pytest -q
python scripts/security_audit.py
```

## 4. Start Only Needed Surfaces

- [ ] Text chat: `python -m rex`
- [ ] Voice loop: `python rex_loop.py`
- [ ] Electron desktop app: `npm.cmd run dev` from `gui/`
- [ ] Python/Flask API and experimental web dashboard: `rex-gui`
- [ ] TTS API: `rex-speak-api`
- [ ] Tool server: `rex-tool-server`
- [ ] Computer agent: `rex-agent`

Ports:

| Surface | Default |
|---|---:|
| Python/Flask API and experimental web dashboard | `127.0.0.1:8765` |
| TTS API | `127.0.0.1:5005` |
| Tool server | `127.0.0.1:18790` |
| Legacy Flask proxy | `127.0.0.1:5000` |

## 5. Smoke Tests

Flask API:

```bash
curl http://127.0.0.1:8765/api/dashboard/status
```

TTS:

```bash
curl http://127.0.0.1:5005/health/live
```

Tool server:

```bash
curl http://127.0.0.1:18790/health/live
```

Electron:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

## 6. Security Review

- [ ] `.env` is not tracked.
- [ ] Network services are localhost-only unless deliberately proxied.
- [ ] TTS API has `REX_SPEAK_API_KEY`.
- [ ] Tool server has `REX_TOOL_API_KEY`.
- [ ] Computer agent has a token and allowlist.
- [ ] Secrets are not present in logs or docs.
- [ ] Public exposure, if any, is behind HTTPS and an external auth layer.

## 7. Operational Notes

- [ ] Record which surfaces are expected to run continuously.
- [ ] Record any non-default ports.
- [ ] Record optional integrations enabled for this host.
- [ ] For Electron-only verification, build `gui/` before requiring
  `gui/dist-electron/main/index.js` from a harness.

## 8. Rollback

Use normal git history and deployment backups. Do not use destructive commands
such as `git reset --hard` in shared working trees unless the operator has
explicitly chosen that rollback path.
