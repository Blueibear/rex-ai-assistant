# Rex AI Assistant — Production Deployment Guide

This guide walks a new operator through deploying Rex AI Assistant from scratch.
Follow each section in order. No prior knowledge of the codebase is assumed.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | 3.10+   | 3.11 recommended |
| Git         | any     | |
| pip         | 23+     | Bundled with Python |
| (Optional) CUDA toolkit | 12.x | For GPU-accelerated TTS/STT |

On **Windows** a virtual environment must be activated using PowerShell, not
Command Prompt (see below).

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd rex-ai-assistant
```

---

## 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install .
pip install -r requirements-dev.txt   # optional: for running tests
```

GPU users only (CUDA 12.4):

```bash
pip install -r requirements-gpu-cu124.txt
```

---

## 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and set the values appropriate for your deployment.
At minimum, set one LLM provider key:

```dotenv
# For OpenAI:
OPENAI_API_KEY=sk-...

# For Anthropic:
ANTHROPIC_API_KEY=sk-ant-...

# For local Ollama (no key needed):
# OLLAMA_BASE_URL=http://localhost:11434
```

See `docs/configuration.md` for a full reference of every environment variable.

**Security:** Never commit `.env` to version control. It is listed in `.gitignore`.

---

## 5. Apply Database Migrations

Rex tracks schema migrations in the `schema_migrations` table. Migrations **must
be applied before starting the service**; the application will refuse to start if
unapplied migrations are detected (exit code 1).

Run the migration tool from the repository root:

```bash
python -m rex.migrations apply
```

To verify the migration state without starting the service:

```bash
python -c "from rex.migrations import get_pending_migrations; print(get_pending_migrations())"
```

An empty list `[]` means the database is up to date.

**Emergency bypass** (not recommended for production):

```bash
SKIP_MIGRATION_CHECK=1 python flask_proxy.py
```

---

## 6. Start the Service

**Flask proxy (main API + dashboard):**

```bash
python flask_proxy.py
```

**TTS API (separate process, optional):**

```bash
python rex_speak_api.py
```

**Voice loop (requires audio hardware):**

```bash
python rex_loop.py
```

The Flask proxy binds to `0.0.0.0:5000` by default. For production, place it
behind a reverse proxy (nginx, Caddy) and restrict direct access.

---

## 7. First-Run Verification

### 7a. Health check

Confirm the service is healthy:

```bash
curl http://localhost:5000/health/live
```

Expected response:

```json
{"status": "ok"}
```

For a full readiness check (verifies all configured dependencies):

```bash
curl http://localhost:5000/health/ready
```

Expected response when all checks pass:

```json
{"status": "ok", "checks": {}}
```

If any check fails, the response body lists the failing check names and reasons.

### 7b. Run the test suite (optional but recommended)

```bash
pytest -q
```

All tests should pass. Pre-existing failures in specific integration tests
(marked as such) do not affect functionality.

### 7c. Doctor script

```bash
python scripts/doctor.py
```

This prints a summary of all configured integrations and their status.

---

## 8. Environment Variable Quick Reference

| Variable               | Default        | Notes |
|------------------------|----------------|-------|
| `REX_PROXY_TOKEN`      | (none)         | Required for bearer-token auth |
| `REX_ACTIVE_USER`      | (none)         | Default user profile key |
| `REX_ALLOWED_ORIGINS`  | localhost URLs | CORS whitelist |
| `DB_QUERY_TIMEOUT`     | `10.0`         | Seconds; `-1` disables |
| `SKIP_MIGRATION_CHECK` | (unset)        | Set to `1` only in emergencies |
| `API_RATE_LIMIT`       | `60 per minute`| Flask-Limiter string |

See `docs/configuration.md` for the complete list.

---

## 9. Stopping the Service

Press `Ctrl+C` in the terminal running the process. The service handles
`SIGTERM` gracefully and logs a shutdown message.

---

## 10. Upgrading

```bash
git pull
pip install .
python -m rex.migrations apply   # apply any new migrations
```

Restart the service after upgrading.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Service exits immediately with code 1 | Unapplied migrations | Run `python -m rex.migrations apply` |
| `/health/ready` returns 503 | Missing config or failing check | Check logs; run `python scripts/doctor.py` |
| `CORS` errors in browser | `REX_ALLOWED_ORIGINS` missing your origin | Add origin to the env var |
| Rate limit 429 on all requests | `API_RATE_LIMIT` too low | Increase limit or whitelist internal IPs |
