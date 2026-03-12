# Rex AI Assistant — Operations Runbook

This runbook covers day-to-day operational tasks: starting, stopping, restarting,
diagnosing, and recovering Rex AI Assistant in production.

For initial deployment from scratch see `docs/deployment.md`.
For environment variable reference see `docs/configuration.md`.

---

## Process List

Rex consists of up to three independent processes. Only the Flask proxy is
required; the others are optional depending on your deployment.

| Process | Entry Point | Port | Required? |
|---------|-------------|------|-----------|
| Flask proxy (API + dashboard) | `python flask_proxy.py` | 5000 | Yes |
| TTS API | `python rex_speak_api.py` | 5001 | No (needed for TTS) |
| Voice loop | `python rex_loop.py` | — (no port) | No (needed for voice) |

### Verify each component is running

**Flask proxy:**

```bash
curl -s http://localhost:5000/health/live
# Expected: {"status": "ok"}
```

**TTS API:**

```bash
curl -s http://localhost:5001/health/live
# Expected: {"status": "ok"}  (or a 200 with TTS-specific payload)
```

**Voice loop:**

The voice loop does not expose an HTTP port. Verify it is running by checking
the process list:

```bash
# Linux / macOS:
ps aux | grep rex_loop.py

# Windows (PowerShell):
Get-Process python | Where-Object { $_.CommandLine -like "*rex_loop*" }
```

Or check the log file (see [Log Access](#log-access-and-filtering) below).

---

## Start Procedure

1. Activate the virtual environment.

   **Windows (PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   source .venv/bin/activate
   ```

2. Verify the database is up to date (skip if `SKIP_MIGRATION_CHECK=1`):

   ```bash
   python -m rex.migrations apply
   ```

3. Start the Flask proxy:

   ```bash
   python flask_proxy.py
   ```

4. (Optional) Start the TTS API in a separate terminal:

   ```bash
   python rex_speak_api.py
   ```

5. (Optional) Start the voice loop in a separate terminal:

   ```bash
   python rex_loop.py
   ```

6. Confirm health:

   ```bash
   curl http://localhost:5000/health/live
   curl http://localhost:5000/health/ready
   ```

---

## Stop Procedure

Press **Ctrl+C** in each terminal running a Rex process.

For processes running in the background or under a process manager:

```bash
# Find the PID:
pgrep -f flask_proxy.py     # Linux / macOS
pgrep -f rex_speak_api.py

# Send graceful shutdown signal:
kill -TERM <PID>
```

Rex handles `SIGTERM` and logs a shutdown message before exiting.

---

## Restart Procedure

1. Stop all Rex processes (see [Stop Procedure](#stop-procedure)).
2. (If upgrading) Run `git pull && pip install . && python -m rex.migrations apply`.
3. Follow the [Start Procedure](#start-procedure).

For a quick in-place restart without upgrade:

```bash
kill -TERM $(pgrep -f flask_proxy.py) && sleep 2 && python flask_proxy.py &
```

---

## Health Check Verification

### Liveness

Returns 200 if the process is running and the event loop is responsive.
Does not check external dependencies.

```bash
curl -s http://localhost:5000/health/live
```

Expected:

```json
{"status": "ok"}
```

### Readiness

Returns 200 when all configured dependencies pass their checks. Returns 503
with a JSON body listing failing checks if something is wrong.

```bash
curl -s http://localhost:5000/health/ready
```

Expected (all healthy):

```json
{"status": "ok", "checks": {}}
```

Expected (a check failing):

```json
{"status": "degraded", "checks": {"config": "REX_PROXY_TOKEN not set"}}
```

A 503 response from `/health/ready` does **not** mean the service is down — it
may still be serving requests. Use `/health/live` to determine whether the
process is alive.

---

## Log Access and Filtering

Rex writes structured log output to **stdout / stderr**. In production, redirect
to a log file or a process supervisor's log sink.

### Redirect to a log file (manual)

```bash
python flask_proxy.py >> logs/flask_proxy.log 2>&1
```

### Filter logs

```bash
# Show only ERROR and above:
grep -E "ERROR|CRITICAL" logs/flask_proxy.log

# Show all logs for a specific request ID:
grep "req-abc123" logs/flask_proxy.log

# Follow in real time:
tail -f logs/flask_proxy.log | grep --line-buffered ERROR
```

### Log levels

Set `LOG_LEVEL` in `.env` to control verbosity:

| Level | Usage |
|-------|-------|
| `DEBUG` | Verbose — trace every request; development only |
| `INFO` | Default — startup, shutdown, errors |
| `WARNING` | Degraded conditions only |
| `ERROR` | Failures only |

---

## What to Do if a Service Fails to Start

1. **Check the exit code.** A non-zero exit code indicates a startup failure.
   Common values:
   - Exit code 1: unapplied database migrations or config validation error.

2. **Read the last 50 lines of log output.** The startup sequence logs each
   step at INFO level; the log line preceding the failure identifies the failing
   step.

3. **Run the doctor script:**

   ```bash
   python scripts/doctor.py
   ```

   This checks all configured integrations and prints a status summary without
   starting the service.

4. **Verify the `.env` file** is present, readable, and contains the required
   variables (at minimum one LLM provider key). Compare against `.env.example`.

5. **Verify Python version:**

   ```bash
   python --version
   # Must be 3.10 or higher
   ```

---

## Error Scenarios

### Scenario 1: Service exits immediately (exit code 1, migration error)

**Symptom:** `python flask_proxy.py` prints an error about pending migrations
and exits with code 1.

**Diagnosis:**

```bash
python -c "from rex.migrations import get_pending_migrations; print(get_pending_migrations())"
```

If the output is a non-empty list, migrations are pending.

**Resolution:**

```bash
python -m rex.migrations apply
```

Then restart the service. If the migration script itself fails, check the
`database_url` in `config/rex_config.json` and verify the database file is
accessible.

**Emergency bypass (not recommended):**

```bash
SKIP_MIGRATION_CHECK=1 python flask_proxy.py
```

---

### Scenario 2: /health/ready returns 503 — config check failing

**Symptom:** `curl http://localhost:5000/health/ready` returns HTTP 503 with
a JSON body naming the failing check.

**Diagnosis:**

Read the response body:

```bash
curl -sv http://localhost:5000/health/ready
```

The `checks` field identifies the failing component. Common values:

- `"config"` — a required environment variable is not set.
- `"database"` — the database file is missing or locked.

**Resolution:**

- For config failures: open `.env` and set the missing variable. Restart.
- For database failures: confirm the database file path in
  `config/rex_config.json`, run `python -m rex.migrations apply`, restart.

---

### Scenario 3: 429 Too Many Requests on all endpoints

**Symptom:** All API responses return HTTP 429 with a `Retry-After` header,
even from a single client.

**Diagnosis:**

```bash
grep API_RATE_LIMIT .env
```

If blank or missing, the default `60 per minute` applies. A single automated
client can hit this limit quickly.

**Resolution:**

Increase the limit in `.env`:

```dotenv
API_RATE_LIMIT=600 per minute
```

Restart the service. Health endpoints (`/health/live`, `/health/ready`) are
exempt from rate limiting and are never affected.

---

### Scenario 4: CORS errors in browser ("blocked by CORS policy")

**Symptom:** Browser console shows a CORS error when the dashboard or a
custom integration makes an API call.

**Diagnosis:**

```bash
grep REX_ALLOWED_ORIGINS .env
```

If blank, only default localhost origins are whitelisted.

**Resolution:**

Add your origin to `.env`:

```dotenv
REX_ALLOWED_ORIGINS=https://my-dashboard.example.com
```

Multiple origins are comma-separated. Restart the service.

---

### Scenario 5: DB query timeout — requests hang and return 503

**Symptom:** Some API requests take exactly `DB_QUERY_TIMEOUT` seconds then
return a 503 or 500 error with `"database timeout"` in the message.

**Diagnosis:**

```bash
grep DB_QUERY_TIMEOUT .env
# Default is 10.0 seconds
```

Check logs for `QueryTimeoutError` entries. This usually indicates a slow
query, a locked database, or a write-heavy workload on an under-provisioned
server.

**Resolution:**

- Increase `DB_QUERY_TIMEOUT` for longer-running queries.
- Check for active write locks: `lsof <database-path>` (Linux) or
  Process Explorer (Windows).
- Increase `DB_POOL_MAX_SIZE` if the pool is exhausted.
- Restart the service after making `.env` changes.

---

### Scenario 6: Voice loop fails to start — audio device not found

**Symptom:** `python rex_loop.py` exits immediately or throws a
`sounddevice.PortAudioError` or similar error about no audio input device.

**Diagnosis:**

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

If the output is empty or shows no input devices, the audio stack is not
configured.

**Resolution:**

- On Linux: install `portaudio19-dev` and `python3-pyaudio`.
  ```bash
  sudo apt-get install portaudio19-dev python3-pyaudio
  ```
- Verify microphone permissions (macOS: System Settings → Privacy → Microphone).
- The Flask proxy and TTS API are unaffected and continue to operate
  without audio hardware.

---

### Scenario 7: LLM provider returns errors — chat responses fail

**Symptom:** Chat requests return errors or empty replies. Logs show HTTP 401,
429, or 503 errors from the LLM provider.

**Diagnosis:**

```bash
grep -E "OPENAI_API_KEY|ANTHROPIC_API_KEY|OLLAMA_BASE_URL" .env
```

Check that the key is present and not expired.

**Resolution:**

- HTTP 401: Regenerate the API key and update `.env`.
- HTTP 429: You have hit the provider's rate limit. Add retry logic or
  reduce request frequency.
- HTTP 503 / connection refused (Ollama): Verify Ollama is running:
  ```bash
  curl http://localhost:11434/api/tags
  ```
  Start Ollama if needed: `ollama serve`.

---

## Quick-Reference Checklist

Use this checklist after any incident or restart.

- [ ] `curl http://localhost:5000/health/live` returns `{"status": "ok"}`
- [ ] `curl http://localhost:5000/health/ready` returns 200 or expected 503
- [ ] No `ERROR` or `CRITICAL` entries in logs since startup
- [ ] Flask proxy process is listed in `ps aux | grep flask_proxy`
- [ ] (If TTS needed) TTS API process is running and `/health/live` returns 200
- [ ] (If voice needed) Voice loop process is listed in `ps aux | grep rex_loop`
- [ ] `python scripts/doctor.py` shows no red failures
