# Windows Computer Control

**Implementation Status: Client Foundation + Agent Server + Policy/Approvals (Cycle 5.1 + 5.2b + 5.3)**

This document covers the full Windows computer control stack:
- **Client** (`rex/computers/`) — queries the agent from any Rex host machine
- **Agent server** (`rex/computers/agent_server.py`) — lightweight HTTP server
  that runs on the Windows target machine and executes allowlisted commands
- **Policy + approvals** (`rex/computers/pc_run_policy.py`) — gates `rex pc run`
  through the policy engine before any network call is made

---

## What exists now

| Component | Status |
|-----------|--------|
| Config models (`computers[]`) | Implemented |
| HTTP client (`AgentClient`) | Implemented |
| High-level service (`ComputerService`) | Implemented |
| CLI commands (`rex pc ...`) | Implemented |
| Windows agent server | **Implemented (Cycle 5.3)** |
| Policy + approvals for `rex pc run` | **Implemented (Cycle 5.2b)** |
| Offline tests | Implemented |

The client code lives in `rex/computers/`:

- `config.py` — Pydantic v2 models for the `computers[]` config section
- `client.py` — `AgentClient` HTTP client (uses `requests` or stdlib `urllib`)
- `service.py` — `ComputerService` facade used by the CLI
- `pc_run_policy.py` — policy + approval gating for `rex pc run` (Cycle 5.2b)
- `agent_server.py` — Flask HTTP agent server (Cycle 5.3)

The agent entry-point script is `scripts/windows_agent.py`.

---

## Agent API contract

The client expects the following HTTP endpoints on the agent server. All
requests include the `X-Auth-Token` header.

### `GET /health`

```
200 OK
{"status": "ok"}
```

### `GET /status`

```
200 OK
{
  "hostname": "DESKTOP-ABC",
  "os": "Windows 11",
  "user": "alice",
  "time": "2026-02-23T14:30:00"
}
```

### `POST /run`

Request body:

```json
{
  "command": "whoami",
  "args": [],
  "cwd": null
}
```

Response:

```json
{
  "exit_code": 0,
  "stdout": "DESKTOP-ABC\\alice\n",
  "stderr": "",
  "duration_ms": 42
}
```

The `duration_ms` field is additional agent metadata; the Rex client ignores it
but it is useful for debugging.

Error responses (auth, allowlist, rate limit) use standard HTTP status codes:

| Status | Meaning |
|--------|---------|
| 400 | Malformed request body |
| 401 | Missing or invalid `X-Auth-Token` |
| 403 | Command not on the agent-side allowlist |
| 429 | Rate limit exceeded |

Both the client and the agent enforce their own allowlists independently
(defence in depth).

---

## Running the agent on Windows

### Quick start

```powershell
# 1. Install Rex (on the target machine)
pip install rex-ai-assistant

# 2. Set the auth token
$env:REX_AGENT_TOKEN = "your-secret-token"

# 3. Set the allowlist (comma-separated command names)
$env:REX_AGENT_ALLOWLIST = "whoami,ipconfig,systeminfo"

# 4. Start the agent
rex-agent
```

Or without installing as a script:

```powershell
python scripts/windows_agent.py
```

### Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `REX_AGENT_TOKEN` | — | **Required.** Auth token for incoming requests. |
| `REX_AGENT_TOKEN_ENV` | `REX_AGENT_TOKEN` | Name of the env var that holds the token (useful when the token is in a differently-named var). |
| `REX_AGENT_HOST` | `127.0.0.1` | Bind address. Only change if you need network access (see security notes). |
| `REX_AGENT_PORT` | `7777` | Listen port. |
| `REX_AGENT_ALLOWLIST` | `whoami` | Comma-separated list of commands allowed for remote execution. |
| `REX_AGENT_RATE_LIMIT` | `60` | Max requests per client IP per minute. Set `0` to disable. |
| `REX_AGENT_TIMEOUT` | `30` | Subprocess execution timeout in seconds. |
| `REX_AGENT_MAX_OUTPUT` | `65536` | Maximum bytes of stdout/stderr returned per `/run` call. |
| `REX_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### curl examples

```bash
# Health check
curl -H "X-Auth-Token: your-secret-token" http://127.0.0.1:7777/health

# Host status
curl -H "X-Auth-Token: your-secret-token" http://127.0.0.1:7777/status

# Run a command
curl -X POST \
     -H "X-Auth-Token: your-secret-token" \
     -H "Content-Type: application/json" \
     -d '{"command":"whoami","args":[]}' \
     http://127.0.0.1:7777/run
```

---

## How to configure a computer entry (Rex client side)

Add a `computers` array to `config/rex_config.json`:

```json
{
  "computers": [
    {
      "id": "desktop",
      "label": "Main Desktop",
      "base_url": "http://127.0.0.1:7777",
      "auth_token_ref": "PC_DESKTOP_TOKEN",
      "enabled": true,
      "allowlists": {
        "commands": ["whoami", "ipconfig", "systeminfo"]
      },
      "connect_timeout": 5.0,
      "read_timeout": 30.0
    }
  ]
}
```

Then add the auth token to `.env` (never to config):

```
PC_DESKTOP_TOKEN=your-secret-token-here
```

The `auth_token_ref` value (`PC_DESKTOP_TOKEN`) is looked up via
`CredentialManager`.  The manager tries the environment variable directly
(i.e., `os.getenv("PC_DESKTOP_TOKEN")`).

---

## Config field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | required | Unique identifier used in CLI commands |
| `label` | string | `""` | Human-friendly display name |
| `base_url` | string | required | http(s) URL of the agent API |
| `auth_token_ref` | string | required | CredentialManager key for the bearer token |
| `enabled` | bool | `true` | Disabled computers are hidden from `rex pc list` |
| `allowlists.commands` | list[str] | `[]` | Command names permitted for remote execution |
| `connect_timeout` | float | `5.0` | TCP connect timeout (seconds) |
| `read_timeout` | float | `30.0` | HTTP read timeout (seconds) |

---

## CLI commands

```
rex pc list                         # show enabled computers
rex pc list --all                   # include disabled computers
rex pc status --id <id>             # query agent for host info
rex pc run --id <id> --yes -- <cmd> # run an allowlisted command (requires approval)
```

### Two-step execution flow for `rex pc run`

Remote execution requires **both** an explicit approval record **and** `--yes`.

**Step 1 — Request execution (creates a pending approval)**

```
rex pc run --id desktop --yes -- whoami
```

Output:

```
Approval required before remote execution can proceed.

  Approval ID : apr_abc123def456
  Computer    : desktop
  Command     : whoami

  To approve : rex approvals --approve apr_abc123def456
  To deny    : rex approvals --deny apr_abc123def456

After approving, re-run this command to execute.
```

**Step 2 — Review and approve**

```
rex approvals                                   # list pending approvals
rex approvals --show apr_abc123def456           # inspect the approval record
rex approvals --approve apr_abc123def456        # approve execution
```

**Step 3 — Re-run to execute**

```
rex pc run --id desktop --yes -- whoami
```

Now the command executes. `--yes` is still required even after approval.

**To deny** (blocks the pending approval without executing):

```
rex approvals --deny apr_abc123def456 --reason "not authorised"
```

---

## Security model

### Tokens

- Auth tokens are **never** stored in `config/rex_config.json`.
- `auth_token_ref` is a lookup key; the actual token lives in `.env` or
  `config/credentials.json`.
- The agent reads its token from an environment variable (never from disk
  config that might be committed to source control).
- Tokens are **never** logged — only the computer `id` and remote address
  appear in log output.

### Policy + approvals gate (Cycle 5.2b)

`rex pc run` is classified as `HIGH`-risk in the policy engine
(`tool_name="pc_run"`).  Before any network call is made:

1. The client-side allowlist is checked (no network).  Non-allowlisted commands
   are refused immediately.
2. The policy engine is consulted.  The default policy requires an explicit
   approval record (`requires_approval=True`).
3. If no approved approval exists, a pending approval record is written to
   `data/approvals/` and the user is told how to approve it.
4. After the user approves via `rex approvals --approve <id>`, re-running the
   command finds the approved record and proceeds.
5. The `--yes` flag is still required even after approval — it acts as a second
   layer of explicit confirmation and **cannot** bypass the approval requirement.

The approval payload includes computer ID, command, args, allowlist decision,
and user identity.  **Auth tokens are never stored in the approval record.**

### Allowlist enforcement (defence in depth)

Allowlists are enforced at three layers:

1. **Policy gate** (`rex/computers/pc_run_policy.py`): before any approval is
   created, non-allowlisted commands are rejected with a clear message.
2. **Client-side** (`rex/computers/service.py`): before any HTTP request is
   made. A command not in the client list raises `AllowlistDeniedError`
   immediately with no network activity.
3. **Agent-side** (`rex/computers/agent_server.py`): at the HTTP endpoint
   before any subprocess is spawned. A command not in the agent list returns
   HTTP 403 and the subprocess is **never called**.

### Subprocess safety

- `subprocess.run` is called with `shell=False` — no shell injection possible.
- The command payload must be a JSON array (`command` + `args`), not a string
  passed to a shell.
- Execution timeout is enforced; timed-out processes return exit code `-1` with
  a descriptive message in `stderr`.

### Localhost recommendation

The agent binds to `127.0.0.1` by default.  **Do not change `REX_AGENT_HOST`
to `0.0.0.0` without placing the agent behind a TLS-terminating reverse proxy.**
Token auth over plain HTTP on a non-loopback interface exposes the token to
network eavesdroppers.

### Reverse proxy guidance (optional)

If you need to access the agent from a remote Rex host, the recommended setup is:

1. **TLS termination** via nginx, Caddy, or a cloud load balancer.
2. Keep the agent bound to `127.0.0.1` and proxy to it locally.
3. Use `REX_AGENT_HOST=127.0.0.1` always; never expose the agent port on a
   public interface.

Example nginx snippet:

```nginx
server {
    listen 7778 ssl;
    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    location / {
        proxy_pass http://127.0.0.1:7777;
        proxy_set_header X-Auth-Token $http_x_auth_token;
    }
}
```

### Rate limiting

The agent enforces a fixed-window per-IP rate limit (default: 60 req/min).
Adjust with `REX_AGENT_RATE_LIMIT`. Set to `0` to disable (not recommended for
network-exposed deployments).

### Disabled computers

A computer with `"enabled": false` is excluded from `rex pc list` output and
cannot be targeted by `rex pc status` or `rex pc run`.

---

## Running the tests

All tests are offline (no real network or subprocess required):

```bash
# Run computer and policy tests
python -m pytest -q tests/test_computers.py tests/test_windows_agent.py tests/test_pc_run_policy.py

# Or the full suite
python -m pytest -q
```

---

## Deferred items (future cycles)

- **Future**: TLS / mTLS support, per-user computer access controls, persistent
  audit log to file, Windows Service wrapper for the agent, configurable
  per-computer policy overrides in `config/rex_config.json`.
