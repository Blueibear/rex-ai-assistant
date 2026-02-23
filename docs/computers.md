# Windows Computer Control

**Implementation Status: Client Foundation Only (Cycle 5.1)**

This document covers the client-side foundation for remote Windows computer
control introduced in Phase 5, Cycle 5.1.  The Windows agent server is **not
included yet** — that is Cycle 5.3.

---

## What exists now (Cycle 5.1)

| Component | Status |
|-----------|--------|
| Config models (`computers[]`) | Implemented |
| HTTP client (`AgentClient`) | Implemented |
| High-level service (`ComputerService`) | Implemented |
| CLI commands (`rex pc ...`) | Implemented |
| Offline tests | Implemented |
| Windows agent server | **Not yet** (Cycle 5.3) |

The client code lives in `rex/computers/`:

- `config.py` — Pydantic v2 models for the `computers[]` config section
- `client.py` — `AgentClient` HTTP client (uses `requests` or stdlib `urllib`)
- `service.py` — `ComputerService` facade used by the CLI

---

## Expected agent endpoints (contract)

The client expects the following HTTP endpoints on the agent server.  All
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
  "stderr": ""
}
```

The agent server is responsible for running only allowlisted commands on its
side.  The client also enforces its own allowlist before making any network
call.

---

## How to configure a computer entry

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
        "commands": ["whoami", "dir", "ipconfig", "systeminfo"]
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
rex pc run --id <id> --yes -- <cmd> # run an allowlisted command
rex pc run --id desktop --yes -- whoami
rex pc run --id desktop --yes -- ipconfig
```

---

## Security notes

### Tokens

- Auth tokens are **never** stored in `config/rex_config.json`.
- `auth_token_ref` is a lookup key; the actual token lives in `.env` or
  `config/credentials.json`.
- Tokens are never logged — only the computer `id` and base URL hostname
  appear in log output.

### Allowlist enforcement

- `allowlists.commands` is enforced **client-side** before any HTTP request
  is made.  A command not in the list raises `AllowlistDeniedError`
  immediately, with no network activity.
- `rex pc run` requires an explicit `--yes` flag as a high-risk safety guard
  before any remote execution is attempted.
- The agent server should enforce its own allowlist as well (defence in depth).

### Localhost recommendation

The default `base_url` in the example config uses `http://127.0.0.1:7777`.
For production use over a network, use HTTPS and a reverse proxy with TLS.
Avoid exposing the agent port on public interfaces without authentication.

### Disabled computers

A computer with `"enabled": false` is excluded from `rex pc list` output and
cannot be targeted by `rex pc status` or `rex pc run`.  This allows you to
define computers in config without activating them.

---

## Deferred items (Cycle 5.2 and 5.3)

- **Cycle 5.2**: Policy / approval integration for `rex pc run` — wire into
  the existing `policy_engine` so commands can be marked approval-required.
  (Current mitigation: `rex pc run` requires explicit `--yes`.)
- **Cycle 5.3**: Windows agent server — the lightweight HTTP server that runs
  on the target Windows machine, processes requests, enforces server-side
  allowlists, and streams command output.
- **Future**: TLS / mTLS support, per-user computer access controls, audit
  logging of remote command executions.

---

## Running the tests

All tests are offline (no real network required):

```bash
pytest -q tests/test_computers.py
```
