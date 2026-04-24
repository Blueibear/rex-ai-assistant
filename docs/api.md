# AskRex Assistant HTTP API Reference

AskRex exposes several local HTTP surfaces. The current primary GUI is the Electron app under `gui/`. `rex-gui` starts a Flask process that serves local API routes and an incomplete, experimental `/ui/` browser dashboard. `flask_proxy.py` remains as a legacy proxy surface for compatibility.

## Service Summary

| Service | Command | Default URL | Purpose |
|---|---|---|---|
| Python/Flask API and experimental web dashboard | `rex-gui` | `http://127.0.0.1:8765` | Serves local dashboard/API routes; `/ui/` exists but is not the primary GUI |
| TTS API | `rex-speak-api` | `http://127.0.0.1:5005` | Converts text to WAV audio |
| OpenClaw tool server | `rex-tool-server` | `http://127.0.0.1:18790` | Exposes Rex tools over HTTP |
| Legacy Flask proxy | `python flask_proxy.py` | `http://0.0.0.0:5000` | Legacy proxy/search/contracts API |

## Common Health Endpoints

Services that register the shared health blueprint expose:

```text
GET /health/live
GET /health/ready
```

Typical responses:

```json
{"status":"ok"}
```

```json
{"status":"ready"}
```

The tool server readiness response includes tool count:

```json
{"status":"ok","tool_count":12}
```

## Python/Flask API and Experimental Web Dashboard (`rex-gui`)

Start:

```bash
rex-gui
```

Override port:

```bash
REX_GUI_PORT=9000 rex-gui
```

### UI

The browser UI at `/ui/` is incomplete in current testing. Use the Electron app
for normal GUI interaction and use these routes for API compatibility and smoke
checks.

| Method | Path | Description |
|---|---|---|
| `GET` | `/ui/` | Serve the built React UI |
| `GET` | `/ui/<filename>` | Serve static UI asset |
| `GET` | `/dashboard` | Redirect to `/ui/` |
| `GET` | `/api/dashboard/status` | Basic status payload |

### Chat

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/chat/history` | Bearer token | Load authenticated user's chat turns |
| `POST` | `/api/chat/clear` | Bearer token | Clear authenticated user's chat turns |
| `POST` | `/api/chat/send` | Bearer token | Stream an assistant reply with SSE framing |

`POST /api/chat/send` request:

```json
{"message":"Hello Rex"}
```

The stream emits `data: ...` SSE chunks.

### Auth, Setup, User

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/setup/status` | Whether first-run setup is needed |
| `POST` | `/api/setup/complete` | Create first user and persist setup settings |
| `POST` | `/api/auth/register` | Register a user |
| `POST` | `/api/auth/login` | Login and receive a bearer token |
| `POST` | `/api/auth/logout` | Logout acknowledgement |
| `GET` | `/api/user/permissions` | Current user's permissions |
| `POST` | `/api/admin/permissions/grant` | Grant a permission; admin required |
| `POST` | `/api/admin/permissions/revoke` | Revoke a permission; admin required |
| `GET` | `/api/user/preferences` | Current user's preferences |
| `PATCH` | `/api/user/preferences` | Merge preference updates |
| `GET` | `/api/user/avatar` | Current user's avatar |
| `POST` | `/api/user/avatar` | Upload avatar |
| `GET` | `/api/personalities` | List available personalities |

### Home Assistant and Devices

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/devices` | List approved devices from config aliases |
| `POST` | `/api/devices/<entity_id>/command` | Send a Home Assistant command |
| `POST` | `/api/ha/test` | Test HA connection |
| `POST` | `/api/ha/save` | Save HA base URL/token config |
| `GET` | `/api/ha/states` | Fetch HA entity states |

### Dashboard Utility APIs

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/logs/stream` | SSE stream of the active runtime log, normally `data/logs/rex.log` |
| `GET` | `/api/logs/download` | Download the current log file |
| `GET` | `/api/usage` | LLM usage summary |
| `GET` | `/api/status/current` | Current Rex status |
| `GET` | `/api/status/stream` | SSE status stream |
| `GET` | `/api/history` | Recent command history |
| `GET` | `/api/integrations` | Configured integration summary |
| `GET` | `/api/calendar/events` | Calendar events from configured provider |
| `GET` | `/api/email/inbox` | Inbox messages from configured provider |
| `GET` | `/api/sms/threads` | SMS threads from configured provider |
| `GET` | `/api/capabilities` | Capability registry |
| `GET` | `/api/tools` | Registered tools; bearer token required |
| `GET` | `/api/quick-actions` | List quick actions |
| `POST` | `/api/quick-actions` | Add quick action |
| `DELETE` | `/api/quick-actions/<action_id>` | Delete quick action |
| `POST` | `/api/quick-actions/<action_id>/run` | Run quick action |

## TTS API (`rex-speak-api`)

Start:

```bash
REX_SPEAK_API_KEY=change-me rex-speak-api
```

Default URL: `http://127.0.0.1:5005`

### `POST /speak`

Authentication: `X-API-Key: <REX_SPEAK_API_KEY>` or `Authorization: Bearer <REX_SPEAK_API_KEY>`.

Request:

```json
{
  "text": "Hello from AskRex",
  "user": "default",
  "language": "en"
}
```

Response: binary WAV audio (`audio/wav`).

Limits:

- `REX_SPEAK_MAX_CHARS`, default 800
- `REX_SPEAK_MAX_REQUEST_BYTES`, default 65536
- `REX_SPEAK_RATE_LIMIT`, default 30
- `REX_SPEAK_RATE_WINDOW`, default 60 seconds
- `REX_SPEAK_PORT`, default 5005

The TTS API also registers Home Assistant and shopping-list blueprints when their imports/config are available.

## OpenClaw Tool Server (`rex-tool-server`)

Start:

```bash
REX_TOOL_API_KEY=change-me rex-tool-server
```

Default URL: `http://127.0.0.1:18790`

### `POST /rex/tools/{tool_name}`

Authentication: `X-API-Key: <REX_TOOL_API_KEY>` or `Authorization: Bearer <REX_TOOL_API_KEY>`.

Request:

```json
{
  "args": {"location": "Dallas, TX"},
  "context": {"session_key": "main"}
}
```

Success:

```json
{
  "status": "success",
  "result": {}
}
```

Tool server environment:

- `REX_TOOL_SERVER_PORT`, default 18790
- `REX_TOOL_API_KEY`, required for tool calls
- `REX_TOOL_RATE_LIMIT`, default 60
- `REX_TOOL_RATE_WINDOW`, default 60 seconds

Tool calls are rate-limited and guarded by the policy adapter. Denied or approval-required actions return 403.

## Legacy Flask Proxy (`flask_proxy.py`)

`flask_proxy.py` is not the normal GUI runtime. It remains for compatibility with older proxy workflows.

Routes defined directly in the file:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Redirect to `/dashboard` |
| `GET` | `/whoami` | Return authenticated user and memory profile summary |
| `GET` | `/search?q=...` | Search through `plugins.web_search` |
| `GET` | `/contracts` | Contract schema metadata |
| `GET` | `/health/live` | Liveness |
| `GET` | `/health/ready` | Readiness |

Auth is based on Cloudflare Access email, `Authorization: Bearer <REX_PROXY_TOKEN>`, `X-Rex-Proxy-Token`, or loopback when `REX_PROXY_ALLOW_LOCAL=1`.

## Error Envelope

Shared error helpers return a structured envelope:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description."
  }
}
```

Common status codes:

| Status | Meaning |
|---|---|
| 400 | Invalid input |
| 401 | Missing or invalid credentials |
| 403 | Policy/auth denied |
| 404 | Unknown endpoint or tool |
| 413 | Request body too large |
| 429 | Rate limit exceeded |
| 500 | Internal error |
| 503 | Dependency/config unavailable |
