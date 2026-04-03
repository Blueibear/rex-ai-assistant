# AskRex Assistant — API Reference

This document covers all public HTTP endpoints exposed by the AskRex Assistant
services. Two processes expose HTTP APIs:

- **Flask proxy** (`python flask_proxy.py`) — default port 5000
- **TTS API** (`python rex_speak_api.py`) — default port 5001

---

## Authentication

### Flask proxy

Most endpoints require a **Bearer token** in the `Authorization` header:

```
Authorization: Bearer <REX_PROXY_TOKEN>
```

Where `REX_PROXY_TOKEN` is set in `.env`. If not set, bearer-token auth is
disabled and requests from `localhost` are accepted without authentication.

### Dashboard API (`/api/*`)

Dashboard endpoints use **session-based auth**. Obtain a session token via
`POST /api/dashboard/login`, then pass it either:

- As a cookie: `rex_dashboard_token=<token>` (set automatically on login), or
- As a `Bearer` token in `Authorization: Bearer <token>`.

Health endpoints (`/health/live`, `/health/ready`) and
`GET /api/dashboard/status` are **public** — no authentication required.

### TTS API

Requires `Authorization: Bearer <REX_SPEAK_API_TOKEN>` where
`REX_SPEAK_API_TOKEN` is set in `.env`.

---

## Error Response Format

All errors use the standard envelope:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description.",
    "request_id": "req-abc123"  // present on 500 errors only
  }
}
```

Common HTTP status codes:

| Status | Meaning |
|--------|---------|
| 400 | Bad request — missing or invalid input |
| 401 | Unauthorized — missing or invalid credentials |
| 403 | Forbidden — authenticated but not authorized |
| 404 | Not found |
| 429 | Too many requests — rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable — dependency not configured |

---

## Flask Proxy Endpoints

Base URL: `http://localhost:5000`

---

### GET /

Check that Rex is online.

**Authentication:** Bearer token (Flask proxy auth)

**Response 200:**

```
Rex is online. Ask away.
```

---

### GET /whoami

Return the authenticated user's key and a summary of their memory profile.

**Authentication:** Bearer token (required)

**Response 200:**

```json
{
  "user": "alice",
  "profile": {
    "name": "Alice",
    "preferences": {}
  }
}
```

**Error codes:** 401 (missing token), 500 (memory file error)

---

### GET /search

Search the web using the configured search provider.

**Authentication:** Bearer token (required)

**Query parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `q` | Yes | Search query string |

**Response 200:**

```json
{
  "query": "weather today",
  "result": "Current weather: 72°F and sunny."
}
```

**Error codes:** 400 (missing `q`), 503 (search plugin not installed), 502 (provider error)

---

### GET /contracts

Return contract schema metadata for API discoverability.

**Authentication:** Bearer token (required)

**Response 200:**

```json
{
  "contract_version": "1.0",
  "schema_docs_path": "docs/contracts/",
  "models": ["ChatRequest", "LoginRequest"]
}
```

**Error codes:** 503 (contracts module not available)

---

### GET /health/live

Liveness probe. Returns 200 if the process is running.

**Authentication:** None (public)

**Response 200:**

```json
{"status": "ok"}
```

---

### GET /health/ready

Readiness probe. Returns 200 when all configured dependencies pass their checks.

**Authentication:** None (public)

**Response 200 (all healthy):**

```json
{"status": "ok", "checks": {}}
```

**Response 503 (a check failing):**

```json
{
  "status": "degraded",
  "checks": {
    "config": "REX_PROXY_TOKEN not set"
  }
}
```

---

## Dashboard API Endpoints

Base URL: `http://localhost:5000`

All `/api/*` endpoints (except `/api/dashboard/status` and `/api/dashboard/login`)
require a valid dashboard session token.

---

### GET /api/dashboard/status

Return server version and uptime. Public endpoint.

**Authentication:** None

**Response 200:**

```json
{
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "auth_enabled": true,
  "server_time": "2026-03-12T10:00:00",
  "status": "ok"
}
```

---

### POST /api/dashboard/login

Authenticate and obtain a session token.

**Authentication:** None

**Request body** (`application/json`):

```json
{"password": "mysecretpassword"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `password` | string | Yes | Dashboard password (set via `REX_DASHBOARD_PASSWORD` env var) |

**Response 200:**

```json
{
  "token": "tok_abc123xyz",
  "expires_at": "2026-03-13T10:00:00"
}
```

Also sets an `HttpOnly` cookie `rex_dashboard_token`.

**Error codes:** 401 (wrong password), 403 (no password configured and remote access), 429 (too many failed attempts)

---

### POST /api/dashboard/logout

Invalidate the current session.

**Authentication:** Dashboard session token

**Response 200:**

```json
{"message": "Logged out"}
```

Also clears the `rex_dashboard_token` cookie.

---

### GET /api/settings

Return current configuration with sensitive values redacted.

**Authentication:** Dashboard session token (required)

**Response 200:**

```json
{
  "settings": {
    "version": "1.0.0",
    "tts": {"voice": "default"}
  },
  "defaults": { "...": "..." },
  "metadata": {
    "tts.voice": {"restart_required": true}
  }
}
```

**Error codes:** 401, 500

---

### PATCH /api/settings

Update one or more configuration keys.

**Authentication:** Dashboard session token (required)

**Request body** (`application/json`):

```json
{"tts.voice": "female_1", "search.provider": "brave"}
```

Keys use dot-notation for nested paths. Only known keys are accepted.

**Response 200:**

```json
{
  "updated": ["tts.voice", "search.provider"],
  "restart_required": false
}
```

**Error codes:** 400 (invalid key or value type), 401, 500

---

### POST /api/chat

Send a chat message and receive an LLM reply.

**Authentication:** Dashboard session token (required)

**Request body** (`application/json`):

```json
{"message": "What is the weather like today?"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | Non-empty message text |

**Response 200:**

```json
{
  "reply": "I don't have live weather data, but I can help you find out!",
  "timestamp": "2026-03-12T10:05:00",
  "elapsed_ms": 312
}
```

**Error codes:** 400 (blank message), 401, 500 (LLM provider error)

---

### GET /api/chat/history

Return recent chat history.

**Authentication:** Dashboard session token (required)

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | 50 | Max entries to return (capped at 100) |
| `offset` | 0 | Entries to skip |

**Response 200:**

```json
{
  "history": [
    {
      "user_message": "Hello",
      "assistant_reply": "Hi there!",
      "timestamp": "2026-03-12T10:00:00",
      "elapsed_ms": 200
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

**Error codes:** 401

---

### GET /api/scheduler/jobs

List all scheduled jobs.

**Authentication:** Dashboard session token (required)

**Response 200:**

```json
{
  "jobs": [
    {
      "job_id": "job_abc123",
      "name": "Daily Briefing",
      "schedule": "at:08:00",
      "enabled": true,
      "next_run": "2026-03-13T08:00:00",
      "last_run_at": "2026-03-12T08:00:00",
      "run_count": 5,
      "max_runs": null,
      "callback_name": "daily_briefing",
      "workflow_id": null,
      "metadata": {}
    }
  ],
  "total": 1,
  "metrics": {}
}
```

**Error codes:** 401, 500

---

### POST /api/scheduler/jobs

Create a new scheduled job.

**Authentication:** Dashboard session token (required)

**Request body** (`application/json`):

```json
{
  "name": "Daily Briefing",
  "schedule": "at:08:00",
  "enabled": true,
  "callback_name": "daily_briefing",
  "workflow_id": null,
  "metadata": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Human-readable job name |
| `schedule` | string | Yes | `"interval:SECONDS"` or `"at:HH:MM"` |
| `enabled` | boolean | No (default `true`) | Whether the job runs automatically |
| `callback_name` | string | No | Name of the registered callback function |
| `workflow_id` | string | No | ID of an associated workflow |
| `metadata` | object | No | Arbitrary key-value metadata |

**Response 201:**

```json
{
  "job_id": "job_abc123",
  "name": "Daily Briefing",
  "schedule": "at:08:00",
  "enabled": true,
  "next_run": "2026-03-13T08:00:00"
}
```

**Error codes:** 400 (invalid schedule format), 401, 500

---

### GET /api/scheduler/jobs/{job_id}

Get a specific job by ID.

**Authentication:** Dashboard session token (required)

**Path parameters:** `job_id` — job identifier returned at creation

**Response 200:**

```json
{
  "job_id": "job_abc123",
  "name": "Daily Briefing",
  "schedule": "at:08:00",
  "enabled": true,
  "next_run": "2026-03-13T08:00:00",
  "last_run_at": "2026-03-12T08:00:00",
  "run_count": 5,
  "max_runs": null,
  "callback_name": "daily_briefing",
  "workflow_id": null,
  "metadata": {}
}
```

**Error codes:** 401, 404 (not found), 500

---

### POST /api/scheduler/jobs/{job_id}/run

Manually trigger a job immediately.

**Authentication:** Dashboard session token (required)

**Path parameters:** `job_id`

**Response 200:**

```json
{
  "job_id": "job_abc123",
  "success": true,
  "message": "Job executed"
}
```

**Error codes:** 401, 404, 500

---

### PATCH /api/scheduler/jobs/{job_id}

Update a job (enable/disable, change schedule, etc.).

**Authentication:** Dashboard session token (required)

**Path parameters:** `job_id`

**Request body** (`application/json`):

```json
{"enabled": false}
```

Accepted fields: `enabled`, `schedule`, `name`, `max_runs`, `metadata`.

**Response 200:**

```json
{
  "job_id": "job_abc123",
  "name": "Daily Briefing",
  "schedule": "at:08:00",
  "enabled": false,
  "next_run": null
}
```

**Error codes:** 400 (no valid updates), 401, 404, 500

---

### DELETE /api/scheduler/jobs/{job_id}

Delete a scheduled job.

**Authentication:** Dashboard session token (required)

**Path parameters:** `job_id`

**Response 200:**

```json
{"job_id": "job_abc123", "deleted": true}
```

**Error codes:** 401, 404, 500

---

### GET /api/notifications

List recent notifications.

**Authentication:** Dashboard session token (required)

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | 50 | Max entries (capped at 200) |
| `unread` | `false` | If `true`, return only unread notifications |
| `priority` | (all) | Filter by priority level |

**Response 200:**

```json
{
  "notifications": [
    {
      "id": "notif_xyz",
      "message": "Daily briefing completed",
      "priority": "normal",
      "read": false,
      "created_at": "2026-03-12T08:00:01"
    }
  ],
  "total": 1,
  "unread_count": 1
}
```

**Error codes:** 401, 403, 500

---

### POST /api/notifications/{notification_id}/read

Mark a notification as read.

**Authentication:** Dashboard session token (required)

**Path parameters:** `notification_id`

**Response 200:**

```json
{"id": "notif_xyz", "read": true}
```

**Error codes:** 401, 404, 500

---

### POST /api/notifications/read-all

Mark all notifications as read.

**Authentication:** Dashboard session token (required)

**Response 200:**

```json
{"marked_read": 5}
```

**Error codes:** 401, 403, 500

---

### GET /api/notifications/stream

Stream notification events via **Server-Sent Events** (SSE).

**Authentication:** Dashboard session token (required)

**Response:** `text/event-stream`

```
event: init
data: {"unread_count": 3}

event: notification
data: {"id": "notif_xyz", "message": "...", "user_id": "alice"}
```

The stream stays open until the client disconnects or a 15-second idle timeout
is reached. Reconnect automatically to resume.

**Error codes:** 401

---

### POST /api/voice

Transcribe audio and return an LLM reply.

**Authentication:** Dashboard session token (required)

**Request:** `multipart/form-data` with field `audio` containing the recorded
audio blob (WAV or WebM).

**Response 200:**

```json
{
  "transcript": "What is the weather today?",
  "reply": "I don't have live weather data...",
  "timestamp": "2026-03-12T10:05:00"
}
```

**Error codes:** 400 (no audio file), 401, 500 (Whisper not installed or transcription error)

---

## TTS API Endpoints

Base URL: `http://localhost:5001`

---

### POST /speak

Convert text to speech and return an audio file.

**Authentication:** `Authorization: Bearer <REX_SPEAK_API_TOKEN>`

**Request body** (`application/json`):

```json
{
  "text": "Hello, how can I help you today?",
  "voice": "default"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `voice` | string | No | Voice identifier (default: configured voice) |

**Response 200:**

Binary audio data (`audio/wav` or `audio/mpeg`).

**Error codes:** 400 (missing text), 401 (invalid token), 503 (TTS model not loaded), 500

---

## Rate Limiting

All Flask proxy endpoints are subject to a default rate limit of
**60 requests per minute** per IP address (configurable via `API_RATE_LIMIT`).

Health endpoints (`/health/live`, `/health/ready`) are **exempt** from rate limiting.

When the limit is exceeded, the server responds with HTTP 429:

```json
{
  "error": {
    "code": "TOO_MANY_REQUESTS",
    "message": "Too many requests. Please slow down."
  }
}
```

The response includes a `Retry-After` header indicating how many seconds to wait.
