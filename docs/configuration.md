# Rex AI Assistant — Configuration Reference

This document lists every environment variable recognised by Rex AI Assistant.
Variables marked **REQUIRED** must be set for the relevant feature to work.
All others are optional and use the shown default when omitted.

**Where to put them:** Secrets and API keys belong in `.env`. Runtime settings
(models, audio devices, wake-word config) belong in `config/rex_config.json`.
See `.env.example` for a copy-paste template.

---

## Server

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REX_PROXY_TOKEN` | (none) | Yes (remote auth) | Bearer token required for API calls from non-local clients |
| `REX_PROXY_ALLOW_LOCAL` | `0` | No | Allow unauthenticated requests from `127.0.0.1` / `::1` |
| `REX_TRUSTED_PROXIES` | `127.0.0.1,::1` | No | Comma-separated list of reverse-proxy IPs whose `X-Forwarded-For` header is trusted |
| `REX_ACTIVE_USER` | `local` | No | Default user profile key for memory/voice identity selection |
| `REX_SHUTDOWN_TIMEOUT` | `5` | No | Graceful shutdown drain timeout in seconds |
| `API_RATE_LIMIT` | `60 per minute` | No | Flask-Limiter rate limit string applied to all public endpoints |
| `FLASK_LIMITER_STORAGE_URI` | `memory://` | No | Storage URI for the main API rate limiter (use Redis in multi-worker setups) |
| `REX_ALLOWED_ORIGINS` | localhost URLs | No | Comma-separated CORS allowed origins |

---

## Database

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `DB_POOL_MIN_SIZE` | `1` | No | Minimum number of SQLite connections kept open |
| `DB_POOL_MAX_SIZE` | `5` | No | Maximum concurrent SQLite connections |
| `DB_POOL_ACQUIRE_TIMEOUT` | `5.0` | No | Seconds to wait for an available connection before raising `ConnectionPoolError` |
| `DB_POOL_IDLE_TIMEOUT` | `300.0` | No | Seconds of inactivity after which an idle connection is replaced. `-1` disables |
| `DB_QUERY_TIMEOUT` | `10.0` | No | Seconds before a running query is interrupted and `QueryTimeoutError` raised. `-1` disables |
| `SKIP_MIGRATION_CHECK` | (unset) | No | Set to `1`, `true`, or `yes` to bypass startup migration validation (emergency use only) |

---

## LLM Providers

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `OPENAI_API_KEY` | (none) | Yes (OpenAI) | OpenAI API key |
| `OPENAI_BASE_URL` | SDK default | No | Override for the OpenAI-compatible API base URL |
| `ANTHROPIC_API_KEY` | (none) | Yes (Anthropic) | Anthropic / Claude API key |
| `OLLAMA_API_KEY` | (none) | No | Auth token for cloud-hosted Ollama endpoints |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | No | Ollama API base URL |

---

## Integrations

### Web Search

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `BRAVE_API_KEY` | (none) | Yes (Brave) | Brave Search API key |
| `BRAVE_URL` | `https://api.search.brave.com/res/v1/web/search` | No | Brave Search API endpoint |
| `SERPAPI_KEY` | (none) | Yes (SerpAPI) | SerpAPI key for Google search |
| `SERPAPI_URL` | `https://serpapi.com/search` | No | SerpAPI endpoint URL |
| `GOOGLE_API_KEY` | (none) | Yes (Google CSE) | Google Custom Search API key |
| `GOOGLE_CSE_ID` | (none) | Yes (Google CSE) | Google Custom Search Engine ID |
| `GOOGLE_URL` | `https://www.googleapis.com/customsearch/v1` | No | Google Custom Search API endpoint |

### Home Assistant

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `HA_BASE_URL` | (none) | Yes (HA) | Home Assistant base URL, e.g. `http://homeassistant.local:8123` |
| `HA_TOKEN` | (none) | Yes (HA) | Home Assistant long-lived access token |
| `HA_SECRET` | (none) | No | Webhook secret for incoming HA webhook authentication |

### Browserless (Web Scraping)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `BROWSERLESS_URL` | `https://chrome.browserless.io` | No | Browserless API endpoint |
| `BROWSERLESS_API_KEY` | (none) | Yes (Browserless) | Browserless API key |

### TTS / Rex Speak API

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REX_SPEAK_API_KEY` | (none) | Yes (Speak API) | Authentication key for the TTS API server |
| `REX_VOICE_MAX_TOKENS` | `150` | No | Voice-only max output tokens for LLM replies. When set, voice interactions cap generated length for concise spoken responses; text/chat mode remains governed by JSON `models.llm_max_tokens`. |
| `REX_SPEAK_PORT` | `5005` | No | Port the Rex Speak API server listens on |
| `REX_TTS_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | No | Coqui XTTS model identifier |
| `REX_SPEAK_MAX_CHARS` | `800` | No | Maximum text length per TTS request (characters) |
| `REX_SPEAK_RATE_LIMIT` | `30` | No | TTS API requests allowed per rate window |
| `REX_SPEAK_RATE_WINDOW` | `60` | No | Rate limit window in seconds |
| `REX_SPEAK_STORAGE_URI` | `memory://` | No | Storage URI for TTS API rate limiter |

### Dashboard

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REX_DASHBOARD_PASSWORD` | (none) | Yes (dashboard login) | Dashboard login password |
| `REX_DASHBOARD_SECRET` | (random) | Yes (production) | Secret key for signing session tokens; a random value is used if absent (invalidated on restart) |
| `REX_DASHBOARD_SESSION_EXPIRY` | `28800` | No | Session expiry in seconds (default 8 hours) |
| `REX_DASHBOARD_ALLOW_LOCAL` | `1` | No | Allow unauthenticated localhost dashboard access |
| `REX_LOGIN_MAX_ATTEMPTS` | `5` | No | Maximum failed login attempts before lockout |
| `REX_LOGIN_LOCKOUT_SECONDS` | `300` | No | Lockout duration in seconds after excessive failed logins |

### Agent Server

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REX_AGENT_HOST` | `127.0.0.1` | No | Host address the agent server binds to |
| `REX_AGENT_TOKEN_ENV` | `REX_AGENT_API_KEY` | No | Name of the env var from which the agent auth token is read |
| `REX_AGENT_ALLOWLIST` | `*` | No | Comma-separated allowlist of permitted agent IDs |

### Plugin System

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REX_PLUGIN_TIMEOUT` | `30` | No | Maximum seconds a plugin may run before being killed |
| `REX_PLUGIN_OUTPUT_LIMIT` | `1048576` | No | Maximum bytes of output per plugin invocation (1 MiB) |
| `REX_PLUGIN_RATE_LIMIT` | `10` | No | Max plugin invocations per minute per plugin |

### Windows Service

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REX_SERVICES` | `speak,proxy` | No | Comma-separated list of Rex sub-services managed by the Windows service |
| `REX_SERVICE_PORT` | `5100` | No | Port used by the Windows service manager |

---

## Logging

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `LOG_LEVEL` | `INFO` | No | Root log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `REX_JSON_LOGS` | auto | No | Enable JSON structured logging (`1`/`true`). Defaults on in production, off under pytest |
| `REX_LOG_FULL_IP` | `0` | No | Log full client IP addresses (`1` = yes, `0` = anonymize last octet/64 bits) |

---

## Testing / Development

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REX_TESTING` | (unset) | No | Set to `1` to enable testing mode (skips certain initialisation side-effects in flask_proxy) |
| `REX_ALLOWED_ORIGINS` | localhost URLs | No | CORS whitelist override |

---

## Adding New Variables

When introducing a new environment variable:

1. Add it to `.env.example` with a comment describing purpose and default.
2. Add it to the appropriate section of this document.
3. Use a `_float()` / `_int()` helper (see `rex/db_pool.py`) for safe parsing with fallback.
