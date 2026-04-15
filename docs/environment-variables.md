# AskRex Assistant Environment Variables

AskRex uses two configuration channels:

- `config/rex_config.json` for runtime settings such as models, audio devices,
  wake word, providers, integrations, and workflow defaults.
- `.env` for secrets and service-specific environment controls.

Non-secret legacy environment variables should be migrated with:

```bash
rex-config migrate-legacy-env --dry-run
rex-config migrate-legacy-env
```

## Core Identity and Voice Settings

| Variable | Default | Description |
|---|---:|---|
| `REX_ACTIVE_USER` | `default` | Active user profile name |
| `REX_LOG_LEVEL` | `INFO` | Agent log level (also see Logging section) |
| `REX_WAKEWORD` | `hey rex` | Wake word keyword override |
| `REX_INPUT_DEVICE` | auto | Audio input device index or name |
| `REX_SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `REX_LLM_PROVIDER` | `openai` | LLM backend provider (`openai`, `ollama`, `local`) |
| `REX_TTS_PROVIDER` | `xtts` | TTS backend provider (`xtts`, `edge-tts`, `pyttsx3`) |

## Core Secrets

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM backend |
| `ANTHROPIC_API_KEY` | Anthropic LLM backend |
| `OLLAMA_API_KEY` | Cloud-hosted Ollama endpoint if it requires auth |
| `HA_TOKEN` | Home Assistant API |
| `HA_SECRET` | Home Assistant webhook/auth flows |
| `BRAVE_API_KEY` | Brave search |
| `SERPAPI_KEY` | SerpAPI search |
| `GOOGLE_API_KEY` | Google Custom Search |
| `GOOGLE_CSE_ID` | Google Custom Search engine ID |
| `OPENWEATHERMAP_API_KEY` | Weather tool |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway |
| `TELEGRAM_BOT_TOKEN` | Telegram integration |
| `PUSH_TOKEN` | Push notification provider |

Provider base URLs and model choices generally belong in `config/rex_config.json`
instead of `.env`.

## TTS API (`rex-speak-api`)

| Variable | Default | Description |
|---|---:|---|
| `REX_SPEAK_API_KEY` | none | Required for `/speak` |
| `REX_SPEAK_PORT` | `5005` | Bind port |
| `REX_TTS_MODEL` | XTTS v2 default | Coqui model override used by the TTS API |
| `REX_SPEAK_MAX_CHARS` | `800` | Request text length limit |
| `REX_SPEAK_MAX_REQUEST_BYTES` | `65536` | Body size limit |
| `REX_SPEAK_RATE_LIMIT` | `30` | Requests per window |
| `REX_SPEAK_RATE_WINDOW` | `60` | Window in seconds |
| `REX_SPEAK_STORAGE_URI` | `memory://` | Limiter storage; use Redis for multi-worker |
| `FLASK_LIMITER_STORAGE_URI` | fallback only | Fallback limiter storage if speak storage is unset |
| `REX_ALLOWED_ORIGINS` | localhost defaults | CORS allowlist |

Default address:

```text
http://127.0.0.1:5005
```

## Tool Server (`rex-tool-server`)

| Variable | Default | Description |
|---|---:|---|
| `REX_TOOL_API_KEY` | none | Required bearer token for `/rex/tools/*` |
| `REX_TOOL_SERVER_PORT` | `18790` | Bind port |
| `REX_TOOL_RATE_LIMIT` | `60` | Requests per window |
| `REX_TOOL_RATE_WINDOW` | `60` | Window in seconds |

Default address:

```text
http://127.0.0.1:18790
```

## Python Web Dashboard (`rex-gui`)

| Variable | Default | Description |
|---|---:|---|
| `REX_GUI_PORT` | `8765` | Dashboard port |
| `REX_DATA_DIR` | `data` | Local auth/permissions/history data directory |
| `SERPAPI_API_KEY` | none | Dashboard integration-status check only |
| `BRAVE_API_KEY` | none | Dashboard integration-status check and search |
| `GOOGLE_CSE_ID` | none | Dashboard integration-status check and search |
| `TWILIO_ACCOUNT_SID` | none | SMS integration/status |
| `TWILIO_AUTH_TOKEN` | none | SMS integration/status |
| `MQTT_BROKER_HOST` | none | MQTT integration-status check |

Default address:

```text
http://127.0.0.1:8765/ui/
```

## Legacy Flask Proxy

`flask_proxy.py` is a compatibility surface. Prefer `rex-gui` for the current
web dashboard.

| Variable | Default | Description |
|---|---:|---|
| `REX_PROXY_TOKEN` | none | Bearer-token auth for non-local/proxy calls |
| `REX_PROXY_ALLOW_LOCAL` | `0` | Allow unauthenticated local requests |
| `REX_ALLOWED_ORIGINS` | localhost defaults | CORS allowlist |
| `REX_TRUSTED_PROXIES` | `127.0.0.1,::1` | Trusted reverse-proxy IPs |
| `API_RATE_LIMIT` | `60 per minute` | Flask-Limiter string |
| `FLASK_LIMITER_STORAGE_URI` | `memory://` | Limiter storage |
| `SKIP_MIGRATION_CHECK` | unset | Emergency bypass for legacy startup migration validation |
| `REX_SHUTDOWN_TIMEOUT` | `5` | Graceful shutdown drain timeout |

## Database/State Helpers

| Variable | Default | Description |
|---|---:|---|
| `REX_DATA_DIR` | `data` | Auth, permissions, and command-history state |
| `DB_POOL_MIN_SIZE` | `1` | SQLite pool minimum |
| `DB_POOL_MAX_SIZE` | `5` | SQLite pool maximum |
| `DB_POOL_ACQUIRE_TIMEOUT` | `5.0` | Pool acquire timeout |
| `DB_POOL_IDLE_TIMEOUT` | `300.0` | Idle connection replacement timeout |
| `DB_QUERY_TIMEOUT` | `10.0` | Query timeout |
| `REX_LLM_USAGE_PATH` | `data/llm_usage.jsonl` | LLM usage log path |

## Computer Agent (`rex-agent`)

| Variable | Default | Description |
|---|---:|---|
| `REX_AGENT_HOST` | `127.0.0.1` | Bind host |
| `REX_AGENT_PORT` | code default | Bind port |
| `REX_AGENT_TOKEN_ENV` | `REX_AGENT_TOKEN` | Env var name containing the auth token |
| `REX_AGENT_TOKEN` | none | Default token env value |
| `REX_AGENT_ALLOWLIST` | `whoami` | Server-side command allowlist |
| `REX_AGENT_RATE_LIMIT` | `60` | Requests per minute per client IP |
| `REX_AGENT_TIMEOUT` | `30` | Command execution timeout in seconds |
| `REX_AGENT_MAX_OUTPUT` | `65536` | Output size limit in bytes |

Keep the agent localhost-only unless an explicit deployment adds a network and
auth boundary.

## Plugin and Tool Execution Controls

| Variable | Default | Description |
|---|---:|---|
| `REX_PLUGIN_TIMEOUT` | `30` | Plugin execution timeout |
| `REX_PLUGIN_OUTPUT_LIMIT` | `1048576` | Plugin output byte limit |
| `REX_PLUGIN_RATE_LIMIT` | `10` | Plugin invocations per minute |

## Messaging, Calendar, Email, Telephony

| Variable | Used by |
|---|---|
| `TWILIO_ACCOUNT_SID` | SMS and telephony |
| `TWILIO_AUTH_TOKEN` | SMS and telephony |
| `TWILIO_FROM_NUMBER` | SMS and telephony |
| `REX_BASE_URL` | Telephony callback base URL |
| `GOOGLE_CALENDAR_ACCESS_TOKEN` | Google calendar integration path |
| `GMAIL_ACCESS_TOKEN` | Gmail integration path |

Connection details and account ownership belong in `config/rex_config.json`
where supported.

## Logging and Debugging

| Variable | Default | Description |
|---|---:|---|
| `LOG_LEVEL` | `INFO` | Root log level in several scripts/services |
| `REX_LOG_LEVEL` | `INFO` | Agent log level |
| `REX_DEBUG` | unset | Enables debug mode for CLI/config paths |
| `REX_JSON_LOGS` | auto | JSON logging toggle |
| `REX_LOG_FULL_IP` | `0` | Log full client IPs instead of anonymized IPs |
| `REX_TESTING` | unset | Testing mode for selected code paths |

## Windows Service

| Variable | Default | Description |
|---|---:|---|
| `REX_SERVICES` | `speak,proxy` | Windows service sub-services |
| `REX_SERVICE_PORT` | `5100` | Windows service manager port |

## Configured in `config/rex_config.json`, Not `.env`

Keep these in JSON config:

- wake word backend, keyword, threshold, and model paths
- audio input/output devices and sample rate
- LLM provider, model, max tokens, temperature, and routing
- OpenAI base URL and model
- Ollama base URL and cloud toggle
- Home Assistant base URL and SSL behavior
- calendar backend and ICS source
- messaging backend/accounts
- notification dashboard and Home Assistant TTS settings
- WordPress/WooCommerce site definitions
- workflow tasks
- user-to-account mappings

## Adding a New Variable

When adding a new environment variable:

1. Add it to `.env.example` if users should set it manually.
2. Add it to this file.
3. Keep secrets in `.env` and non-secrets in `config/rex_config.json` unless
   the variable is genuinely process-level service control.
