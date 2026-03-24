# OpenClaw Agent Setup

This document describes how Rex integrates with the OpenClaw gateway over
HTTP, how to configure the integration, and how Rex's tools are exposed to
OpenClaw channels.

> **Status:** Phase 8 -- HTTP integration complete.  Rex routes LLM calls
> through OpenClaw's `/v1/chat/completions` endpoint and exposes its tools
> via an HTTP tool server.

---

## 1. Architecture

Rex communicates with OpenClaw exclusively over HTTP.  OpenClaw is a
TypeScript/Node.js gateway; there is no Python package to import.

```
Rex voice loop / CLI
        |
        | (1) POST /v1/chat/completions
        v
  OpenClaw Gateway  (:18789)
        |
        +---> Ollama / OpenAI / Anthropic / ...
        |
        | (2) POST /rex/tools/{tool_name}
        v
  Rex Tool Server  (:18790)
```

Path (1): Rex sends user prompts to OpenClaw, which routes them to
whichever model provider is configured.

Path (2): OpenClaw (or any authorized caller) invokes Rex's tools
(time, weather, email, SMS, calendar, Home Assistant, Plex, WooCommerce)
over HTTP.

---

## 2. Package Layout

All OpenClaw integration code lives in `rex/openclaw/`:

```
rex/openclaw/
+-- __init__.py          # Subpackage docstring
+-- http_client.py       # OpenClawClient -- shared HTTP client
+-- errors.py            # OpenClawConnectionError, AuthError, APIError
+-- agent.py             # RexAgent class
+-- config.py            # build_agent_config(), build_system_prompt()
+-- session.py           # build_session_context()
+-- voice_bridge.py      # VoiceBridge for voice loop integration
+-- tool_bridge.py       # ToolBridge for dual-mode tool dispatch
+-- tool_server.py       # Flask HTTP server exposing Rex tools
+-- event_bridge.py      # Local event bus (no HTTP)
+-- browser_bridge.py    # Local Playwright automation (no HTTP)
+-- identity_adapter.py  # User resolution + OpenClaw user key
+-- memory_adapter.py    # Local memory with implicit session sync
+-- approval_adapter.py  # Local file-based approvals
+-- policy_adapter.py    # Local policy engine
+-- workflow_bridge.py   # Local workflow execution
+-- tools/               # Tool handler functions
    +-- time_tool.py
    +-- weather_tool.py
    +-- email_tool.py
    +-- sms_tool.py
    +-- calendar_tool.py
    +-- ha_tool.py
    +-- plex_tool.py
    +-- woocommerce_tool.py
    +-- wordpress_tool.py
    +-- business_tool.py
```

---

## 3. Configuration

### 3.1 `config/rex_config.json`

```json
{
  "llm": {
    "provider": "openai",
    "model": "openclaw:main"
  },
  "openai": {
    "base_url": "http://127.0.0.1:18789/v1"
  },
  "openclaw": {
    "gateway_url": "http://127.0.0.1:18789",
    "gateway_timeout": 30,
    "gateway_max_retries": 3,
    "use_tools": false,
    "use_voice_backend": false
  }
}
```

### 3.2 `.env`

```
OPENCLAW_GATEWAY_TOKEN=<your-operator-token>
```

The token is sent as `Authorization: Bearer <token>` on every HTTP request
to the OpenClaw gateway.

### 3.3 Feature flags

| Flag | Config path | Effect |
|------|-------------|--------|
| `use_openclaw_voice_backend` | `openclaw.use_voice_backend` | When True, voice loops swap `Assistant` for `VoiceBridge`, routing LLM calls through the OpenClaw gateway |
| `use_openclaw_tools` | `openclaw.use_tools` | When True, `ToolBridge.execute_tool()` dispatches tool calls to OpenClaw's `/tools/invoke` endpoint instead of running them locally |

Both flags default to `false`.  When false, Rex operates in standalone
mode with zero HTTP calls to OpenClaw.

---

## 4. Quick Start

1. Install and start the OpenClaw gateway (see OpenClaw docs).
2. Copy `.env.example` to `.env` and set `OPENCLAW_GATEWAY_TOKEN`.
3. Update `config/rex_config.json` with the values in section 3.1.
4. Start Rex: `python -m rex` (text mode) or `python rex_loop.py` (voice mode).
5. Verify: watch OpenClaw logs for incoming `/v1/chat/completions` requests.

---

## 5. LLM Routing via OpenClaw

Rex uses its existing `OpenAIStrategy` in `rex/llm_client.py` to talk to
OpenClaw.  Because OpenClaw's `/v1/chat/completions` endpoint is
OpenAI-compatible, no special client code is needed for basic LLM routing.

When `use_openclaw_voice_backend` is True and an OpenClaw gateway URL is
configured, `RexAgent.respond()` sends prompts directly to the gateway
via `OpenClawClient.post("/v1/chat/completions", ...)`.  On HTTP error,
it falls back to the local LLM automatically.

### Session persistence via the `user` field

Every chat completions request includes a `user` field derived from
`IdentityAdapter.get_openclaw_user_key()`:

- Explicit `user_key` parameter takes priority
- Otherwise: session user > config `active_user` > config `user_id` > `"rex"`

OpenClaw uses this string to maintain per-user session state, so
conversation context persists across Rex restarts.

---

## 6. Persona / System Prompt

`build_system_prompt(config)` in `rex/openclaw/config.py` derives the
persona string from `AppConfig` fields:

| AppConfig field    | Effect                                |
|--------------------|---------------------------------------|
| `wakeword`         | Agent name (capitalized)              |
| `active_profile`   | Mentioned if not `"default"`          |
| `default_location` | `"The user's location is {loc}."`     |
| `default_timezone` | `"The local timezone is {tz}."`       |
| `capabilities`     | `"Your capabilities include: {caps}"` |

The persona is injected as a `system` role message in every LLM call.

---

## 7. Rex Tool Server (HTTP endpoint for OpenClaw)

Rex exposes its tools as HTTP endpoints so that the OpenClaw gateway (and
any other authorized caller) can invoke Rex's tools directly.

### 7.1 Starting the tool server

```bash
export REX_TOOL_API_KEY=<strong-random-secret>
export REX_TOOL_SERVER_PORT=18790   # default

rex-tool-server
```

Health checks (no auth required):

```bash
curl http://127.0.0.1:18790/health/live
# {"status": "ok"}

curl http://127.0.0.1:18790/health/ready
# {"status": "ok", "tool_count": 14}
```

### 7.2 Calling a tool

```bash
curl -X POST http://127.0.0.1:18790/rex/tools/time_now \
  -H "Authorization: Bearer $REX_TOOL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"args": {"location": "Edinburgh"}, "context": {"session_key": "main"}}'
```

### 7.3 Available tool endpoints

| Endpoint path                            | Tool function         | Required env var         |
|------------------------------------------|-----------------------|--------------------------|
| `/rex/tools/time_now`                    | `time_now`            | none                     |
| `/rex/tools/weather_now`                 | `weather_now`         | `OPENWEATHERMAP_API_KEY` |
| `/rex/tools/send_email`                  | `send_email`          | email backend configured |
| `/rex/tools/send_sms`                    | `send_sms`            | `TWILIO_*` vars          |
| `/rex/tools/calendar_create`             | `calendar_create`     | calendar backend config  |
| `/rex/tools/home_assistant_call_service` | `ha_call_service`     | `HOME_ASSISTANT_URL`     |
| `/rex/tools/plex_search`                | `plex_search`         | `PLEX_*` vars            |
| `/rex/tools/plex_play`                  | `plex_play`           | `PLEX_*` vars            |
| `/rex/tools/plex_pause`                 | `plex_pause`          | `PLEX_*` vars            |
| `/rex/tools/plex_stop`                  | `plex_stop`           | `PLEX_*` vars            |
| `/rex/tools/wordpress_health_check`     | `wp_health_check`     | `WORDPRESS_*` vars       |
| `/rex/tools/wc_list_orders`             | `wc_list_orders`      | `WOOCOMMERCE_*` vars     |
| `/rex/tools/wc_list_products`           | `wc_list_products`    | `WOOCOMMERCE_*` vars     |
| `/rex/tools/wc_set_order_status`        | `wc_set_order_status` | `WOOCOMMERCE_*` vars     |
| `/rex/tools/wc_create_coupon`           | `wc_create_coupon`    | `WOOCOMMERCE_*` vars     |
| `/rex/tools/wc_disable_coupon`          | `wc_disable_coupon`   | `WOOCOMMERCE_*` vars     |

Tools whose optional dependencies are not installed are omitted from the
registry at startup (logged at WARNING level).

### 7.4 Auth and rate limiting

- Auth: `Authorization: Bearer <REX_TOOL_API_KEY>` or `X-API-Key: <REX_TOOL_API_KEY>`
- Default rate limit: 60 requests / 60 seconds (configurable via `REX_TOOL_RATE_LIMIT` / `REX_TOOL_RATE_WINDOW`)
- Policy checks run before every tool call; denied requests return 403

### 7.5 Configuring OpenClaw to call Rex tools

Add Rex's tool server to OpenClaw's skill/tool configuration so any
OpenClaw channel (WhatsApp, Telegram, Discord, etc.) can invoke Rex's tools.

Example OpenClaw skill config (JSON):

```json
{
  "skills": [
    {
      "name": "rex_time_now",
      "description": "Get the current local time for a location",
      "endpoint": "http://127.0.0.1:18790/rex/tools/time_now",
      "method": "POST",
      "auth": { "type": "bearer", "token_env": "REX_TOOL_API_KEY" },
      "schema": {
        "args": {
          "location": { "type": "string", "description": "City or timezone" }
        }
      }
    }
  ]
}
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on startup | OpenClaw not running | Start OpenClaw gateway first |
| `401 Unauthorized` | Wrong or missing token | Check `OPENCLAW_GATEWAY_TOKEN` in `.env` |
| Slow responses | High gateway timeout | Lower `openclaw.gateway_timeout` |
| Rex falls back to echo mode | `openai` package not installed | `pip install openai` |
| Tool calls return 404 | Tool server not started | Run `rex-tool-server` |
| Tool calls return 403 | Policy denied | Check Rex policy config |

---

## 9. Local-Only Components

These adapters operate locally and make no HTTP calls to OpenClaw:

- **EventBridge**: Rex's internal event bus.  OpenClaw event bridging (WebSocket) is a future concern.
- **BrowserBridge**: Runs Playwright locally.  OpenClaw has its own browser automation.
- **ApprovalAdapter**: File-based approvals.  OpenClaw has its own WebSocket-based approval flow.
- **MemoryAdapter**: Rex's file-based conversation memory.  Session state in OpenClaw is maintained implicitly via the `user` field in chat completions.
- **IdentityAdapter**: Resolves users locally.  Provides `get_openclaw_user_key()` for the `user` field.
