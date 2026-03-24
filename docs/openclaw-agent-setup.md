# OpenClaw Agent Setup

This document describes how to bootstrap Rex as an OpenClaw agent, how Rex
config maps to OpenClaw agent configuration, and how Rex tools are registered
with OpenClaw's tool system.

> **Status:** Phase 2 foundation — agent, config, session, and tool adapters
> are in place.  OpenClaw registration stubs will be filled in once the
> OpenClaw API surface is confirmed (see PRD §8.3 open dependencies).

---

## 1. Package Layout

All OpenClaw integration code lives in `rex/openclaw/`:

```
rex/openclaw/
├── __init__.py          # Subpackage docstring; no auto-imports
├── agent.py             # RexAgent class
├── config.py            # build_agent_config(), build_system_prompt()
├── session.py           # build_session_context()
└── tools/
    ├── __init__.py      # Tools subpackage
    ├── time_tool.py     # time_now tool adapter
    └── weather_tool.py  # weather_now tool adapter
```

---

## 2. Bootstrap Steps

### 2.1 Create the Agent

```python
from rex.openclaw.agent import RexAgent

agent = RexAgent()          # loads global AppConfig automatically
agent.register()            # registers with OpenClaw (no-op if not installed)
```

To pass an explicit config or override the persona:

```python
from rex.config import AppConfig
from rex.openclaw.agent import RexAgent

cfg = AppConfig(
    wakeword="rex",
    default_location="Edinburgh, Scotland",
    default_timezone="Europe/London",
)
agent = RexAgent(config=cfg)
```

### 2.2 Send a Prompt

```python
reply = agent.respond("What time is it?")
print(reply)
```

`respond()` raises `ValueError` for empty or whitespace-only prompts.

### 2.3 Register Tools

```python
from rex.openclaw.tools.time_tool import register as register_time
from rex.openclaw.tools.weather_tool import register as register_weather

register_time(agent=agent)      # no-op + warning if openclaw not installed
register_weather(agent=agent)
```

---

## 3. Config Mapping

`build_agent_config(config)` in `rex/openclaw/config.py` maps `AppConfig`
fields to a flat dict for OpenClaw:

| AppConfig field         | OpenClaw key          | Notes                                      |
|-------------------------|-----------------------|--------------------------------------------|
| `wakeword.capitalize()` | `agent_name`          | "rex" → "Rex"                              |
| `user_id`               | `user_id`             |                                            |
| `active_profile`        | `active_profile`      |                                            |
| `llm_provider`          | `llm_provider`        |                                            |
| `llm_model`             | `llm_model`           |                                            |
| `llm_temperature`       | `llm_temperature`     |                                            |
| `llm_top_p`             | `llm_top_p`           |                                            |
| `llm_top_k`             | `llm_top_k`           |                                            |
| `llm_max_tokens`        | `llm_max_tokens`      |                                            |
| `default_location`      | `default_location`    |                                            |
| `default_timezone`      | `default_timezone`    |                                            |
| `memory_max_turns`      | `memory_max_turns`    |                                            |
| `memory_max_bytes`      | `memory_max_bytes`    |                                            |
| `capabilities`          | `rex_capabilities`    | prefixed `rex_*` until OpenClaw schema confirmed |
| `tts_provider`          | `rex_tts_provider`    | Rex-specific                               |
| `speak_language`        | `rex_speak_language`  | Rex-specific                               |

Keys prefixed `rex_*` are Rex-specific extras that will be remapped once the
OpenClaw config schema is confirmed.

---

## 4. Persona / System Prompt

`build_system_prompt(config)` in `rex/openclaw/config.py` derives the persona
string from `AppConfig` fields:

```python
from rex.openclaw.config import build_system_prompt

prompt = build_system_prompt()   # uses global config
# "You are Rex, a helpful and friendly AI assistant. The user's location is Edinburgh, Scotland."
```

Fields included in the prompt when set:

| AppConfig field    | Effect                                          |
|--------------------|-------------------------------------------------|
| `wakeword`         | Agent name — `wakeword.capitalize()`            |
| `active_profile`   | Mentioned if not `"default"`                    |
| `default_location` | `"The user's location is {location}."`          |
| `default_timezone` | `"The local timezone is {timezone}."`           |
| `capabilities`     | `"Your capabilities include: {caps}."`          |

The persona is injected as a `system` role message in every LLM call made
through `RexAgent.respond()`.

---

## 5. Session Bridge

`build_session_context()` in `rex/openclaw/session.py` maps Rex user identity
to a flat session dict for OpenClaw:

```python
from rex.openclaw.session import build_session_context

ctx = build_session_context(explicit_user="alice", metadata={"channel": "voice"})
# {
#   "user_id": "alice",
#   "session_started_at": "2026-03-22T14:30:00+00:00",
#   "rex_known_users": [...],
#   "rex_user_profile": {...},
#   "channel": "voice",
# }
```

---

## 6. Tool Registration

Each tool module in `rex/openclaw/tools/` exposes:

- A standalone callable (e.g. `time_now(location, context)`)
- A `register(agent=None)` function for OpenClaw registration
- `TOOL_NAME` — the canonical tool name matching `rex/tool_router.py`
- `TOOL_DESCRIPTION` — human-readable description

Both tool callables delegate to `rex.tool_router.execute_tool` with policy,
credential, and audit checks disabled (read-only, no side effects).

### Available Tools

| Module                              | TOOL_NAME      | Required env var            |
|-------------------------------------|----------------|-----------------------------|
| `rex.openclaw.tools.time_tool`      | `time_now`     | none                        |
| `rex.openclaw.tools.weather_tool`   | `weather_now`  | `OPENWEATHERMAP_API_KEY`    |

### Direct Tool Usage (without OpenClaw)

```python
from rex.openclaw.tools.time_tool import time_now
from rex.openclaw.tools.weather_tool import weather_now

print(time_now("London"))
# {'local_time': '2026-03-22 14:30', 'date': '2026-03-22', 'timezone': 'Europe/London'}

print(weather_now("Edinburgh"))
# {'temperature': 8.5, 'description': 'light rain', ...}
# or {'error': {...}} if OPENWEATHERMAP_API_KEY not set
```

---

## 7. OpenClaw Availability Flag

Every module in `rex/openclaw/` sets:

```python
from importlib.util import find_spec

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None
```

When `OPENCLAW_AVAILABLE` is `False`, all `register()` calls log a warning
and return `None`.  All callables (`respond`, `time_now`, `weather_now`,
`build_session_context`) work normally without OpenClaw installed.

---

## 8. HTTP Integration

Rex can route all LLM calls through the OpenClaw gateway's
`/v1/chat/completions` endpoint, which is OpenAI-compatible.  This lets Rex
use any model provider configured in OpenClaw (Ollama, OpenAI, Anthropic,
etc.) without changing Rex's own configuration for each provider.

```
Rex voice loop → LanguageModel (openai strategy)
                      │
                      │  POST /v1/chat/completions
                      ▼
              OpenClaw Gateway  (:18789)
                      │
                      ├─→ Ollama
                      ├─→ OpenAI
                      └─→ Anthropic ...
```

### 8.1 Config

Set these values in `config/rex_config.json` and `.env`:

**`config/rex_config.json`**

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

**`.env`**

```
OPENCLAW_GATEWAY_TOKEN=<your-operator-token>
```

The token is used as the OpenAI `api_key` when constructing the HTTP client.
It is passed as `Authorization: Bearer <token>` on every request.

### 8.2 Session persistence via the `user` field

Every chat completions request Rex sends includes a `user` field derived from
`AppConfig`:

- If `user_id` is set (and is not `"default"`), `user = user_id`.
- Otherwise, `user = active_profile`.

OpenClaw uses this stable string to maintain per-user session state, so
conversation context persists across Rex restarts.

### 8.3 Quick start

1. Install and start the OpenClaw gateway (see OpenClaw docs).
2. Copy `.env.example` to `.env` and set `OPENCLAW_GATEWAY_TOKEN`.
3. Update `config/rex_config.json` with the values in §8.1.
4. Start Rex: `python -m rex` (text mode) or `python rex_loop.py` (voice mode).
5. Rex will send all LLM requests to OpenClaw.  Verify with:

   ```bash
   # Watch OpenClaw logs for incoming /v1/chat/completions requests.
   ```

### 8.4 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on startup | OpenClaw not running | Start OpenClaw gateway first |
| `401 Unauthorized` | Wrong or missing token | Check `OPENCLAW_GATEWAY_TOKEN` in `.env` |
| Slow responses | High gateway timeout | Lower `openclaw.gateway_timeout` in config |
| Rex falls back to echo mode | `openai` package not installed | `pip install openai` |

---

## 9. Open Dependencies (PRD §8.3)

The following OpenClaw API surfaces are not yet confirmed.  Stub code is in
place; fill in the `# TODO` sections once confirmed:

- `agent.py`: `RexAgent.register()` — OpenClaw agent registration call
- `tools/time_tool.py`: `register()` — OpenClaw tool registration call
- `tools/weather_tool.py`: `register()` — OpenClaw tool registration call

See `PRD-openclaw-pivot-for-rex.md` Section 8.3 for the full open-dependency
checklist.
