# Tool Registry and Tool Execution

AskRex currently has two related tool surfaces:

- Local model-routed tool execution in `rex/openclaw/tool_executor.py`
- HTTP-exposed adapter tools served by `rex-tool-server` from `rex/openclaw/tool_server.py`

## CLI Status

```bash
rex tools
rex tools -v
rex tools --all
```

Example tools registered by the local registry:

| Tool | Description | Credential readiness |
|---|---|---|
| `time_now` | Resolve current local time for a location | None |
| `weather_now` | Fetch current weather through OpenWeatherMap | `OPENWEATHERMAP_API_KEY` |
| `web_search` | Search through `plugins/web_search.py` | Brave or SerpAPI credentials for healthy status |
| `send_email` | Registered email send capability | Email credentials; implementation is limited |
| `home_assistant` | Home Assistant control capability | Home Assistant token |

Status labels:

| Label | Meaning |
|---|---|
| `[READY]` | Enabled, health check passes, and credentials are available |
| `[NO CREDS]` | Required credentials are missing |
| `[UNHEALTHY]` | Health check failed |
| `[DISABLED]` | Tool is registered but disabled |

## Local Tool Execution

Local model-routed tools use this single-line protocol:

```text
TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}
```

The executor returns:

```text
TOOL_RESULT: {"tool":"time_now","args":{"location":"Dallas, TX"},"result":{"local_time":"YYYY-MM-DD HH:MM","date":"YYYY-MM-DD","timezone":"America/Chicago"}}
```

Implementation:

- Parser and executor: `rex/openclaw/tool_executor.py`
- Registry metadata: `rex/openclaw/tool_registry.py`
- Policy checks: `rex/policy_engine.py`
- Audit logging: `rex/audit.py`

Local execution currently supports `time_now`, `weather_now`, and `web_search`. The registry contains additional metadata entries used for readiness and UI/status reporting.

## Risk Classes and Confirmation Gates

Every registered canonical tool declares a risk class on its `ToolSpec`
(`rex/tools/registry.py`, `risk` field) — this registry is the authoritative
list of which tools require confirmation:

| Risk | Meaning | Lifecycle behavior (`rex/tools/execution.py`) |
|---|---|---|
| `safe` | Read-only or low-impact | Dispatches after availability/argument/identity/permission checks. |
| `sensitive` | Destructive or high-impact (e.g. `run_sfc_scan`) | Without prior confirmation the call does **not** execute; it returns outcome `confirmation_required`. Re-invoking with confirmation completes the action. |
| `prohibited` | Never allowed by policy | Denied with a user-visible error; never dispatched. |

The refusal is always surfaced as a structured outcome
(`confirmation_required` / `denied`), never a silent skip. Mutations are
additionally deduplicated by `(user_id, tool, request_id)`: replaying a
completed request returns the recorded result, and reusing a `request_id`
with different arguments is denied. Read-only success reports `completed`;
mutation success reports `verified` only after independent verification.

Home Assistant mutations use the stricter action-bound confirmation tokens
described in [docs/home_assistant.md](home_assistant.md) (signed, single-use,
user/entity/service/parameter-bound, expiring). Coverage:
`tests/test_tool_execution_lifecycle.py`, `tests/test_ha_mutation_service.py`.

## OpenClaw Tool Server

Start the standalone HTTP tool server:

```bash
REX_TOOL_API_KEY=example-key  # pragma: allowlist secret rex-tool-server
```

PowerShell:

```powershell
$env:REX_TOOL_API_KEY="example-key  # pragma: allowlist secret"
rex-tool-server
```

Default bind address:

```text
http://127.0.0.1:18790
```

Health endpoints:

```bash
curl http://127.0.0.1:18790/health/live
curl http://127.0.0.1:18790/health/ready
```

Invoke a tool:

```bash
curl -X POST http://127.0.0.1:18790/rex/tools/time_now \
  -H "Content-Type: application/json" \
  -H "X-API-Key: example-key  # pragma: allowlist secret" \
  -d '{"args":{"location":"Dallas, TX"},"context":{}}'
```

The tool server imports adapter tools from `rex/openclaw/tools/`. Depending on optional dependencies and credentials, the server may expose:

- `time_now`
- `weather_now`
- `send_email`
- `send_sms`
- `calendar_create`
- `home_assistant_call_service`
- `plex_search`, `plex_play`, `plex_pause`, `plex_stop`
- `wordpress_health_check`
- `wc_list_orders`, `wc_list_products`, `wc_set_order_status`, `wc_create_coupon`, `wc_disable_coupon`

Requests require `X-API-Key: <REX_TOOL_API_KEY>` or `Authorization: Bearer <REX_TOOL_API_KEY>`. Rate limits default to 60 requests per 60 seconds and can be adjusted with `REX_TOOL_RATE_LIMIT` and `REX_TOOL_RATE_WINDOW`.

## Adding a Local Tool

1. Register metadata in `rex/openclaw/tool_registry.py`.
2. Add execution logic in `rex/openclaw/tool_executor.py`.
3. Add policy coverage if the tool can mutate state or reach external systems.
4. Add audit-sensitive redaction if arguments may contain secrets.
5. Add tests for parse, policy, credential, and execution paths.

## Adding an HTTP Adapter Tool

1. Add or update a module in `rex/openclaw/tools/`.
2. Register it in `_build_tool_registry()` in `rex/openclaw/tool_server.py`.
3. Ensure the function accepts keyword args plus `context`.
4. Add tests for auth, rate limiting, policy guard, missing dependency behavior, and successful invocation.

## Credential Names

Tool registry credential names are resolved through `rex.credentials.CredentialManager`. Environment-backed examples include:

- `OPENWEATHERMAP_API_KEY` -> `openweathermap`
- `BRAVE_API_KEY` -> `brave`
- `SERPAPI_KEY` -> `serpapi`
- `HA_TOKEN` -> `home_assistant`
- `GITHUB_TOKEN` -> `github`
- `REX_SPEAK_API_KEY` -> `speak`

See [credentials.md](credentials.md) and [policy.md](policy.md).
