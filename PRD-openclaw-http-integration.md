# PRD: OpenClaw HTTP Integration (Phase 8+)

## Introduction

Rex AI Assistant completed Phases 5-7 of its OpenClaw migration, which built bridge/adapter abstractions and retired legacy modules. However, all 35 OpenClaw integration stubs assumed OpenClaw would be a Python package (`import openclaw`). OpenClaw is actually a TypeScript/Node.js gateway that exposes two HTTP APIs:

- `POST /v1/chat/completions` (OpenAI-compatible)
- `POST /tools/invoke` (direct tool execution)

Plus a WebSocket control plane at `ws://127.0.0.1:18789`.

This PRD covers the work to replace the dead Python import stubs with real HTTP integration, create a shared HTTP client, rewire the feature flags, register Rex's tools with OpenClaw, and deliver a seamless experience where Rex acts as the voice-first interface to an OpenClaw-powered backend.

The target branch is a new branch off `master` (PR #216 has been merged).

## Goals

- Rex routes LLM calls through OpenClaw's `/v1/chat/completions` endpoint when configured, gaining access to any model provider OpenClaw supports (Ollama, OpenAI, Anthropic, etc.)
- Rex's tools (time, weather, HA, email, SMS, calendar, Plex, WooCommerce) are callable from any OpenClaw channel (WhatsApp, Telegram, Discord, etc.), not just the voice loop
- Rex's voice loop works as an OpenClaw "front-end": wake word triggers STT, transcript goes to OpenClaw, response comes back through TTS
- All integration is HTTP-based with proper error handling, retries, timeouts, and auth
- Feature flags (`use_openclaw_tools`, `use_openclaw_voice_backend`) actually do something useful
- Zero regressions: standalone Rex (both flags false) works exactly as before
- All CI checks pass (lint, mypy, pytest, commitlint, secrets scan)

## User Stories

### US-001: Create OpenClaw HTTP client module
**Description:** As a developer, I need a shared HTTP client for all OpenClaw API calls so that auth, retries, timeouts, and error handling are centralized in one place.

**Acceptance Criteria:**
- [x] Create `rex/openclaw/http_client.py` with class `OpenClawClient`
- [x] Constructor accepts `base_url: str`, `auth_token: str`, `timeout: int = 30`, `max_retries: int = 3`
- [x] Implements `post(path, json) -> dict` and `get(path, params) -> dict` methods
- [x] Implements `patch(path, json) -> dict` and `delete(path) -> dict` methods
- [x] Uses `requests` library (already a project dependency via Flask)
- [x] Sets `Authorization: Bearer {token}` header on all requests
- [x] Sets `Content-Type: application/json` header
- [x] Retries on 429 (respects `Retry-After` header) and 5xx with exponential backoff
- [x] Raises `OpenClawConnectionError(url, cause)` on connection failure
- [x] Raises `OpenClawAuthError` on 401
- [x] Raises `OpenClawAPIError(status, body)` on 4xx/5xx after retries exhausted
- [x] All three exception classes defined in `rex/openclaw/errors.py` inheriting from `AssistantError`
- [x] Singleton accessor `get_openclaw_client(config: AppConfig) -> OpenClawClient | None` returns None if `openclaw_gateway_url` is empty
- [x] Logging on every request (DEBUG level: method, path, status; WARNING level: retries, errors)
- [x] Unit tests in `tests/test_openclaw_http_client.py` using `responses` or `unittest.mock` to mock HTTP
- [x] Tests cover: success, 401, 429 retry, 5xx retry, connection error, missing config returns None
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-002: Add OpenClaw gateway config fields to AppConfig
**Description:** As a developer, I need config fields for the OpenClaw gateway URL and auth so that the HTTP client can be configured via `rex_config.json` and `.env`.

**Acceptance Criteria:**
- [x] Add to `AppConfig` in `rex/config.py`:
  - `openclaw_gateway_url: str = ""` (empty = disabled)
  - `openclaw_gateway_timeout: int = 30`
  - `openclaw_gateway_max_retries: int = 3`
- [x] Auth token loaded from env: `OPENCLAW_GATEWAY_TOKEN` (added to `_load_config_from_json` via `os.getenv`)
- [x] `openclaw_gateway_url` loaded from `rex_config.json` key `openclaw.gateway_url`
- [x] `openclaw_gateway_timeout` loaded from `openclaw.gateway_timeout` (default 30)
- [x] `openclaw_gateway_max_retries` loaded from `openclaw.gateway_max_retries` (default 3)
- [x] Update `config/rex_config.json` with new openclaw section:
  ```json
  "openclaw": {
    "gateway_url": "http://127.0.0.1:18789",
    "gateway_timeout": 30,
    "gateway_max_retries": 3,
    "use_tools": false,
    "use_voice_backend": false
  }
  ```
- [x] Add `OPENCLAW_GATEWAY_TOKEN=` to `.env.example` with comment
- [x] Do NOT add the actual token to any committed file
- [x] Existing `use_openclaw_tools` and `use_openclaw_voice_backend` fields unchanged
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-003: Wire OpenAI strategy to OpenClaw chat completions
**Description:** As a user, I want Rex to route LLM calls through OpenClaw's `/v1/chat/completions` endpoint so I get access to any model provider configured in OpenClaw.

**Acceptance Criteria:**
- [x] When `llm_provider` is `"openai"` and `openai.base_url` points to the OpenClaw gateway (e.g. `http://127.0.0.1:18789/v1`), Rex sends chat completions to OpenClaw
- [x] The `model` field in requests uses the value from `llm_model` (e.g. `"openclaw:main"`)
- [x] Rex passes conversation history as `messages` array in standard OpenAI format
- [x] Rex sends a stable `user` field derived from `AppConfig.user_id` or `active_profile` so OpenClaw can maintain session state
- [x] Streaming is NOT required for this story (non-streaming request/response)
- [x] Tool calls in responses are handled by existing `OpenAIStrategy` tool-call loop (no changes needed there)
- [x] Add integration test in `tests/test_openclaw_chat_completions.py` that mocks the HTTP endpoint and verifies the request format
- [x] Test verifies `Authorization: Bearer` header is sent
- [x] Test verifies `model` field matches config
- [x] Test verifies `user` field is present
- [x] Document the config in `docs/openclaw-agent-setup.md` under a new "HTTP Integration" section
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-004: Remove dead `import openclaw` stubs from all bridge/adapter modules
**Description:** As a developer, I want to remove the dead Python import pattern from all OpenClaw modules so the codebase stops pretending OpenClaw is a Python package.

**Acceptance Criteria:**
- [x] In every file under `rex/openclaw/` that has `OPENCLAW_AVAILABLE = find_spec("openclaw")`:
  - Remove the `find_spec("openclaw")` check
  - Remove the `import openclaw as _openclaw` block
  - Remove the `OPENCLAW_AVAILABLE` constant
  - Remove any `if not OPENCLAW_AVAILABLE: return None` guards in `register()` methods
- [x] Files to modify (13 total): `agent.py`, `tool_bridge.py`, `voice_bridge.py`, `event_bridge.py`, `browser_bridge.py`, `policy_adapter.py`, `approval_adapter.py`, `memory_adapter.py`, `identity_adapter.py`, and all files in `tools/` (`time_tool.py`, `weather_tool.py`, `email_tool.py`, `sms_tool.py`, `calendar_tool.py`, `ha_tool.py`, `wordpress_tool.py`, `woocommerce_tool.py`, `plex_tool.py`, `business_tool.py`)
- [x] Replace `OPENCLAW_AVAILABLE` guards with a check for `get_openclaw_client()` returning non-None (from US-001)
- [x] All `register()` functions that were pure no-ops should now log `"OpenClaw gateway not configured"` instead of `"openclaw package not installed"`
- [x] No remaining references to `find_spec("openclaw")` anywhere in `rex/openclaw/`
- [x] No remaining `import openclaw` anywhere in `rex/openclaw/`
- [x] All existing tests that mock `OPENCLAW_AVAILABLE` are updated to mock `get_openclaw_client` instead
- [x] `grep -r "find_spec.*openclaw" rex/openclaw/` returns zero results
- [x] `grep -r "import openclaw" rex/openclaw/` returns zero results
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-005: Rewire RexAgent.respond() to use OpenClaw HTTP
**Description:** As a user, I want `RexAgent.respond()` to route through OpenClaw's chat completions API when the gateway is configured, so that voice and text interactions use the same backend as all other OpenClaw channels.

**Acceptance Criteria:**
- [x] `RexAgent.respond(prompt, user_key)` checks if `get_openclaw_client()` returns a client
- [x] If client exists AND `use_openclaw_voice_backend` is True: POST to `/v1/chat/completions` with `model` from config, `messages` array including system prompt + conversation history + user prompt, and `user` set to `user_key`
- [x] If client is None OR flag is False: fall back to existing local `llm.generate()` path (no behavior change)
- [x] MemoryAdapter is still used to build conversation history for the messages array
- [x] System prompt from `build_system_prompt(config)` is prepended as a `system` role message
- [x] Response content extracted from `choices[0].message.content`
- [x] On HTTP error: log the error, fall back to local LLM generation, log that fallback occurred
- [x] Unit test: mock HTTP, verify request payload shape
- [x] Unit test: mock HTTP 500, verify fallback to local LLM
- [x] Unit test: verify standalone mode (no gateway URL) uses local LLM unchanged
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-006: Rewire VoiceBridge.generate_reply() to use OpenClaw HTTP
**Description:** As a user, I want the voice loop to send my transcripts through OpenClaw when `use_openclaw_voice_backend` is True, so "Hey Rex" conversations use the same agent as my WhatsApp/Telegram chats.

**Acceptance Criteria:**
- [x] `VoiceBridge.generate_reply(transcript, voice_mode, **kwargs)` delegates to `self.agent.respond()` (which now uses HTTP per US-005)
- [x] Voice mode flag is passed through (if OpenClaw supports it; otherwise logged and ignored)
- [x] Empty or whitespace transcripts return empty string without making HTTP call
- [x] On any exception from `respond()`: log error, return a spoken error message like `"I had trouble reaching the server. Try again."`
- [x] Existing tests in `test_openclaw_root_voice_loop_text_mode.py` still pass (they mock the bridge)
- [x] Existing tests in `test_openclaw_root_voice_loop_flag.py` still pass
- [x] New test: verify HTTP call is made when gateway is configured
- [x] New test: verify graceful error message on HTTP failure
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-007: Create OpenClaw tool registration via /tools/invoke
**Description:** As a developer, I need a mechanism to register Rex's tools with OpenClaw so they can be invoked from any OpenClaw channel, not just the voice loop.

**Acceptance Criteria:**
- [x] Create `rex/openclaw/tool_server.py` with class `ToolServer`
- [x] `ToolServer` is a lightweight Flask Blueprint that exposes Rex's tools as HTTP endpoints
- [x] Endpoint: `POST /rex/tools/{tool_name}` accepts `{"args": {...}, "context": {...}}` and returns `{"status": "success"|"error", "result": ...}`
- [x] `ToolServer.register_all(app)` registers the Blueprint on the Flask app
- [x] Each Rex tool (time, weather, HA, email, SMS, calendar, Plex, WooCommerce, WordPress) is mapped to a route
- [x] Policy checking via `PolicyAdapter.guard()` runs before each tool execution
- [x] Tool execution uses existing functions from `rex/openclaw/tools/*.py` (the actual handler functions, not the dead `register()` stubs)
- [x] Error responses use the existing `rex.http_errors` envelope format
- [x] Auth required via same `X-API-Key` mechanism as `rex_speak_api.py`
- [x] Rate limiting applied (reuse existing Flask-Limiter setup)
- [x] Unit tests for at least `time_now` and one policy-gated tool (`send_email`)
- [x] Test verifies auth is required
- [x] Test verifies policy denial returns 403
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-008: Create OpenClaw-to-Rex webhook receiver for /tools/invoke
**Description:** As a user, I want OpenClaw to be able to call Rex's tools (like Home Assistant, Plex control, email) from any channel, so when I message "turn off the lights" on Telegram, OpenClaw can invoke Rex's HA tool.

**Acceptance Criteria:**
- [x] `ToolServer` from US-007 is started alongside `rex_speak_api.py` (or as a standalone entrypoint)
- [x] New entry point `rex-tool-server` added to `pyproject.toml` pointing to `rex.openclaw.tool_server:main`
- [x] `main()` starts a Flask app on `127.0.0.1:18790` (configurable via `REX_TOOL_SERVER_PORT`)
- [x] Server registers health check at `/health/live` and `/health/ready`
- [x] Document in `docs/openclaw-agent-setup.md` how to configure OpenClaw to call Rex's tool server:
  - Add Rex tools to OpenClaw's skill/tool config pointing at `http://127.0.0.1:18790/rex/tools/{tool_name}`
- [x] Include example OpenClaw skill YAML/JSON config snippet in docs
- [x] Integration test: start tool server, POST to `/rex/tools/time_now`, verify response
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-009: Rewire ToolBridge.execute_tool() for dual-mode dispatch
**Description:** As a developer, I want `ToolBridge.execute_tool()` to dispatch tool calls either locally or via OpenClaw depending on the feature flag, so the same tool call works in both standalone and integrated modes.

**Acceptance Criteria:**
- [x] `ToolBridge.execute_tool()` checks `config.use_openclaw_tools` and `get_openclaw_client()`
- [x] If both are truthy: POST to OpenClaw's `/tools/invoke` with `{"tool": tool_name, "args": args, "sessionKey": context.get("session_key", "main")}`
- [x] Parse response: `200` with result, `403` raises `PolicyDeniedError`, `404` falls back to local execution, `429` retries per client config
- [x] If flag is False or client is None: execute locally via existing `_execute_tool()` (no behavior change)
- [x] `parse_tool_request()` always runs locally (no HTTP needed for parsing)
- [x] `route_if_tool_request()` uses the updated `execute_tool()` internally
- [x] Unit test: mock HTTP, verify `/tools/invoke` request format
- [x] Unit test: verify 404 falls back to local
- [x] Unit test: verify standalone mode unchanged
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-010: Rewire MemoryAdapter to support OpenClaw session persistence
**Description:** As a user, I want my conversation history to be shared between Rex voice and other OpenClaw channels, so context carries over whether I talk or text.

**Acceptance Criteria:**
- [x] `MemoryAdapter.append_entry()`: when gateway is configured, POST to `/v1/chat/completions` with the `user` field set (OpenClaw manages session state server-side) AND write locally (dual-write for resilience)
- [x] `MemoryAdapter.load_recent()`: always load from local Rex memory (Rex is the source of truth for voice conversations; OpenClaw maintains its own per-channel)
- [x] `MemoryAdapter.trim_history()`: local only (no HTTP needed)
- [x] Remove all `# TODO: replace with OpenClaw storage` comments from the three methods
- [x] The `user` field sent in chat completions (US-005) creates a stable session in OpenClaw automatically (per OpenClaw docs), so explicit session API calls are not needed
- [x] Unit test: verify dual-write on append (local + user field in next chat completion)
- [x] Unit test: verify load_recent reads local only
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-011: Clean up dead register() stubs in all tool modules ✓
**Description:** As a developer, I want to remove the no-op `register()` functions from all tool modules since tool registration now happens via the ToolServer HTTP endpoint (US-007/US-008), not via Python function registration.

**Acceptance Criteria:**
- [x] In each of these files, remove the `register(agent=None)` function entirely:
  - `rex/openclaw/tools/time_tool.py`
  - `rex/openclaw/tools/weather_tool.py`
  - `rex/openclaw/tools/email_tool.py`
  - `rex/openclaw/tools/sms_tool.py`
  - `rex/openclaw/tools/calendar_tool.py`
  - `rex/openclaw/tools/ha_tool.py`
  - `rex/openclaw/tools/wordpress_tool.py`
  - `rex/openclaw/tools/woocommerce_tool.py`
  - `rex/openclaw/tools/plex_tool.py`
  - `rex/openclaw/tools/business_tool.py`
- [x] Keep the actual tool handler functions (e.g. `time_now()`, `weather_now()`, `send_email()`, etc.) unchanged
- [x] Remove `register()` from `ToolBridge` class as well (lines 220-240 in `tool_bridge.py`)
- [x] Remove `register()` from `PolicyAdapter`, `ApprovalAdapter`, `IdentityAdapter`, `EventBridge`, `BrowserBridge`
- [x] Update any imports that reference these removed `register` functions
- [x] `grep -r "def register(" rex/openclaw/` returns zero results (except `tool_registry.py` which has a different `register` for internal tool metadata)
- [x] All existing tests pass (update any that called `register()`)
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-012: Update EventBridge for local-only operation
**Description:** As a developer, I want EventBridge to stop pretending it will delegate to OpenClaw and instead commit to being the local event bus, since OpenClaw uses WebSocket events (not HTTP) and bridging those is a separate concern.

**Acceptance Criteria:**
- [x] `EventBridge` class keeps its current delegation to `rex.openclaw.event_bus.EventBus`
- [x] Remove all TODO comments about replacing with OpenClaw event registration
- [x] Remove the dead `register()` method (already addressed in US-011, but verify)
- [x] Add docstring clarifying: "EventBridge is Rex's internal event bus. OpenClaw event bridging (WebSocket) is out of scope for HTTP integration."
- [x] If future WebSocket integration is desired, add a `# FUTURE: WebSocket bridge to OpenClaw gateway events` comment in the module docstring
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-013: Update BrowserBridge for local-only operation
**Description:** As a developer, I want BrowserBridge to stop pretending it will delegate to OpenClaw and instead commit to local Playwright execution, since OpenClaw has its own browser automation that operates independently.

**Acceptance Criteria:**
- [x] `BrowserBridge` keeps its current delegation to `browser_core.py` (Playwright)
- [x] Remove all TODO comments about replacing with OpenClaw browser registration
- [x] Remove the dead `register()` method
- [x] Add docstring clarifying: "BrowserBridge runs Playwright locally. OpenClaw has its own Chromium instance for web tasks; they operate independently."
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-014: Update IdentityAdapter for local-only operation with session key passthrough
**Description:** As a developer, I want IdentityAdapter to resolve users locally but pass the user key through to OpenClaw via the `user` field in chat completions, rather than making separate HTTP calls for identity.

**Acceptance Criteria:**
- [x] `IdentityAdapter.resolve_user()` continues to use local `resolve_active_user()` (no HTTP)
- [x] `IdentityAdapter.build_session()` continues to build session context locally
- [x] Add a `get_openclaw_user_key() -> str` method that returns a stable string suitable for the `user` field in OpenClaw chat completions (derived from `resolve_user()` output)
- [x] Remove all TODO comments about replacing with OpenClaw identity registration
- [x] Remove the dead `register()` method
- [x] `RexAgent.respond()` (from US-005) uses `IdentityAdapter.get_openclaw_user_key()` for the `user` field
- [x] Unit test for `get_openclaw_user_key()` returning consistent values for same user
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-015: Update ApprovalAdapter for local-only operation
**Description:** As a developer, I want ApprovalAdapter to keep using local file-based approvals since OpenClaw's exec approval system is WebSocket-based and not something we can bridge via simple HTTP calls.

**Acceptance Criteria:**
- [x] All CRUD methods (`create`, `load`, `approve`, `deny`, `list_pending`) continue using local `rex.workflow_runner` functions
- [x] Remove all TODO comments about replacing with OpenClaw approval registration
- [x] Remove the dead `register()` method
- [x] Add docstring: "Approvals are managed locally by Rex. OpenClaw has its own exec approval flow via WebSocket; bridging is out of scope for HTTP integration."
- [x] Typecheck passes
- [x] Ruff and black pass

---

### US-016: Update documentation for HTTP integration architecture
**Description:** As a developer or operator, I need documentation that explains how Rex integrates with OpenClaw via HTTP, replacing the outdated Python SDK assumptions.

**Acceptance Criteria:**
- [x] Rewrite `docs/openclaw-agent-setup.md` to cover:
  - Architecture diagram (text-based): Rex <-> HTTP <-> OpenClaw Gateway <-> Model Providers
  - Config walkthrough: `rex_config.json` openclaw section + `.env` token
  - Quick start: 5 steps to get Rex talking through OpenClaw
  - Tool server setup: how to expose Rex tools to OpenClaw channels
  - Troubleshooting: connection refused, auth errors, timeout tuning
- [x] Update `CLAUDE.md` OpenClaw Migration Status section:
  - Change "Phase 7 (retirement)" to "Phase 8 (HTTP integration)"
  - Remove references to `find_spec("openclaw")` and Python package
  - Document the two feature flags and what they actually control now
  - Document the new config fields
- [x] Update `docs/openclaw-migration-status.md`:
  - Add Phase 8 rows for HTTP client, config, chat completions, tool server
  - Mark Python import stubs as "Removed" status
- [x] Update `README.md` OpenClaw section to mention HTTP integration (1-2 sentences)
- [x] Typecheck passes (no code changes, but verify docs don't reference removed functions)
- [x] Ruff and black pass

---

### US-017: Add streaming support for OpenClaw chat completions
**Description:** As a user, I want Rex to stream responses from OpenClaw so I hear TTS output faster instead of waiting for the full response.

**Acceptance Criteria:**
- [ ] Add `stream: bool = False` parameter to `OpenClawClient.post()` method
- [ ] When `stream=True`, return a generator that yields SSE `data:` lines parsed as JSON
- [ ] `RexAgent.respond()` accepts `stream=False` parameter; when True, yields partial content strings
- [ ] `VoiceBridge.generate_reply()` uses streaming when available: feeds partial sentences to TTS as they arrive
- [ ] Sentence boundary detection: accumulate streamed tokens until `.`, `!`, `?`, or `\n` before sending to TTS
- [ ] Fallback: if streaming fails mid-response, concatenate received content and return as complete response
- [ ] Non-streaming mode is unchanged (default behavior)
- [ ] Unit test: mock SSE stream, verify sentence-boundary chunking
- [ ] Unit test: mock stream interruption, verify fallback
- [ ] Typecheck passes
- [ ] Ruff and black pass

---

### US-018: End-to-end integration test with mock OpenClaw gateway
**Description:** As a developer, I need a comprehensive integration test that simulates the full Rex-to-OpenClaw flow to catch regressions.

**Acceptance Criteria:**
- [ ] Create `tests/test_openclaw_e2e_integration.py`
- [ ] Test fixture: mock HTTP server (using `responses` library or `pytest-httpserver`) that mimics OpenClaw's `/v1/chat/completions` and `/tools/invoke`
- [ ] Test 1: Text mode end-to-end: user prompt -> RexAgent.respond() -> mock OpenClaw -> response
- [ ] Test 2: Tool call end-to-end: user prompt -> OpenClaw returns tool_call -> Rex executes tool locally -> Rex re-calls OpenClaw with result -> final response
- [ ] Test 3: Voice bridge end-to-end: transcript -> VoiceBridge.generate_reply() -> mock OpenClaw -> spoken response string
- [ ] Test 4: Tool server end-to-end: POST to `/rex/tools/time_now` -> response with current time
- [ ] Test 5: Fallback test: mock OpenClaw returns 500 -> Rex falls back to local LLM -> response returned
- [ ] Test 6: Standalone test: no gateway URL configured -> all paths use local backends -> no HTTP calls made
- [ ] All tests run in CI without requiring a real OpenClaw instance
- [ ] Typecheck passes
- [ ] Ruff and black pass

---

### US-019: Update CI workflow for new integration patterns
**Description:** As a developer, I need CI to validate the new HTTP integration code including the tool server and mock gateway tests.

**Acceptance Criteria:**
- [ ] Add `responses` (or `pytest-httpserver`) to `requirements-dev.txt` for HTTP mocking
- [ ] Verify all new test files are picked up by `pytest -q`
- [ ] Verify `mypy rex --ignore-missing-imports` passes with all new modules
- [ ] Verify ruff and black pass on all new files
- [ ] Add a CI step or marker that runs integration tests separately: `pytest -m integration -q`
- [ ] Mark the e2e tests from US-018 with `@pytest.mark.integration`
- [ ] Standard `pytest -q` still runs unit tests without the integration marker
- [ ] Typecheck passes
- [ ] Ruff and black pass

---

### US-020: Fix string-typed config values and config validation
**Description:** As a developer, I need `rex_config.json` to use correct types so Pydantic doesn't silently coerce or fail on startup.

**Acceptance Criteria:**
- [ ] In `config/rex_config.json`, change `llm_temperature` from `"0.6"` (string) to `0.6` (float)
- [ ] Change `llm_top_p` from `"0.9"` (string) to `0.9` (float)
- [ ] Audit all other numeric fields in `rex_config.json` for string-typed values and fix them
- [ ] Add a startup validation log line in `_load_config_from_json()` that warns if any numeric AppConfig field received a string value (coercion warning)
- [ ] Unit test: load config with string-typed numeric, verify warning is logged
- [ ] Unit test: load config with correct types, verify no warning
- [ ] Typecheck passes
- [ ] Ruff and black pass

## Non-Goals

- **WebSocket integration with OpenClaw's control plane.** The WS protocol is complex (269 files in the gateway server) and is not needed for the core LLM + tool routing use case. HTTP covers the critical paths. WebSocket can be a future phase.
- **Rex as an OpenClaw "node" or "channel."** This would require Rex to accept inbound WebSocket connections from OpenClaw's gateway, which is architecturally different from the HTTP client pattern. Future phase.
- **Replacing Rex's local memory with OpenClaw's session storage.** Rex keeps its own conversation memory for voice interactions. OpenClaw maintains its own per-channel sessions. They coexist; we don't merge them.
- **Replacing Rex's local policy engine with OpenClaw's.** Rex's policy engine runs locally for latency reasons (tool calls in the voice loop need sub-second policy decisions). OpenClaw has its own policy system for its own channels.
- **Multi-agent orchestration.** OpenClaw supports multiple agents, but Rex registers as a single agent. Orchestrating between multiple Rex instances or Rex + other agents is out of scope.
- **Browser automation bridging.** OpenClaw and Rex both have Playwright-based browser automation. They operate independently; merging them is not needed.
- **Android/iOS node integration.** OpenClaw has mobile nodes; Rex doesn't integrate with them.

## Technical Considerations

- **OpenClaw's `/v1/chat/completions` is OpenAI-compatible.** Rex's existing `OpenAIStrategy` in `llm_client.py` already speaks this protocol. The simplest path (US-003) requires zero code changes beyond pointing `openai.base_url` at the gateway. The deeper integration (US-005/US-006) adds conversation history and fallback handling.
- **Auth token is operator-level access.** Per OpenClaw docs, the `/v1/chat/completions` endpoint grants "full operator-access." The token must be treated as a secret (`.env` only, never committed).
- **Tool invocation via `/tools/invoke` has a default deny list** (`cron`, `sessions_spawn`, `sessions_send`, `gateway`, `whatsapp_login`). Rex's tools won't conflict with these.
- **Session persistence via the `user` field.** OpenClaw creates a stable session when the `user` string in chat completions is consistent. No explicit session API calls needed.
- **The `requests` library is already available** (transitive dependency via Flask). No new heavy dependency needed for the HTTP client.
- **Rex's tool handler functions are sync.** The tool server (US-007/US-008) can call them directly from Flask routes without async complications.
- **Fallback is critical.** Every HTTP-backed path must have a local fallback so Rex continues working if OpenClaw is down. This is a voice assistant; silence is unacceptable.
