# AskRex Mobile API Gateway Architecture

Status: Implementation architecture for issue #323  
Date: 2026-07-14

## 1. Architecture decision

Add a dedicated Flask application factory under `rex/mobile_api/`. It runs as a separate process from the Electron GUI server but imports and reuses the same canonical Rex services and data stores.

The gateway owns:

- mobile HTTP and WebSocket transport;
- access-token validation;
- rotating refresh sessions;
- request IDs and structured errors;
- rate limits and body limits;
- cross-transport idempotency;
- translation between mobile wire events and canonical Rex runtime results.

The gateway does not own a second assistant, user directory, permission model, memory implementation, policy engine, approval engine, Home Assistant client, STT model, or TTS engine.

## 2. Proposed package layout

```text
rex/mobile_api/
  __init__.py
  app.py                  # Flask application factory and middleware wiring
  cli.py                  # mobile-api and mobile-user command handlers
  config.py               # MobileApiConfig helpers/validation
  auth.py                 # access JWT creation/validation and auth decorator
  sessions.py             # session + refresh-token lifecycle
  users.py                # mobile user/session projection from canonical stores
  errors.py               # mobile error codes using rex.http_errors envelope
  capabilities.py         # truthful feature detection
  idempotency.py          # shared HTTP/WebSocket request reservation/result store
  chat.py                 # Assistant adapter and canonical event production
  websocket.py            # first-frame auth and WS protocol state machine
  voice.py                # upload validation, STT adapter, TTS adapter
  routes/
    __init__.py
    auth.py
    status.py
    chat.py
    voice.py
    scaffolds.py
```

Recommended tests:

```text
tests/mobile_api/
  conftest.py
  test_app.py
  test_auth.py
  test_sessions.py
  test_status_capabilities.py
  test_chat_http.py
  test_chat_stream.py
  test_chat_websocket.py
  test_idempotency.py
  test_voice_upload.py
  test_tts.py
  test_identity_isolation.py
  test_scaffolds.py
  test_cli.py
  test_security_logging.py
```

Exact module names may change to fit existing conventions, but responsibilities must remain separated and testable.

## 3. Process model

### 3.1 Application factory

`create_mobile_app(config=None, services=None)` returns a configured Flask app.

The factory:

1. loads typed configuration;
2. creates neutral database/service handles;
3. installs request IDs and privacy-safe request logging;
4. installs the standard error envelope;
5. applies maximum request sizes;
6. installs deny-by-default CORS;
7. installs route-specific rate limiting;
8. registers mobile blueprints;
9. registers the WebSocket route;
10. exposes dependencies through `app.extensions` or an explicit service container.

Tests inject temporary data directories, fake clocks, fake Assistant/STT/TTS adapters, and deterministic token generators. Production code must not use module-global mutable request identity.

### 3.2 Server

Use the existing Flask stack for HTTP. Add one actively maintained Flask-compatible WebSocket extension, validated on Python 3.11 and Windows before it is pinned. Prefer a small dependency such as Flask-Sock/simple-websocket rather than introducing a separate FastAPI/ASGI runtime.

Development uses the documented CLI server. Public deployment later requires a WebSocket-capable production server or reverse proxy; the development server is not presented as internet-ready.

## 4. Service container

A process-level `MobileApiServices` object should contain neutral, reusable dependencies:

- configuration;
- session store;
- user/permission resolver;
- idempotency store;
- unbound or safely managed canonical `Assistant`;
- STT adapter;
- TTS adapter;
- capability resolver;
- clock and ID/token generators.

Every private operation receives a validated user ID explicitly. Service objects may cache neutral resources, but user-private state must remain keyed by validated user ID or live inside the existing user-scoped components.

## 5. Authentication architecture

### 5.1 Existing user database

Continue using `data/users.db`. Extend schema through idempotent migrations rather than creating a second database.

Existing `users` rows remain valid. Add user-active state only if absent:

```sql
ALTER TABLE users ADD COLUMN disabled_at TEXT NULL;
```

SQLite migrations must inspect `PRAGMA table_info` before adding a column.

### 5.2 Mobile sessions

```sql
CREATE TABLE IF NOT EXISTS mobile_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    device_name TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL DEFAULT '',
    app_version TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT NULL,
    revoke_reason TEXT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_mobile_sessions_user
ON mobile_sessions(user_id, revoked_at);
```

`device_id` is validated for length/format and is not trusted as identity. Multiple sessions may exist for one device after reinstall or explicit re-login; server session IDs remain authoritative.

### 5.3 Refresh tokens

```sql
CREATE TABLE IF NOT EXISTS mobile_refresh_tokens (
    token_hash TEXT PRIMARY KEY,
    family_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT NULL,
    revoked_at TEXT NULL,
    replacement_hash TEXT NULL,
    FOREIGN KEY (session_id) REFERENCES mobile_sessions(session_id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_mobile_refresh_family
ON mobile_refresh_tokens(family_id);
```

Generate at least 256 bits of randomness. Encode with URL-safe base64. Store SHA-256 hashes because the source tokens have high entropy. Never store or log raw values.

Refresh algorithm inside one immediate SQLite transaction:

1. hash presented token;
2. load token row;
3. verify unexpired, unrevoked session and active user;
4. when `consumed_at` is already set, revoke the full family/session and return `AUTH_REFRESH_REUSED`;
5. mark current token consumed;
6. create replacement token/hash in the same family;
7. set `replacement_hash`;
8. issue a new access token and return the raw replacement once;
9. commit atomically.

Concurrent use of the same refresh token must produce exactly one success. The loser triggers reuse handling rather than receiving a second token pair.

### 5.4 Access JWT

Access JWTs are short-lived and include:

- `iss = askrex-assistant`;
- `aud = askrex-mobile`;
- `sub = canonical user ID`;
- `sid = mobile session ID`;
- `jti = unique token ID`;
- `iat`, `nbf`, `exp`.

Validation order:

1. parse Bearer syntax;
2. decode only with the configured algorithm and secret;
3. validate signature, issuer, audience, `nbf`, and expiration;
4. validate required claims and canonical `sub`;
5. load session and require matching `sub`/`sid`;
6. require session active and unexpired;
7. require user exists and is active;
8. resolve current display profile and permissions from server stores;
9. attach a request-scoped principal to Flask `g`.

Do not use client-supplied identity or stale token permission claims for authorization.

### 5.5 Principal

Use a typed request principal:

```python
@dataclass(frozen=True)
class MobilePrincipal:
    user_id: str
    session_id: str
    username: str
    display_name: str
    role: str
    permissions: frozenset[str]
```

`role` is a presentation projection. `permissions` comes from the current database for every authenticated request or from a short-lived invalidatable cache keyed by user ID and permission version.

## 6. HTTP middleware

### 6.1 Request IDs

Reuse `rex.request_logging` but ensure:

- every request has a UUID before authentication;
- every HTTP response includes `X-Request-ID`;
- every mobile error includes `error.request_id`;
- WebSocket connections receive a connection ID and each message gets a request ID;
- bodies and tokens are never logged.

### 6.2 Errors

Extend `rex.http_errors.error_response` or add a narrow compatible helper that supports `retryable`. Preserve the nested error object throughout mobile routes.

Expected exceptions are translated to stable codes. Unexpected exceptions are logged with request ID and return generic text.

### 6.3 Rate limiting

Reuse Flask-Limiter. Apply explicit limits to login, refresh, chat, WebSocket connection attempts/messages, voice, and TTS.

Login keys should combine anonymized remote address and normalized username hash to reduce brute force without revealing account existence. Authenticated route keys may include session/user plus remote address. Production multi-process deployment will require shared limiter storage; in-memory storage is acceptable only for local single-process development and must be documented.

### 6.4 Body limits and content types

- Set a global JSON body limit.
- Apply the 15 MiB audio limit before full processing.
- Reject unsupported content types.
- Parse JSON strictly enough to distinguish malformed JSON from an empty object.
- Validate all external payloads with Pydantic models or equivalent explicit validators.

## 7. Chat architecture

### 7.1 Canonical Assistant adapter

Create a narrow `MobileChatService` that invokes the existing Assistant:

```python
reply = await assistant.generate_reply(
    message,
    active_user_id=principal.user_id,
)
```

Streaming uses:

```python
async for token in assistant.stream_reply(
    message,
    active_user_id=principal.user_id,
):
    ...
```

The adapter must not call `LanguageModel.generate()` directly. It must not mutate a global Assistant identity. It must not accept user ID from the request body.

### 7.2 Structured events

The current Assistant streaming API primarily emits text chunks. Add a narrow event sink/observer at existing action-result boundaries only when needed for:

- tool call;
- tool result;
- approval required;
- terminal status.

Do not parse model text to infer tool events. Do not bypass `ActionDispatcher`, `PolicyEngine`, or result verification.

When a structured event cannot be produced truthfully, omit it and keep the corresponding capability false rather than fabricating an event.

### 7.3 Status mapping

- ordinary completed conversational response: `completed`;
- tool request sent without evidence: `attempted`;
- tool finished but no state readback: `completed`;
- tool finished with trusted evidence/readback: `verified`;
- approval required: `needs_confirmation`;
- exception/denial: `failed` or structured permission/approval error.

## 8. Idempotency architecture

Use a shared SQLite-backed store so HTTP and WebSocket transports coordinate.

```sql
CREATE TABLE IF NOT EXISTS mobile_message_requests (
    user_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    response_json TEXT NULL,
    error_code TEXT NULL,
    PRIMARY KEY (user_id, message_id)
);
```

Reservation algorithm:

1. validate principal and payload;
2. compute a deterministic hash of normalized semantic fields;
3. insert `processing` using a transaction;
4. if row exists with a different hash, return conflict;
5. if row exists completed, return/replay stored terminal response;
6. if row exists processing, do not execute again;
7. acknowledge only after reservation succeeds;
8. persist terminal result before announcing `message_done` when practical.

Idempotency records must not store raw tokens. Retention is configurable and pruned safely.

## 9. WebSocket state machine

Per connection:

```text
OPENED -> AWAITING_AUTH -> AUTHENTICATED -> CLOSED
                       \-> REJECTED
```

Rules:

- Start an authentication timeout immediately.
- Only accept one initial `auth` frame.
- Reject chat/ping/other frames before authentication.
- Validate token/session/user through the same service as HTTP.
- Bind an immutable principal to that connection.
- Never allow a later frame to replace user identity.
- Revalidate session status before each privileged message or at a short bounded interval.
- Apply message size and frequency limits.
- Reserve idempotency before sending `ack`.
- Use structured errors and documented close codes.
- Do not echo raw malformed frames.

A reconnect is a new connection and must authenticate again. Pending client messages retain their original IDs.

## 10. Voice architecture

### 10.1 Upload validation

Validation order:

1. require authenticated principal;
2. enforce request and file size before loading fully where possible;
3. require exactly one audio part;
4. sniff supported container signatures;
5. decode through a temporary file in a private data/temp directory;
6. verify non-empty audio and duration <= configured limit;
7. delete temporary data in `finally`;
8. pass decoded/transcribable audio through the existing STT path;
9. send transcript through canonical Assistant with explicit user ID;
10. optionally synthesize TTS through the existing configured engine.

Do not use filename or declared MIME type as sole proof. Use successful decode as the final media validation.

### 10.2 Optional dependencies

When Whisper/audio/TTS dependencies are absent, return `BACKEND_UNAVAILABLE` with `retryable: false` or an appropriately truthful capability false. Do not auto-download heavy models during a request.

### 10.3 TTS

Initial response uses JSON base64 audio. Enforce text length, available voice IDs, generation timeout, output size, and correct MIME type. Do not write response text or audio to normal logs.

If temporary synthesis artifacts are required, store them in a private per-request path and remove them after encoding.

## 11. Capabilities

Capability calculation is code- and runtime-aware:

- `authentication`: schema and token/session services available;
- `chat`: canonical Assistant available;
- `chat_streaming`: streaming path implemented/tested;
- `websocket_chat`: WebSocket dependency and route active;
- `voice_upload`: decoder/STT adapter available;
- `tts`: selected engine available;
- `home_assistant`: only true when mobile authorization, execution, and verification path is implemented and configured;
- `approvals`: only true when server-authoritative mobile approval challenge/resolution exists;
- scaffolds remain false.

Capabilities must not expose secrets, local file paths, model paths, account IDs, or integration tokens.

## 12. CLI integration

Add a command module following current `rex.commands` conventions. `rex/cli.py` keeps registration and backward-compatible re-exports where required.

Commands:

```text
rex mobile-api [--host HOST] [--port PORT]
rex mobile-user create --username USERNAME
```

CLI precedence:

1. explicit flags;
2. typed `mobile_api` config;
3. safe localhost defaults.

`mobile-user create` prompts securely and reuses canonical user/profile/permission functions.

## 13. Client changes required

The mobile repo must:

- add session and capabilities constants;
- add stable SecureStore device ID;
- send device metadata at login;
- restore user through `/mobile/auth/session`;
- parse nested errors;
- update PR #3 to `auth` and snake_case events;
- use cryptographic UUID generation;
- update PR #5 response field names/base64 contract;
- retain explicit, labeled demo mode separately;
- add unit/integration tests and CI scripts before calling the client integrated.

## 14. Deployment notes

Local development:

- bind localhost by default;
- use `0.0.0.0` only deliberately;
- restrict Windows Firewall to private networks and the configured port;
- set the iPhone app URL to `http://<PC-LAN-IP>:8765`;
- never disable authentication on LAN.

Public deployment:

- TLS termination is mandatory;
- forward WebSocket upgrades;
- configure trusted proxies explicitly;
- use secure shared rate-limit/session infrastructure if horizontally scaled;
- expose only the mobile app, not the Electron GUI/admin server;
- rotate secrets and provide backup/migration procedures.

## 15. Architecture risks

| Risk | Mitigation |
|---|---|
| Two users share process state | explicit immutable principal and `active_user_id` propagation |
| Refresh race | immediate SQLite transaction and one-use token rows |
| Duplicate tool execution | shared cross-transport idempotency reservation |
| Stale permissions in JWT | resolve permissions live server-side |
| Token leakage | no URL tokens/body logging; redaction tests |
| WebSocket library/platform issue | validate dependency on Python 3.11/Windows before pinning |
| Fake verification | preserve canonical result status/evidence only |
| Heavy audio model request startup | capability false and explicit unavailable error |
| Public exposure of GUI routes | separate app factory/process and explicit reverse-proxy routes |
| Mixed mobile contracts | master spec and transport tests in both repos |
