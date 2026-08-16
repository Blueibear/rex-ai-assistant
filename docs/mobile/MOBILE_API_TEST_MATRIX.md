# AskRex Mobile API Gateway Test Matrix

Tracking issue: #323  
Date: 2026-07-14

## 1. Test principles

- Tests are deterministic and use injected clocks, ID/token generators, temporary data directories, and fake runtime adapters.
- No test depends on real network services, audio hardware, Home Assistant, cloud APIs, or timing luck.
- Security tests assert that protected code was not reached, not merely that the final response was an error.
- Cross-user tests use conflicting state so accidental global/default behavior is visible.
- Contract tests validate exact required keys, field casing, status values, content type, and close codes.
- Tests never write tracked fixtures or leave repository changes.
- Real LAN/iPhone/audio checks are smoke tests recorded separately from automated results.

## 2. Foundation and configuration

| ID | Case | Expected result |
|---|---|---|
| FND-001 | Import mobile package | No listener, DB mutation, model load, or secret read side effect |
| FND-002 | Create app twice | Independent testable app instances |
| FND-003 | Default config | Binds `127.0.0.1:8765`; mobile API disabled unless explicitly run/enabled |
| FND-004 | Explicit LAN host | `0.0.0.0` accepted with warning |
| FND-005 | Invalid port/TTL/limit | Startup validation fails before serving |
| FND-006 | Missing JWT secret | Auth service fails closed with non-secret configuration error |
| FND-007 | Weak JWT secret | Rejected according to documented minimum |
| FND-008 | CLI flags vs config | Explicit flags win; otherwise typed config; otherwise safe defaults |
| FND-009 | Unknown route | Canonical nested 404 error with request ID |
| FND-010 | Unexpected exception | Generic 500; no stack/path/secret leak |
| FND-011 | Request ID | Header and error field match the request context |
| FND-012 | API version header | Present on every mobile HTTP response |
| FND-013 | Unsupported content type | Rejected before route business logic |
| FND-014 | Oversized JSON body | 413 before full processing |
| FND-015 | CORS default | No wildcard production CORS; native requests unaffected |
| FND-016 | Status response | Minimal non-sensitive health/version data only |
| FND-017 | Capability response | Values reflect actual registered/tested services |
| FND-018 | Scaffold route | 501 `NOT_IMPLEMENTED`, never fake 200 |

## 3. User schema and local setup

| ID | Case | Expected result |
|---|---|---|
| USR-001 | Fresh users DB migration | Existing and new tables/columns created |
| USR-002 | Legacy users DB migration | Existing users/password hashes preserved |
| USR-003 | Migration repeated | No error or duplicate/destructive change |
| USR-004 | Existing user login after migration | Succeeds with original password |
| USR-005 | Disabled user | Cannot log in or refresh |
| USR-006 | Deleted/missing user with live session | Access and refresh fail closed |
| USR-007 | CLI user creation | User, profile, and first-user admin are created canonically |
| USR-008 | CLI duplicate username | Safe nonzero failure; no partial profile/permission changes |
| USR-009 | CLI password prompt | Password not echoed, logged, or accepted through normal argv by default |
| USR-010 | CLI interrupted prompt | No partial user record |
| USR-011 | Username enumeration | Wrong user and wrong password produce indistinguishable external errors |
| USR-012 | Canonical ID validation | Invalid/reserved/traversal ID fails before profile/private access |

## 4. Login and access tokens

| ID | Case | Expected result |
|---|---|---|
| AUTH-001 | Valid login | Token pair, session ID, TTLs, and live user projection returned |
| AUTH-002 | Invalid password | 401 `AUTH_INVALID_CREDENTIALS`; no user-existence leak |
| AUTH-003 | Missing username/password | Canonical 400 error |
| AUTH-004 | Malformed JSON | Canonical 400 error before authentication work |
| AUTH-005 | Device metadata valid | Stored with session; device ID not treated as identity |
| AUTH-006 | Device ID missing/invalid/too long | Rejected or safely generated per final contract |
| AUTH-007 | Password/token logging | Raw values absent from captured logs |
| AUTH-008 | Access JWT claims | Required `iss/aud/sub/sid/jti/iat/nbf/exp` present |
| AUTH-009 | Wrong signature | 401 `AUTH_TOKEN_INVALID` |
| AUTH-010 | Wrong algorithm | Rejected; no algorithm confusion |
| AUTH-011 | Wrong issuer | Rejected |
| AUTH-012 | Wrong audience | Rejected |
| AUTH-013 | Expired token | 401 `AUTH_TOKEN_EXPIRED` |
| AUTH-014 | Future `nbf` | Rejected |
| AUTH-015 | Missing required claim | Rejected |
| AUTH-016 | Invalid `sub` | Fails before user/profile/permission lookup |
| AUTH-017 | Unknown session ID | Rejected |
| AUTH-018 | Session belongs to different `sub` | Rejected and audited without enumeration |
| AUTH-019 | Revoked session with unexpired JWT | 401 `AUTH_SESSION_REVOKED` |
| AUTH-020 | Expired session with unexpired JWT | Rejected |
| AUTH-021 | Client-supplied role/permissions | Ignored; live server values returned/used |
| AUTH-022 | Permission changed after issue | `/session` and authorization reflect current DB state |
| AUTH-023 | Profile changed after issue | `/session` reflects current display profile |
| AUTH-024 | Authentication rate limit | 429 canonical error and Retry-After |

## 5. Refresh rotation and revocation

| ID | Case | Expected result |
|---|---|---|
| REF-001 | Valid refresh | New access and refresh tokens; same session/family |
| REF-002 | Raw refresh storage check | Raw token absent from database |
| REF-003 | Old token after rotation | Reuse rejected and family/session revoked |
| REF-004 | Two concurrent refreshes | Exactly one success; no two valid replacements |
| REF-005 | Expired refresh token | Rejected; no replacement |
| REF-006 | Revoked refresh token | Rejected |
| REF-007 | Refresh token/session mismatch | Rejected |
| REF-008 | Refresh user disabled | Rejected and session revoked/invalidated |
| REF-009 | Random token | Non-enumerating invalid-token error |
| REF-010 | Malformed token | Canonical error; no crash |
| REF-011 | Current logout | Session revoked; access and refresh stop working |
| REF-012 | Repeated current logout | Idempotent success or documented harmless result |
| REF-013 | Logout all | Every session for current user revoked |
| REF-014 | Logout all isolation | Other user's sessions remain active |
| REF-015 | User A token against User B session | Rejected |
| REF-016 | Refresh rate limit | 429 without token leakage |
| REF-017 | Transaction failure | No consumed token without coherent replacement/family state |
| REF-018 | Token-family audit | Reuse event records safe IDs/status only, no raw token |

## 6. Session endpoint and authorization

| ID | Case | Expected result |
|---|---|---|
| SES-001 | Valid current session | Canonical user/session projection |
| SES-002 | Missing Bearer token | 401 canonical error |
| SES-003 | Wrong auth scheme | Rejected |
| SES-004 | Empty Bearer token | Rejected |
| SES-005 | Client sends `user_id` query/body | Ignored or rejected; never changes principal |
| SES-006 | Role projection | Admin maps to owner; non-admin to member for presentation only |
| SES-007 | Live permissions | Current server permissions returned |
| SES-008 | Cross-user session enumeration | IDs/errors do not reveal another user's session existence |

## 7. HTTP chat

| ID | Case | Expected result |
|---|---|---|
| CHAT-001 | Valid non-streaming chat | Canonical Assistant called once with principal user ID |
| CHAT-002 | Missing auth | Rejected before idempotency/Assistant/history/cache |
| CHAT-003 | Client `user_id` field | Rejected or ignored; never passed as identity |
| CHAT-004 | Client role/permissions/risk/approval | Rejected/ignored; never affects execution |
| CHAT-005 | Empty/oversized message | Canonical validation error before Assistant |
| CHAT-006 | Invalid message ID | Rejected before reservation |
| CHAT-007 | Invalid conversation ID | Rejected |
| CHAT-008 | Header/body idempotency mismatch | Conflict; no execution |
| CHAT-009 | Normal response status | `completed`, not `verified` |
| CHAT-010 | Verified tool result | `verified` only with evidence/readback |
| CHAT-011 | Approval required | `needs_confirmation`/structured event; no execution beyond gate |
| CHAT-012 | Permission denied | 403 or structured failed event; no tool call |
| CHAT-013 | Assistant unavailable | `BACKEND_UNAVAILABLE`; no mock response |
| CHAT-014 | Two users conflicting history | Each response sees only its user's state |
| CHAT-015 | Two users same prompt/cache | Cache remains partitioned |
| CHAT-016 | Tool context identity | Same principal ID reaches Assistant, tools, credentials, and history |
| CHAT-017 | Error logging | Message body/private response absent from logs |

## 8. HTTP/SSE streaming

| ID | Case | Expected result |
|---|---|---|
| SSE-001 | Content type | `text/event-stream` with no-cache headers |
| SSE-002 | Token sequence | Valid JSON `data:` events with snake_case fields |
| SSE-003 | Terminal event | Exactly one `message_done` |
| SSE-004 | Malformed internal chunk | Structured error; never plain-text assistant token |
| SSE-005 | Client disconnect | Task cleanup/cancellation; idempotency state remains coherent |
| SSE-006 | Assistant error midstream | Structured error and terminal state; no fake success |
| SSE-007 | Tool events | Emitted from real action boundaries, not parsed prose |
| SSE-008 | Duplicate completed request | Stored terminal result replayed without execution |
| SSE-009 | Duplicate processing request | No second execution; documented in-progress behavior |

## 9. Cross-transport idempotency

| ID | Case | Expected result |
|---|---|---|
| IDP-001 | First reservation | Inserted atomically before ack/execution |
| IDP-002 | Exact HTTP duplicate | One Assistant/tool execution |
| IDP-003 | Exact WS duplicate | One execution |
| IDP-004 | WS then HTTP fallback | One execution and same terminal result |
| IDP-005 | HTTP then WS reconnect | One execution |
| IDP-006 | Concurrent HTTP/WS duplicate | One winner; one non-executing duplicate path |
| IDP-007 | Same ID, different payload | Conflict; no second execution |
| IDP-008 | Same ID, different user | Independent requests; no cross-user result leak |
| IDP-009 | Completed response persistence | Survives service restart for retention window |
| IDP-010 | Error result persistence | Retry policy documented; no accidental duplicate tool call |
| IDP-011 | Pruning | Expired rows removed without deleting active/foreign rows |
| IDP-012 | Request hash | Excludes transport-only fields but includes semantic execution fields |

## 10. WebSocket protocol

| ID | Case | Expected result |
|---|---|---|
| WS-001 | URL contains no token | Connection URL is clean |
| WS-002 | First frame valid `auth` | `auth_ok` with snake_case session/user projection |
| WS-003 | First frame chat/ping | Rejected; close 4401 |
| WS-004 | Auth frame timeout | Close 4408 |
| WS-005 | Invalid/expired/revoked token | `auth_error`, close 4401 |
| WS-006 | Authenticated but forbidden | Close 4403 |
| WS-007 | Connection rate limit | Close 4429 |
| WS-008 | Second auth frame | Rejected; identity cannot be replaced |
| WS-009 | Client identity fields in chat | Rejected/ignored; bound principal unchanged |
| WS-010 | Valid chat | Reservation then ack, then events |
| WS-011 | Ack timing | Sent only after durable reservation |
| WS-012 | Ack casing | `message_id`, `accepted_at` |
| WS-013 | Event casing | All wire fields snake_case |
| WS-014 | Malformed JSON | Structured protocol error; never token text |
| WS-015 | Oversized frame | Rejected without processing |
| WS-016 | Message flood | Rate-limited without process failure |
| WS-017 | Ping/pong | No private content; authenticated state preserved |
| WS-018 | Session revoked while connected | Next privileged message rejected/connection closed |
| WS-019 | Reconnect and replay | Same IDs; no duplicate execution |
| WS-020 | Close 4403 client behavior contract | Client does not loop reconnect automatically |
| WS-021 | Two simultaneous users | Connection principal/state/events remain isolated |
| WS-022 | Connection/message IDs in logs | Safe correlation IDs only; no token/message body |

## 11. Voice upload

| ID | Case | Expected result |
|---|---|---|
| VOI-001 | Missing auth | Rejected before multipart/temp file/STT |
| VOI-002 | Valid supported audio | Transcript and canonical Assistant response |
| VOI-003 | Missing audio part | Canonical 400 |
| VOI-004 | Multiple audio parts | Rejected or deterministic documented selection; prefer rejection |
| VOI-005 | Empty file | `INVALID_MEDIA` |
| VOI-006 | Declared MIME lies | Signature/decode controls; rejected if inconsistent/invalid |
| VOI-007 | Extension lies | Signature/decode controls |
| VOI-008 | Unknown signature | 415 `INVALID_MEDIA` |
| VOI-009 | Malformed supported container | Rejected after decode failure |
| VOI-010 | File >15 MiB | 413 before STT |
| VOI-011 | Duration >60 sec | Rejected before Assistant |
| VOI-012 | Decoder unavailable | Truthful `BACKEND_UNAVAILABLE` |
| VOI-013 | Whisper/model unavailable | Truthful unavailable response; no download/mock |
| VOI-014 | Timeout | Structured retryability; temp cleanup |
| VOI-015 | Cancellation/disconnect | Cleanup and no false success |
| VOI-016 | Two users same audio | Correct separate identity/history/tool context |
| VOI-017 | Temporary file permissions/path | Private temp location; removed in success/failure |
| VOI-018 | Voice content logging | Audio/transcript/private response absent from normal logs |
| VOI-019 | Action status | Never upgraded beyond canonical result evidence |
| VOI-020 | Voice rate limit | 429 before expensive processing |

## 12. TTS

| ID | Case | Expected result |
|---|---|---|
| TTS-001 | Missing auth | Rejected before synthesis |
| TTS-002 | Valid text/default voice | JSON base64, correct MIME, request ID |
| TTS-003 | Empty text | Canonical validation error |
| TTS-004 | Oversized text | Rejected before synthesis |
| TTS-005 | Unknown/unavailable voice | Structured error; no fallback pretending requested voice |
| TTS-006 | User-selected allowed voice | Correct canonical TTS voice path |
| TTS-007 | Engine unavailable | Truthful `BACKEND_UNAVAILABLE` |
| TTS-008 | Synthesis timeout | Structured error and cleanup |
| TTS-009 | Oversized generated audio | Rejected/controlled; no memory blowup |
| TTS-010 | MIME/base64 integrity | Decodes to expected audio format |
| TTS-011 | Text in URL | Never present |
| TTS-012 | Text/audio in logs | Absent from normal logs |
| TTS-013 | Temporary artifact | Removed after encoding/error |
| TTS-014 | TTS rate limit | 429 before synthesis |

## 13. Permissions, policy, approvals, and actions

| ID | Case | Expected result |
|---|---|---|
| POL-001 | Low-risk allowed tool | Executes only with required permission/policy |
| POL-002 | Missing permission | Denied before tool/backend credential lookup |
| POL-003 | Medium/high risk | Existing policy requires approval |
| POL-004 | Client says approved | Ignored |
| POL-005 | Client says biometric passed | Ignored as server proof |
| POL-006 | Approval capability false | No fake approval list/challenge/confirmation |
| POL-007 | Approval event available | Owned by principal and expires correctly |
| POL-008 | User A approval ID used by B | Indistinguishable denied/not-found behavior |
| POL-009 | HA result without readback | At most `completed`, not `verified` |
| POL-010 | HA readback confirms state | May be `verified` with evidence |
| POL-011 | Retry after approval/tool | Idempotency prevents second execution |
| POL-012 | Credential routing | Principal user reaches canonical owned credentials only |

## 14. Capabilities and scaffolds

| ID | Case | Expected result |
|---|---|---|
| CAP-001 | Session 1 capabilities | Auth true; chat/WS/voice/TTS false |
| CAP-002 | Chat code absent | Chat false even if config says enabled |
| CAP-003 | WS dependency absent | WebSocket false |
| CAP-004 | STT dependency/model absent | Voice upload false |
| CAP-005 | TTS engine unavailable | TTS false |
| CAP-006 | HA configured but mobile path absent | HA false |
| CAP-007 | Approval models exist but mobile flow absent | Approvals false |
| CAP-008 | Scaffold request | Authenticated 501 and false capability |
| CAP-009 | Capability response privacy | No paths, tokens, account IDs, usernames, model paths |
| CAP-010 | Live voice | False until real duplex implementation/testing |

## 15. Mobile client contract tests

| ID | Case | Expected result |
|---|---|---|
| MOB-001 | Login request | Includes stable device object; no user/role/permission authority |
| MOB-002 | Login response parsing | Rejects malformed/missing token pair/user/session fields |
| MOB-003 | Nested error parsing | Reads code/message/retryable/request_id |
| MOB-004 | Restore session | Uses `/mobile/auth/session`, not decoded claim authority |
| MOB-005 | Device ID | Generated with secure random API and persisted securely |
| MOB-006 | WS URL | Contains no token |
| MOB-007 | First WS frame | Exact `auth` shape |
| MOB-008 | Auth state | Not connected until `auth_ok` |
| MOB-009 | Wire casing | Snake_case across incoming/outgoing types |
| MOB-010 | Message IDs | Cryptographic UUIDs, stable through replay/fallback |
| MOB-011 | Pending replay | Exact original payload; bounded queue |
| MOB-012 | 4401 | Refresh/re-auth path |
| MOB-013 | 4403 | No infinite reconnect |
| MOB-014 | Invalid server frame | Error state; never rendered as assistant text |
| MOB-015 | HTTP fallback | Bearer auth and same idempotency/message ID |
| MOB-016 | Network failure | Failed state, not silent mock success |
| MOB-017 | Normal chat status | Completed, not verified |
| MOB-018 | Voice local validation | UX only; backend remains authority |
| MOB-019 | TTS response | Base64/MIME handled; no text query URL |
| MOB-020 | Demo mode | Explicitly labeled and never automatic production fallback |

## 16. Integration and smoke matrix

| ID | Environment | Procedure | Required evidence |
|---|---|---|---|
| INT-001 | Local test client | status -> login -> session -> refresh -> logout | Commands, statuses, request IDs |
| INT-002 | Local WS client | connect -> auth -> chat -> ack -> done | Frame transcript with secrets redacted |
| INT-003 | Local transports | send same ID over WS and HTTP | One execution/audit result |
| INT-004 | Local two-user | conflicting prompts/state | Isolation results |
| INT-005 | Local audio dependencies available | upload short real sample | Transcript/response/status; no committed audio |
| INT-006 | Local TTS available | request speech and decode payload | MIME/decoding result |
| INT-007 | Windows LAN | bind `0.0.0.0`, private firewall, call from second device | IP/port/result, secrets redacted |
| INT-008 | Physical iPhone | login/restore/chat/reconnect | Screen/run notes and versions |
| INT-009 | Physical iPhone audio | record/upload/play TTS | Permission/container/playback result |
| INT-010 | Physical iPhone revocation | logout-all/remote revoke then retry | Access rejected as expected |
| INT-011 | Public deployment | TLS and WebSocket proxy | Out of initial scope; track separately |

## 17. Repository validation

Backend minimum:

```powershell
python -m rex mobile-api --help
python -m rex mobile-user --help
python -m rex doctor
pytest -q tests/mobile_api
pytest -q
ruff check .
black --check --diff rex/ tests/ bridge/ *.py
python -m compileall -q rex scripts
mypy rex --ignore-missing-imports
detect-secrets scan --baseline .secrets.baseline
```

Mobile minimum:

```powershell
npm ci
npm run lint
npx tsc --noEmit
npx expo-doctor
```

Add real unit/contract/export scripts before claiming those gates pass. After every validation run, confirm the Git working tree contains only intentional changes.

## US-088 External Gateway Policy

`tests/mobile_api/test_external_gateway_policy.py` is the regression gate for the future `askrex.app` boundary. It proves that:

- the dedicated mobile Flask app registers only `/mobile/*` routes and does not expose local `/api/*`, tool, TTS, agent, or GUI paths;
- protected mobile routes reject missing bearer authentication;
- wildcard CORS remains invalid and an exact `https://askrex.app` origin can be allowlisted without reflecting other origins;
- the public-facing limiter returns the canonical 429 envelope and `Retry-After`;
- loopback origin without a secure transport binding cannot create a pairing challenge;
- dynamic/OpenClaw capability metadata cannot manufacture a mobile grant scope;
- the committed public-gateway documentation keeps `askrex.app`, Cloudflare Tunnel, CORS, rate limiting, revocation, and the closed public-ingress gate explicit.

These tests do not declare the public tunnel production-ready. They enforce the fail-closed boundary while `ASKREX_APP_GATEWAY.md` lists the remaining public transport-binding/deployment gates.