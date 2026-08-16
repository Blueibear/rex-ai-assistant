# AskRex Mobile/API Threat Model

Status: current-state security baseline for US-088 slice A
Date: 2026-08-14
Target public hostname: `askrex.app` (design target only; not an implicit live deployment)

## 1. Security objective

The mobile gateway lets an authenticated iOS/mobile client use the canonical Rex runtime without exposing desktop-admin authority or creating a weaker parallel assistant. External access must preserve Rex identity, permissions, action verification, user isolation, and truthful result semantics.

The current supported remote topology is the S5-S8 desktop-paired mobile gateway on a TLS-protected LAN endpoint. A future `askrex.app` path is a separate public deployment profile and must remain gated until its reverse-proxy/tunnel and transport-binding requirements are implemented and verified.

## 2. System and trust boundaries

The public/mobile security boundary is `rex.mobile_api.create_mobile_app()`, a dedicated Flask application that registers only `/mobile/*` routes. It is separate from the Electron GUI, `rex.gui_app`, the computer agent, TTS service, OpenClaw tool server, and other local/developer/admin surfaces.

Mobile chat and voice are transport adapters, not a second intelligence path. They pass validated identity and device provenance into `MobileChatService`, which enters the canonical `Assistant.generate_reply()` / `Assistant.stream_reply()` TurnEngine path under `TurnSource.MOBILE`.
## 3. Protected assets

- Canonical Rex user identities, profiles, permissions, and private per-user state.
- Conversation history, memory, context caches, transcripts, and voice content.
- Mobile access JWTs, opaque refresh tokens, sessions, device grants, and revocation state.
- P-256 device public-key bindings, challenge/nonce state, strong-auth approvals, and audit records.
- Household credential-vault entries, especially `REX_JWT_SECRET` and integration credentials.
- Home Assistant and other action authority, including confirmation and independent verification state.
- Desktop-owned TLS private key, certificate fingerprint, SPKI pins, and paired transport binding.
- Cross-transport idempotency records that prevent duplicate tool/action execution.

## 4. Threat actors and failure assumptions

This model assumes hostile Internet traffic once `askrex.app` exists, not merely trusted LAN clients. Relevant actors include an unauthenticated remote attacker, a malicious browser origin, a stolen bearer/refresh token holder, a compromised or modified mobile client, a local-network attacker, and a malicious or compromised optional external capability provider.

The reverse proxy/tunnel and DNS/edge account are also trusted infrastructure whose compromise can affect availability or transport routing. They are not allowed to become Rex authorization authorities.
## 5. Trust and authorization invariants

1. Server-validated credentials establish the principal. Client `user_id`, role, permission, risk, approval, biometric, capability tags, or decoded JWT display claims never grant authority.
2. Password login is bootstrap-only. Mobile capability access requires an active desktop-approved device/grant binding.
3. Device scopes only restrict authority. Current Rex user permissions are resolved server-side and must also allow the action.
4. Revoked, expired, superseded, cross-user, cross-desktop, malformed, or partially bound grants fail closed; revocation/supersession also invalidates bound sessions and refresh families.
5. High/critical actions require a fresh server-owned strong-auth challenge bound to the exact canonical action and a separate short-lived single-use approval at execution time.
6. A successful authentication/approval check is not action-success evidence. Mutations retain canonical attempted/completed/verified/failed semantics and require independent verification to reach `verified`.
7. New or dynamic tools are unavailable to mobile until an explicit mobile scope mapping and enforcement test exists. Descriptive capability metadata cannot widen authority.
8. OpenClaw remains optional. Its health, absence, or remote metadata can never grant additional mobile authority or bypass Rex policy/verification.
## 6. Threats and required controls

| Threat | Required control / current evidence |
|---|---|
| Public exposure of desktop/admin surfaces | Public routing may target only the dedicated mobile gateway. Electron/admin, `rex.gui_app`, computer-control, secret-management, TTS-service, and OpenClaw tool-server routes are separate trust zones and must not be forwarded by the public mobile hostname. |
| Plaintext/MITM | Current S7 requires in-process TLS for every non-loopback bind and pairs against the advertised HTTPS URL, certificate fingerprint, and SPKI pins. A future `askrex.app` edge must require public HTTPS; no direct plaintext Internet listener is permitted. |
| Credential theft/replay | Access JWTs are short-lived and session-bound; refresh tokens are high-entropy, hash-only at rest, rotating, and family-revocable. Every authenticated request validates current session/user/grant state. |
| Authorization escalation | Authorization is the intersection of the immutable current device grant and live Rex permissions. Client claims and capability tags are non-authoritative. Unknown tools fail closed. |
| Pairing takeover | Initial enrollment is desktop-owned, P-256 signed, nonce/challenge bound, short-lived, single-use, and locally approved; the mobile HTTP API cannot create or approve its own grant. |
| Risky-action replay/substitution | S8 challenges bind exact action hash, server risk, scope, session/user/device/grant/version/desktop identity, nonce, and expiry; resulting approval IDs are distinct, short-lived, and single-use. |
| Duplicate tool execution | HTTP/SSE/WebSocket share server-side `(user_id, message_id)` idempotency before acknowledgement/execution. |
| Cross-user leakage | Canonical user IDs are validated before private access and passed explicitly to the shared Assistant; no mutable process-global user authority is used. |
| Brute force / resource exhaustion | Authentication, refresh, chat, voice, WebSocket, and default routes require rate limits and body/media bounds. The current in-memory limiter is single-process only; public/multi-process deployment requires shared limiter state before being called Internet-ready. |
| Browser drive-by / hostile origin | Mobile CORS is deny-by-default; wildcard `*` is rejected by `MobileApiConfig`. Native mobile clients do not require permissive browser CORS. |
| Proxy/client-IP spoofing | A public reverse proxy may supply forwarded addresses only through an explicit trusted-proxy configuration. The origin must not trust arbitrary `X-Forwarded-For` headers from direct clients. |
| Token/content leakage | Access/refresh tokens never belong in URLs; request logging omits bodies. Passwords, chat text, transcripts, TTS text/audio, signatures, and private action bodies must not enter normal logs or diagnostics. |
| Tunnel/DNS route drift | Public ingress must use an explicit route allowlist to the mobile origin only. A configuration change that would expose another local service is a security change requiring review, not an automatic capability expansion. |
| Optional provider compromise | External/OpenClaw capabilities remain behind Rex's canonical registry, policy, mobile scope mapping, and verification. Current mobile code denies dynamic OpenClaw tools by default. |

## 7. Network deployment states

**Loopback development:** `127.0.0.1` may use HTTP unless local TLS is explicitly requested. It is not remote access.

**Paired LAN:** any non-loopback bind is HTTPS-only with desktop-owned S7 certificate material. Pairing binds the endpoint and pins. This is the current supported remote topology.

**Public `askrex.app`:** not yet a supported direct-server mode. The intended future topology is an authenticated reverse proxy/tunnel terminating public HTTPS and forwarding only `/mobile/*` to a loopback-only mobile gateway origin. The desktop must not open the Flask development server or any admin service directly to the Internet.
Public deployment introduces a different transport identity from S7 LAN pinning. Existing pairing records bind an advertised endpoint/fingerprint/SPKI set owned by the gateway. A tunnel terminating TLS at an external edge must therefore not be described as deployable until the public-origin binding and re-pair/migration behavior is explicitly designed and tested. Slice C/D of US-088 owns that gateway design and deployment gate.

## 8. Current iOS/mobile authority surface

Currently authorized mobile capability scopes are deliberately narrow:

- `chat.send`: HTTP, SSE, and WebSocket chat through canonical Assistant/TurnEngine.
- `voice.use`: authenticated audio upload/STT/canonical Assistant response and protected TTS.
- `home.control`: structured Home Assistant mutations only, requiring live user permission and S8 action-bound strong authentication.
- Reserved future scopes include `home.read`, `tasks.read`, `tasks.write`, and `approvals.respond`; a route is not enabled merely because a scope name exists.

Email, calendar, SMS, shell, filesystem, diagnostics, arbitrary computer control, credential/secret management, dynamic OpenClaw tools, and future unknown tools are not mobile-authorized by default.

## 9. Residual and unverified risk

- Physical iPhone validation of certificate/pin enforcement, SecureStore/device-key behavior, Face ID/passcode orchestration, and real network transitions remains a mobile-repository/hardware gate.
- The Flask development server and in-memory limiter are not a production public serving stack.
- `askrex.app` public ingress, trusted-proxy handling, external transport binding, and shared rate-limit storage are not yet implemented/verified by Slice A.
- Availability of cloud/local LLM, STT, TTS, Home Assistant, or OpenClaw does not weaken authorization; unavailable dependencies must fail truthfully.
## 10. Evidence and verification references

Current security behavior is grounded in:

- `rex/mobile_api/app.py`: dedicated `/mobile/*` application, deny-by-default CORS, limits, TLS ownership, privacy-safe middleware.
- `rex/mobile_api/auth.py`, `authorization.py`, `sessions.py`, and `grants.py`: server-derived identity, live session/grant/permission authority, refresh rotation/revocation.
- `rex/mobile_api/pairing.py`, `device_proof.py`, and `strong_auth.py`: desktop-owned P-256 pairing and exact-action strong authentication.
- `rex/mobile_api/chat.py` and `routes/voice.py`: explicit identity and `TurnSource.MOBILE` into canonical Assistant/TurnEngine for text and voice.
- `rex/mobile_api/action_context.py`: explicit mobile tool allowlisting; dynamic/unknown/OpenClaw tools fail closed.
- `rex/mobile_api/tls.py`: mandatory TLS for non-loopback binds and persistent pairing transport identity.
- `tests/mobile_api/`: authentication, revocation, grants, CORS, rate limits, transport binding, strong-auth, idempotency, chat, voice, and user-isolation coverage.

This threat model is a security baseline, not evidence that `askrex.app` is already deployed. Subsequent US-088 slices must classify legacy/local Flask surfaces, define the public gateway and iOS scope, document the tunnel path, and formalize external-boundary tests before the story can close.