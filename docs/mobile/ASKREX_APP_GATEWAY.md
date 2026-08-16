# `askrex.app` Secure Mobile Gateway Design

Status: US-088 slice C design; public ingress gate is **closed**
Date: 2026-08-14
Target origin: `https://askrex.app`

## 1. Design decision

`askrex.app` may expose only the dedicated AskRex mobile API contract. It must never proxy the Electron/developer Flask API, computer agent, TTS service, OpenClaw tool server, credential surfaces, or arbitrary localhost ports.

The desktop origin remains loopback-only for the public topology. Public HTTPS is terminated by an authenticated reverse-proxy/tunnel layer; no inbound Internet firewall port is opened to Flask.

This design does not make the current Flask development server production-ready. The public gate remains closed until the transport-binding, trusted-proxy/rate-limit, ingress-route, and deployment tests listed below are implemented.
## 2. Required request path

```text
iOS app
  -> HTTPS + normal platform certificate validation for askrex.app
  -> public edge/tunnel
  -> authenticated/private connector
  -> 127.0.0.1:<mobile-port>
  -> rex.mobile_api only
  -> canonical Assistant / TurnEngine / policy / verification
```

The ingress rule must be path-allowlisted to the supported `/mobile/*` contract and return a hard deny for any `/api/*`, `/rex/tools/*`, `/run`, `/speak`, `/ui/*`, or other local-service path. A catch-all localhost reverse proxy is prohibited.

WebSocket upgrades are permitted only for the authenticated mobile chat endpoint and retain first-frame `auth`, session/grant validation, message-rate limits, and cross-transport idempotency.
## 3. Authentication and token requirements

- Access JWTs remain short-lived (15-minute default), fixed-algorithm, issuer/audience validated, and bound to a live server-side mobile session.
- Refresh tokens remain opaque, high-entropy, hash-only at rest, rotate on every use, and revoke their family/session on reuse.
- Every authenticated request re-resolves the active user, paired device, current immutable grant/version, scopes, and live Rex permissions; token/client display claims cannot authorize.
- Device revocation, grant expiry/supersession, logout, disabled users, or refresh reuse must invalidate otherwise-unexpired access.
- Password login remains bootstrap-only. Capability-bearing sessions require the desktop-approved P-256 device/grant activation flow.
- High/critical actions keep S8 exact-action strong authentication and single-use approval consumption at the execution boundary.

`askrex.app` does not introduce a separate cloud user database, permission model, API key, or assistant identity. Canonical desktop Rex state remains authoritative.
## 4. HTTPS and transport binding

Current S7 LAN pairing binds a desktop-owned HTTPS URL, leaf-certificate fingerprint, and SPKI pin. `PairingAuthority` fails closed when that binding is absent or incomplete, and device-session activation rechecks the stored binding.

A public tunnel presents the edge provider's certificate to iOS rather than Rex's local S7 certificate. Therefore **the existing S7 pin record must not be reused or bypassed for `askrex.app`**.

Before public pairing is enabled, Rex needs a versioned public transport-binding contract that:

1. binds the exact canonical server URL `https://askrex.app` into pairing/device/grant/session activation;
2. uses normal iOS WebPKI hostname/certificate validation at the public edge;
3. identifies the binding mode separately from the current self-signed LAN fingerprint/SPKI mode;
4. fails closed if the configured public hostname/mode changes and requires explicit re-pair/migration;
5. never accepts an arbitrary client-supplied public URL as authority.

Until that contract is implemented and tested, a loopback/tunnel deployment is documentation-only and must not be advertised as a usable production pairing path.
## 5. Rate limiting, proxy trust, and body limits

All current mobile route limits and JSON/audio bounds remain mandatory. Public/multi-process operation additionally requires shared limiter storage; the current `memory://` limiter is not sufficient for horizontally scaled or multi-worker Internet service.

Client-address derivation must be explicit. Only a configured trusted local connector/reverse proxy may supply forwarded client-address metadata. Direct callers cannot choose their own rate-limit identity by sending `X-Forwarded-For` or equivalent headers.

Authentication endpoints require stricter buckets than ordinary reads/chat. Voice/TTS must be limited before expensive decode/model/synthesis work. WebSocket connection attempts and messages remain bounded independently.

## 6. CORS

Native iOS traffic does not require permissive CORS. The default remains no browser origins. If a future first-party web client is introduced, its exact HTTPS origin must be allowlisted; wildcard `*`, credentials-based broad CORS, reflected arbitrary origins, and HTTP production origins are prohibited.
## 7. iOS API scope

The public iOS contract is allowlist-based. Implemented/currently eligible functions are:

| Capability | Public mobile operation | Authority |
|---|---|---|
| Health/capability discovery | `/mobile/status`, `/mobile/capabilities` | Minimal non-sensitive public response only. |
| Bootstrap/session | login, pairing submission/status, device challenge/activation, refresh, current session, logout/logout-all | Server credentials plus desktop-owned pairing/grant state; no client identity authority. |
| Text chat | authenticated `/mobile/chat`, SSE and WebSocket chat | `chat.send` grant scope, live session/grant, canonical Assistant/TurnEngine. |
| Voice/TTS | authenticated `/mobile/voice/upload`, `/mobile/tts/playback` | `voice.use` scope; voice transcript re-enters canonical Assistant/TurnEngine. |
| Home state | authenticated `/mobile/home/entities` | `home.read` scope plus current `ha_control` or admin permission. |
| Home mutation | `/mobile/home/command` | `home.control` + live permission + exact-action S8 strong authentication + canonical verification. |
| Strong authentication | challenge/verify endpoints | Existing paired session/grant; proof never equals action success. |
Explicitly **not public-authorized by default**:

- Electron/developer `/api/*` routes, setup/admin permission management, logs, history administration, and integration credential configuration;
- arbitrary filesystem/shell/computer commands or desktop agent routes;
- direct TTS-service or OpenClaw tool-server access;
- email, calendar, SMS, secrets/credential management, diagnostics, or arbitrary plugin/OpenClaw capabilities;
- any scaffold merely because a route name exists.

Adding a public mobile capability requires a server-owned scope mapping, current Rex permission mapping where applicable, risk/strong-auth decision, verification semantics, privacy review, and explicit tests. Remote metadata or a mobile app update cannot widen the server allowlist.

## 8. OpenClaw boundary

OpenClaw is an optional capability provider behind Rex. `askrex.app` never proxies the OpenClaw gateway/tool server directly. When OpenClaw is absent or unhealthy, core mobile chat/voice/local Rex functions continue according to their own provider availability, and no mobile authority is broadened to compensate.
## 9. Public-ingress release gate

`askrex.app` remains disabled until all of the following are true:

- a versioned public/WebPKI transport-binding mode is implemented and paired-device migration/re-pair behavior is tested;
- only the dedicated loopback mobile origin is reachable from the tunnel and ingress routing rejects non-mobile/admin paths;
- public HTTPS and WebSocket upgrade behavior are verified end to end;
- trusted-proxy handling is explicit and cannot be spoofed by direct requests;
- rate limiting uses deployment-appropriate shared state or the deployment is provably one process with an equivalent enforced limit;
- CORS remains deny-by-default with any browser origin explicitly enumerated;
- access/refresh/session/device-grant revocation tests pass through the public topology;
- strong-auth and truthful action verification remain intact for remotely reachable mutations;
- logs/diagnostics expose no credentials or private request/response content;
- iOS client contract tests and the required physical-device smoke are recorded truthfully.

Slice D documents a concrete tunnel/reverse-proxy procedure without credentials. Slice E formalizes the authentication-rejection, rate-limit, CORS, and external-route safety tests.