# AskRex Mobile Device Pairing

AskRex uses a desktop-owned enrollment flow. A username and password can create a normal bootstrap session, but that session is not a paired device and carries no mobile capability grant.

## S5 flow

1. Open **Mobile Pairing** in the Electron desktop app.
2. Select **Generate pairing code**. The desktop creates an eight-digit code and a JSON QR payload that expire after 120 seconds.
3. The mobile client creates a P-256 key pair locally. The private key stays on the device.
4. The mobile client signs the canonical pairing transcript, binding:
   - desktop ID
   - challenge ID and nonce
   - canonical mobile public key
   - approved Rex user ID
   - requested capability scopes
   - one-time code
5. The mobile client submits the public key and signature to `POST /mobile/pairing/submit`.
6. The desktop app displays the pending request, key thumbprint, user, device name, platform, and requested scopes.
7. A local desktop user explicitly approves or denies the request.
8. Approval creates a versioned, expiring grant. The mobile client polls `POST /mobile/pairing/status` with its private poll token.

## S6 device-bound session activation

Password login creates a bootstrap session only. It can call authentication,
pairing, status, and logout surfaces, but it has an empty capability scope set
and cannot call chat, voice, Home Assistant, task, or approval actions.

After desktop approval, the mobile client upgrades the bootstrap session:

1. `POST /mobile/auth/device-challenge` requests a 120-second challenge for the
   approved `device_id` and `grant_id`.
2. The desktop validates the current device/grant/user/desktop/version from
   SQLite. Client-supplied scopes are ignored for authorization.
3. The phone signs the `AskRex-Device-Session-v1` transcript with its paired
   P-256 private key. The transcript binds the desktop, bootstrap session,
   challenge/nonce, device, grant/version, and user.
4. `POST /mobile/auth/activate-device` verifies the proof and atomically creates
   a new paired session/refresh family, marks the challenge used, and revokes
   the bootstrap session and its refresh family.
5. Every authenticated request and refresh resolves the current device and
   grant from SQLite. A revoked, expired, superseded, cross-user, cross-desktop,
   partially bound, or malformed grant invalidates the session.

The session endpoint reports only server-derived binding metadata: `paired`,
`device_id`, `grant_id`, `grant_version`, `desktop_id`, `scopes`, and
`strong_auth_at`. Pairing proof establishes device-key possession, not strong
authentication for a privileged action, so `strong_auth_at` remains `null`
during pairing/session activation. S8 sets it only after the enrolled device
signs a short-lived server challenge bound to one exact action. Mobile
Face ID/passcode gating is an additional client factor and remains a separate
physical-device integration gate; it is never accepted as server authority by
itself.

## Capability enforcement

The centralized server-side route map currently enforces:

- HTTP chat, SSE chat, and WebSocket chat: `chat.send`
- voice upload and TTS playback: `voice.use`
- structured Home Assistant commands: `home.control`, plus S8 one-time strong authentication;
- reserved mappings for Home Assistant reads, tasks, and approval actions use
  `home.read`, `tasks.read`, `tasks.write`, and `approvals.respond` as those
  remaining mobile routes are enabled.

Device scopes are an additional restriction, not a replacement for Rex user
permissions. Home scopes also require the live `ha_control` or `admin` user
permission, and `approvals.respond` requires `admin`. Permission changes are
revalidated while long-running actions stream.

SSE and WebSocket sessions revalidate authorization while streaming, before
emitting each subsequent chunk. Device revocation or grant replacement revokes
bound sessions and refresh tokens, so access stops without waiting for JWT
expiry.

## Security properties

- Challenges expire after 120 seconds and are single-use.
- The one-time code is stored only as a SHA-256 hash bound to its challenge and nonce.
- Only canonical P-256 public keys are accepted.
- Proof verification fails when the desktop, nonce, user, scopes, code, key, advertised HTTPS URL, certificate fingerprint, SPKI pins, or signature differs.
- Challenge creation, approval, denial, device listing, and revocation are not exposed through the mobile HTTP API. They are available only through the local Electron IPC bridge.
- Renderer-facing failures are fixed messages; bridge stderr, filesystem paths, database details, and exception text are not returned.
- Grants are immutable and versioned. Scope changes require a new grant version rather than mutation of an existing row.
- Device revocation revokes its active grants, bound sessions, and refresh
  families atomically and is audited.
- A new grant version supersedes and revokes sessions bound to older versions.
- A paired public key/device identity cannot be reassigned to another Rex user.

## S7 transport: TLS enforcement and certificate pinning

The supported mobile topology is a LAN-paired desktop: the mobile app talks
directly to one desktop-owned gateway on the local network. There is no
default hosted URL and no certificate authority in this topology.

- Any non-loopback bind (`--host 0.0.0.0`, a LAN IP, etc.) always requires
  usable TLS. `rex.mobile_api.tls.resolve_mobile_tls()` provisions (or
  reuses) one long-lived self-signed P-256 certificate under
  `<household_data_dir>/mobile_tls/` and fails closed —
  `MobileTlsConfigurationError`, no socket opened — if that material cannot
  be generated or loaded. `mobile_api.require_tls` cannot weaken this
  boundary; it only opts a **loopback** bind into TLS for local testing.
  Loopback (`127.0.0.1`/`localhost`) stays plain HTTP by default for local
  development and the test suite.
- The SHA-256 fingerprint of that certificate (`desktop_cert_fingerprint`),
  advertised HTTPS URL, and SPKI pins are included in every S5 pairing
  challenge QR and in the approved `/mobile/pairing/status` response.
  Pairing proof transcript v2 signs all three values so the phone explicitly
  acknowledges the same transport identity the desktop records.
- The transport binding is persisted immutably on paired device and grant
  records at approval time, mirroring the immutable `key_thumbprint` binding.
- **Certificate/host mismatch fails closed.** `create_mobile_app()` refuses
  TLS-required injected service containers that lack material, and a TLS-owned
  app rejects plaintext requests with `TLS_REQUIRED`. During
  `POST /mobile/auth/activate-device`, the current gateway-owned URL,
  certificate fingerprint, and SPKI pins must match the immutable device and
  grant bindings. A rotated/reset certificate, changed endpoint, or pre-S7
  unbound legacy device rejects activation with `PAIRING_INVALID`; the device
  must re-pair rather than silently accepting a changed transport.
- **Mobile client contract (implemented in the separate AskRex mobile repo,
  not this one):** production mobile builds must reject `http://`/`ws://`
  URLs for any non-loopback host, and must validate the desktop's presented
  TLS certificate against the pinned fingerprint before any HTTP/SSE/WS
  traffic. This repository cannot enforce that client-side behavior; it only
  provisions and exposes the pin. See "Current boundary" below for what
  remains unverified.

## Current boundary

S5 establishes the pairing authority, S6 enforces device-bound grants on
mobile sessions and action transports, and S7 enforces in-process TLS for
non-loopback binds plus certificate pinning through the S5/S6 authority (see
above). What S7 does **not** cover, because the mobile client lives in a
separate repository not present here:

- The mobile app's own enforcement of "reject insecure non-loopback URLs" and
  "validate the pin before use" — that is a client-side contract this
  repository documents but cannot implement or test.
- Physical LAN/WAN validation with a real phone against a real non-loopback
  bind. All S7 coverage here is automated/local (temp data dirs, fake
  clocks); no physical device or network has exercised this in this cycle.

## Mobile endpoints

- `POST /mobile/pairing/submit`
- `POST /mobile/pairing/status`
- `POST /mobile/auth/device-challenge` (authenticated bootstrap session)
- `POST /mobile/auth/activate-device` (authenticated bootstrap session)

There is intentionally no mobile endpoint for creating or approving the initial
pairing enrollment challenge.
