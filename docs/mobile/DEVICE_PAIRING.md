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
`strong_auth_at`. Pairing proof establishes device-key possession, not user
strong authentication, so `strong_auth_at` remains `null` until S8 verifies a
challenge-bound biometric/passcode assertion for a specific high-risk action.

## Capability enforcement

The centralized server-side route map currently enforces:

- HTTP chat, SSE chat, and WebSocket chat: `chat.send`
- voice upload and TTS playback: `voice.use`
- reserved mappings for Home Assistant, tasks, and approval actions use
  `home.read`, `home.control`, `tasks.read`, `tasks.write`, and
  `approvals.respond` as those mobile routes are enabled.

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
- Proof verification fails when the desktop, nonce, user, scopes, code, key, or signature differs.
- Challenge creation, approval, denial, device listing, and revocation are not exposed through the mobile HTTP API. They are available only through the local Electron IPC bridge.
- Renderer-facing failures are fixed messages; bridge stderr, filesystem paths, database details, and exception text are not returned.
- Grants are immutable and versioned. Scope changes require a new grant version rather than mutation of an existing row.
- Device revocation revokes its active grants, bound sessions, and refresh
  families atomically and is audited.
- A new grant version supersedes and revokes sessions bound to older versions.
- A paired public key/device identity cannot be reassigned to another Rex user.

## Current boundary

S5 establishes the pairing authority and S6 enforces device-bound grants on
mobile sessions and action transports. S7 must still enforce supported encrypted
LAN transport and certificate/public-key pinning. Until S7 is complete,
non-loopback production mobile access is not transport-hardened.

## Mobile endpoints

- `POST /mobile/pairing/submit`
- `POST /mobile/pairing/status`
- `POST /mobile/auth/device-challenge` (authenticated bootstrap session)
- `POST /mobile/auth/activate-device` (authenticated bootstrap session)

There is intentionally no mobile endpoint for creating or approving the initial
pairing enrollment challenge.
