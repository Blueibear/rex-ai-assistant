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

## Security properties

- Challenges expire after 120 seconds and are single-use.
- The one-time code is stored only as a SHA-256 hash bound to its challenge and nonce.
- Only canonical P-256 public keys are accepted.
- Proof verification fails when the desktop, nonce, user, scopes, code, key, or signature differs.
- Challenge creation, approval, denial, device listing, and revocation are not exposed through the mobile HTTP API. They are available only through the local Electron IPC bridge.
- Renderer-facing failures are fixed messages; bridge stderr, filesystem paths, database details, and exception text are not returned.
- Grants are immutable and versioned. Scope changes require a new grant version rather than mutation of an existing row.
- Device revocation revokes its active grants and is audited.

## Current boundary

S5 establishes the pairing authority and persists device/grant state. S6 must enforce those grants on every mobile action request. S7 must enforce the supported encrypted LAN transport and certificate/public-key pinning. Until S6 and S7 are complete, pairing is not a claim that non-loopback production mobile access is fully authorized or transport-hardened.

## Mobile endpoints

- `POST /mobile/pairing/submit`
- `POST /mobile/pairing/status`

There is intentionally no mobile endpoint for creating or approving a challenge.
