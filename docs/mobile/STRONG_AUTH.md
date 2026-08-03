# Mobile Strong Authentication (S8)

Status: desktop/server implementation complete; CI/merge and mobile hardware validation pending.

## Security boundary

Pairing proves that a phone possesses its enrolled P-256 private key. It does
not authorize a privileged operation by itself. High- and critical-risk mobile
actions require a fresh, action-bound strong-authentication exchange.

The server owns:

- action allowlisting and canonicalization;
- risk classification;
- required capability scope and live user permission checks;
- challenge, nonce, expiry, and single-use state;
- current session/device/grant/version/desktop binding;
- proof verification, approval consumption, and audit records.

Client-supplied risk labels, biometric flags, timestamps, scopes, roles, or
permissions never grant authority.

## Protocol

1. The client submits one structured action to
   `POST /mobile/auth/strong-auth/challenge`.
2. The server canonicalizes the exact action, assigns risk and scope, validates
   the current paired session/grant, and issues a 90-second challenge.
3. The phone applies its local user-verification policy (for example Face ID or
   device passcode) and signs the `AskRex-Strong-Auth-v1` transcript with the
   already enrolled non-exportable P-256 key.
4. `POST /mobile/auth/strong-auth/verify` verifies the signature and current
   grant, then returns a separate 45-second approval ID.
5. The exact execution route atomically consumes that approval before invoking
   the action. A changed action, another session/device/user/grant, expiry, or
   replay is denied.

The signed transcript binds:

- challenge ID and nonce;
- canonical action name and SHA-256 action hash;
- server-owned risk level and required scope;
- desktop, session, user, device, grant, and grant version;
- challenge expiry.

A verified proof is not evidence that the requested action succeeded. The
execution response independently reports `verified`, `attempted_unverified`,
`denied`, or `failed` based on the real action/readback lifecycle.

## Current action surface

`home_assistant_call_service` is the first enabled privileged mobile action.
Its action object contains exactly `domain`, `service`, `entity_id`, and
optional `data`. Domain/service are normalized and the entity domain must
match. Unknown, malformed, unsupported, or prohibited Home Assistant actions
fail before a challenge is issued.
The canonical command route is `POST /mobile/home/command`. It consumes the
approval, revalidates the principal, executes through the shared Home Assistant
mutation policy/readback service, and removes internal confirmation tokens from
the response.

Free-form Home Assistant transcript routing and post-processing are desktop
only while a mobile action context is active. Mobile pre-LLM tool dispatch is
also read-only. These restrictions prevent an unstructured transcript from
bypassing the exact action hash.

When structured chat discovers a privileged action without a valid approval,
HTTP and SSE return `STRONG_AUTH_REQUIRED` with:

- the complete signable challenge;
- the exact canonical action to execute;
- the canonical execution path and approval field.

The original chat message is terminally recorded as requiring strong auth. The
client verifies the challenge and executes the returned structured action; it
does not rerun the language model and hope it reconstructs the same action.

## Persistence and privacy

`mobile_strong_auth_challenges` stores protocol binding fields, action hashes,
expiry, verification, approval, and consumption timestamps. It does not store
raw action payloads, signatures, biometric data, passwords, or private keys.

`mobile_strong_auth_audit` stores event type, identities, action hash, risk,
reason, and timestamp. Denials, replay, expiry, verification, and consumption
are auditable without logging the action body or cryptographic proof.
## Validation status

Automated coverage includes canonicalization, risk classification, scope and
permission intersection, wrong-session/device/grant denial, invalid signature,
expiry, replay, changed action, revocation, audit privacy, route validation,
HTTP/SSE challenge delivery, nested authorization layers, and truthful Home
Assistant execution outcomes.

The physical mobile factor is not verified in this repository. The mobile app
must still:

- require Face ID/passcode for high and critical actions and fail closed on
  unavailable, cancel, error, or timeout;
- sign the exact server transcript using its non-exportable paired key;
- verify and consume the returned approval through pinned HTTPS;
- display the exact action/risk before approval;
- handle expiry, revocation, changed action, and replay without fallback.

Those checks require the mobile repository, an Xcode development build, and a
physical iPhone. They must remain labeled pending until exercised successfully.
