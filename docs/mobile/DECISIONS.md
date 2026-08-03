# Resolved Mobile Gateway Decisions

These decisions are already resolved for issue #323 and should not be reopened during implementation unless a concrete repository constraint proves one impossible.

1. Reuse Flask and the existing Rex runtime; do not introduce a parallel API/assistant stack.
2. Reuse `data/users.db`, bcrypt users, canonical profiles, and `rex.permissions`.
3. Add short-lived access JWTs plus opaque, hashed, rotating refresh tokens and per-device sessions.
4. Validate issuer, audience, time claims, session, user status, and canonical identity on every authenticated request.
5. Resolve current permissions server-side; client claims are display-only.
6. Restore mobile user state through `GET /mobile/auth/session`.
7. WebSocket authentication uses the first frame with `type: "auth"`; tokens never appear in URLs.
8. All wire fields use `snake_case`.
9. HTTP and WebSocket share `(user_id, message_id)` idempotency before acknowledgement/tool execution.
10. Chat uses canonical `Assistant.generate_reply()` / `stream_reply()` with explicit `active_user_id`.
11. Normal conversation status is `completed`; `verified` requires completion evidence/readback.
12. Voice upload is authenticated, limited to 15 MiB/60 seconds, container-sniffed, and decode-validated.
13. Initial TTS delivery is authenticated JSON base64 plus MIME type; text is never placed in a URL.
14. Unsupported routes return 501 `NOT_IMPLEMENTED` and false capabilities.
15. Live duplex voice remains out of scope and false.
16. Default bind is localhost for development; every non-loopback bind is HTTPS-only with a desktop-owned certificate whose URL, fingerprint, and SPKI pins are signed during pairing and pinned by the mobile client.
17. High/critical mobile actions require S8 strong authentication: a short-lived server challenge signed by the enrolled P-256 device key and bound to the exact canonical action hash, live session/user/device/grant/version/desktop identity, server-owned risk, scope, nonce, and expiry.
18. Successful challenge verification creates a separate short-lived approval ID that is atomically single-use at the actual execution boundary. Client biometric state, a recent timestamp, or approval verification alone never proves the action completed.
19. Free-form Home Assistant transcript routing is desktop-only for mobile requests; mobile mutations require the structured, action-bound S8 route.
20. The desktop/server S8 implementation may merge after repository gates pass. Face ID/passcode orchestration, native signing, and physical-iPhone validation remain separate mobile-repository gates.
