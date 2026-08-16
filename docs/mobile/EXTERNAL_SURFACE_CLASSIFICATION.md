# AskRex External Surface Classification

Status: US-088 slice B evidence classification
Date: 2026-08-14

## Decision

The existing developer Flask/API bridge (`rex.gui_app`) is **unsafe and not approved for external/mobile exposure**. It remains a loopback developer/local surface.

The dedicated `rex.mobile_api.create_mobile_app()` gateway is the **only eligible mobile/public API origin**, because it owns the mobile authentication, pairing/grant, revocation, scope, strong-auth, CORS, rate-limit, idempotency, TLS, and canonical Assistant boundaries. Its current approved remote topology is paired TLS LAN; direct Internet exposure is still prohibited until later US-088 slices complete the public `askrex.app` gateway design and deployment gate.

## Evidence: `rex.gui_app`

`rex.gui_app.main()` binds to `127.0.0.1` and explicitly describes the Flask/API surface as an incomplete standalone experience. Its registered `/api/*` blueprints include auth/setup, logs, users/admin permissions, Home Assistant configuration/control, quick actions, history, integrations, status, and chat.
Its `/api/auth/login` issues the legacy `rex.auth.authenticate()` token: a signed JWT valid for 24 hours with `sub`, username, `iat`, and `exp`. The legacy `/api/auth/logout` is explicitly stateless and does not revoke the token/session server-side. `_require_auth()` validates only that JWT and does not resolve mobile paired-device grants, grant versions, scope intersection, refresh-family revocation, or S8 action-bound strong authentication.

`rex.gui_app` also does not install the mobile gateway's deny-by-default CORS policy, route-specific Flask-Limiter controls, mobile request/body limits, mobile TLS ownership check, or mobile error/session middleware.

Examples of sensitive/local routes on this surface include:

- `/api/admin/permissions/grant` and `/api/admin/permissions/revoke`;
- `/api/ha/save`, which persists the Home Assistant URL/token;
- `/api/devices/<entity_id>/command` and quick-action mutation routes;
- `/api/logs/stream` and `/api/logs/download`;
- `/api/history` and user preference/avatar surfaces;
- setup/registration routes and integration configuration/status routes.

Some local information routes, including `/api/devices` and `/api/ha/states`, do not use the mobile authentication boundary at all. This is acceptable only within the documented local/developer trust model and is further evidence that the app must not be placed behind `askrex.app`.
## Other active HTTP service surfaces

| Surface | Classification for `askrex.app` | Evidence / reason |
|---|---|---|
| `rex.mobile_api` | **Eligible origin, public deployment still gated** | Dedicated `/mobile/*` app; server-derived identity; paired grants; live permissions; refresh/session revocation; strong auth; CORS/rate limits/idempotency/TLS; canonical Assistant/TurnEngine. |
| `rex.gui_app` | **Do not expose** | Developer/local `/api/*` surface with legacy stateless auth and admin/config/log/device routes; not the mobile trust model. |
| `rex_speak_api` | **Do not expose** | Local TTS service bound to `127.0.0.1`; service-level API key rather than per-user mobile grants; also registers HA and shopping surfaces. Mobile TTS already has a narrower authenticated `/mobile/tts/playback` adapter. |
| `rex.computers.agent_server` | **Do not expose** | Executes server-allowlisted OS commands. It defaults to loopback and a static service token; it is desktop/device infrastructure, not an iOS grant/strong-auth surface. |
| `rex.openclaw.tool_server` | **Do not expose** | Invokes Rex/OpenClaw tools using a service API key and local policy adapter. It defaults to loopback and is not scoped through mobile device grants. Dynamic OpenClaw tools are mobile-denied by default. |

Authentication on an internal service is not sufficient to make it a mobile/public API. Each surface has a different principal model, authority scope, and failure impact; reverse-proxying them together would collapse those trust boundaries.

## Required ingress rule

A future `askrex.app` deployment may forward only the dedicated mobile gateway routes required by the published iOS API scope. It must never use a catch-all origin that also reaches `/api/*`, `/rex/tools/*`, `/run`, `/speak`, local GUI content, credential-management endpoints, or other desktop services.