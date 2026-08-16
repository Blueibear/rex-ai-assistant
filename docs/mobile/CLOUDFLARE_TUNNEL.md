# `askrex.app` Cloudflare Tunnel Reference

Status: US-088 slice D deployment reference; **do not activate production public pairing yet**
Verified against current Cloudflare Tunnel documentation: 2026-08-14

This document shows the intended outbound-tunnel shape without storing a tunnel token, account certificate, API token, UUID credential JSON, or any other secret in the repository.

The US-088 public-ingress gate in `ASKREX_APP_GATEWAY.md` remains authoritative. In particular, current S7 LAN certificate/SPKI pinning cannot simply be replaced by an externally terminated tunnel certificate.

## 1. Preconditions

- `askrex.app` is managed in the chosen DNS/tunnel provider.
- `cloudflared` is installed on the AskRex desktop.
- The dedicated mobile gateway is configured to bind **loopback only** for this topology, for example `127.0.0.1:8765`.
- No Electron/developer API, tool server, TTS API, computer agent, or other localhost service is bound to the tunnel hostname.
## 2. Create a locally managed tunnel

Run these interactively on the desktop; do not paste generated credentials into the repo:

```powershell
cloudflared tunnel login
cloudflared tunnel create askrex-mobile
cloudflared tunnel list
```

Cloudflare writes the locally managed tunnel credential JSON outside this repository (normally under the user's `.cloudflared` directory). Treat that file like a secret. Do not copy it into `config/`, `.env`, documentation, logs, screenshots, or commits.

Create the DNS route only when the public-ingress release gate is intentionally being exercised:

```powershell
cloudflared tunnel route dns askrex-mobile askrex.app
```
## 3. Path-allowlisted configuration

Create a local `cloudflared` configuration **outside the AskRex repository**. Replace placeholders only on the machine that owns the tunnel:

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: "C:/Users/<WINDOWS_USER>/.cloudflared/<TUNNEL_UUID>.json"

ingress:
  - hostname: askrex.app
    path: ^/mobile/.*
    service: http://127.0.0.1:8765
  - service: http_status:404
```

The final catch-all is mandatory. Do not add a second `askrex.app` rule without a path, because that would turn the hostname into a general localhost reverse proxy. Do not map `/api/*`, `/rex/tools/*`, `/run`, `/speak`, `/ui/*`, or another service port.
## 4. Validate routing before run

Cloudflare provides local ingress validation and rule-matching commands. Use both before starting a connector:

```powershell
cloudflared tunnel ingress validate
cloudflared tunnel ingress rule https://askrex.app/mobile/status
cloudflared tunnel ingress rule https://askrex.app/mobile/chat
cloudflared tunnel ingress rule https://askrex.app/api/status
cloudflared tunnel ingress rule https://askrex.app/rex/tools/time_now
```

The two `/mobile/*` examples must match the mobile-origin rule. The `/api/*` and `/rex/tools/*` examples must match the final 404 rule.

Only after the US-088 public transport-binding and other release gates are implemented may the connector be started for a public validation run:

```powershell
cloudflared tunnel run askrex-mobile
```

A remotely managed tunnel is an acceptable equivalent, but its token is a credential and must remain in Cloudflare/service configuration rather than the AskRex repo or command examples.
## 5. Security requirements after the tunnel

The tunnel provides transport/routing, not Rex authorization. The origin must still enforce mobile JWT/session/grant revocation, current Rex permissions, S8 strong authentication, idempotency, payload bounds, CORS, rate limits, and truthful action verification.

For Internet deployment, forwarded client-address handling must trust only the local connector/reverse-proxy hop. Do not accept arbitrary client-supplied forwarding headers. Public/multi-worker service also requires deployment-appropriate shared rate-limit state rather than treating the current in-memory limiter as horizontally safe.

Cloudflare's edge certificate is not the desktop-owned S7 LAN certificate. Do not disable or fake the pairing transport-binding check to make the tunnel work. Public pairing stays off until the versioned WebPKI transport-binding contract in `ASKREX_APP_GATEWAY.md` is implemented and tested.

## 6. Rollback

Stopping `cloudflared` removes the outbound connector without requiring an inbound firewall change. Remove or disable the published DNS/tunnel route when testing ends. Revoking a tunnel credential does not replace Rex-side session/device-grant revocation; use both controls when compromise is suspected.