# Home Assistant Integration

**Implementation Status: Beta** — TTS notification channel functional; intent bridge available.

---

## Overview

Rex integrates with [Home Assistant](https://www.home-assistant.io/) in two ways:

1. **Intent bridge** (`rex/ha_bridge.py`) — translates natural-language voice commands
   into Home Assistant service calls (turn on/off lights, set temperature, etc.).
2. **TTS notification channel** (`rex/ha_tts/`) — announces urgent and normal
   notifications over a Home Assistant media player entity.

This document focuses on the TTS notification channel (Cycle 8.3a).  For intent
bridge configuration see the existing inline docs in `rex/ha_bridge.py`.

---

## TTS Notification Channel

### What it does

When a notification is routed to the `ha_tts` channel, Rex calls the Home
Assistant REST API to synthesise and play the notification text on a configured
media player entity (e.g. a smart speaker).

API call made:

```
POST /api/services/{tts_domain}/{tts_service}
Authorization: Bearer <long-lived access token>
Content-Type: application/json

{
  "entity_id": "<media_player entity>",
  "message": "<notification title>: <notification body>"
}
```

### Config keys

All keys live under `notifications.ha_tts` in `config/rex_config.json`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable the channel. **Disabled by default.** |
| `base_url` | string | `null` | HTTPS URL of your HA instance, e.g. `"https://homeassistant.local:8123"`. |
| `token_ref` | string | `null` | `CredentialManager` lookup key for the long-lived access token. |
| `default_entity_id` | string | `null` | Default media player entity, e.g. `"media_player.living_room"`. |
| `default_tts_domain` | string | `"tts"` | HA TTS service domain. |
| `default_tts_service` | string | `"speak"` | HA TTS service within the domain. |
| `timeout_seconds` | float | `10.0` | HTTP request timeout. |
| `allow_http` | bool | `false` | Accept `http://` base URLs. Keep `false` in production. |

#### Example `config/rex_config.json` snippet

```json
{
  "notifications": {
    "ha_tts": {
      "enabled": true,
      "base_url": "https://homeassistant.local:8123",
      "token_ref": "ha:tts_token",
      "default_entity_id": "media_player.living_room",
      "default_tts_domain": "tts",
      "default_tts_service": "speak",
      "timeout_seconds": 10.0
    }
  }
}
```

### Credential setup

The long-lived access token must **never** be stored in `rex_config.json`.
Set it in `.env` or `config/credentials.json` and reference it via `token_ref`.

**Example using a `.env` variable:**

```
# .env
HA_TTS_TOKEN=your-long-lived-access-token-here
```

```json
// config/credentials.json  (or resolved automatically via CredentialManager)
{
  "ha:tts_token": "${HA_TTS_TOKEN}"
}
```

`CredentialManager.get_token("ha:tts_token")` will resolve the token at runtime.

### Routing notifications to HA TTS

Add `"ha_tts"` to a notification's `channel_preferences`:

```python
from rex.notification import NotificationRequest, get_notifier

notifier = get_notifier()
notifier.send(NotificationRequest(
    priority="urgent",
    title="Doorbell",
    body="Someone is at the front door.",
    channel_preferences=["ha_tts", "dashboard"],
))
```

#### Per-notification metadata overrides

| Metadata key | Description |
|---|---|
| `ha_entity_id` | Override the target entity for this notification only. |
| `ha_tts_domain` | Override the TTS service domain. |
| `ha_tts_service` | Override the TTS service name. |

```python
notifier.send(NotificationRequest(
    priority="normal",
    title="Reminder",
    body="Meeting starts in 5 minutes.",
    channel_preferences=["ha_tts"],
    metadata={"ha_entity_id": "media_player.office"},
))
```

---

## CLI test command

Verify the channel is configured and reachable:

```bash
# Show channel status (no network call)
rex ha tts test

# Send a real announcement
rex ha tts test --message "Hello from Rex" --entity-id media_player.living_room
```

**Output when disabled:**
```
HA TTS channel: disabled
  Set notifications.ha_tts.enabled=true in config/rex_config.json to enable.
```

**Output on success:**
```
HA TTS channel: sending test announcement
  base_url : https://homeassistant.local:8123
  entity_id: media_player.living_room
  message  : 'Hello from Rex'
OK: announcement sent successfully.
```

---

## Security posture (SSRF hardening)

The TTS client enforces the same SSRF defences used by the WordPress and
WooCommerce clients:

1. **Scheme check** — only `https://` is accepted unless `allow_http=true`
   (intended for local dev only).
2. **No embedded credentials** — `base_url` must not contain a username or
   password in the URL itself.
3. **DNS-based IP validation** — the hostname is resolved via
   `socket.getaddrinfo`; the following address classes are rejected:
   - Loopback (`127.0.0.0/8`, `::1`)
   - Private (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
   - Link-local (`169.254.0.0/16`, `fe80::/10`)
   - Reserved, multicast, unspecified
4. **Token not logged** — the Bearer token is kept in a private attribute and
   never written to log output or error messages.
5. **Sanitised error messages** — `requests` exceptions are mapped to safe
   strings (`"HA TTS request timed out"`, `"HA TTS HTTP error (status=401)"`,
   etc.) before being surfaced to callers.

---

## Local testing (offline / CI)

Tests for this channel are in `tests/test_ha_tts.py` and run entirely offline:

- `socket.getaddrinfo` is mocked in every test that exercises URL validation.
- `requests.post` is mocked in every test that exercises the HTTP client.
- No real network calls are made.

```bash
pytest -q tests/test_ha_tts.py
```

---

## Graceful degradation

- When the channel is **disabled** (default), notifications sent to `ha_tts`
  are logged and silently skipped — the notification flow continues to other
  channels.
- When the channel is **configured but unreachable**, `HaTtsClient.speak()`
  returns `TtsResult(ok=False, error=<safe message>)`.  The `Notifier`
  converts this to a `RuntimeError`, which triggers the built-in retry logic
  (3 attempts, exponential backoff).  If all retries fail, normal-priority
  notifications fall through to the next channel preference.
- Urgent notifications attempt all channels; a single HA TTS failure does not
  block delivery to other channels.

---

## Known limitations

- The HA TTS channel uses an in-process HTTP call; it does not support streaming
  audio progress or playback confirmation.
- Digest summaries are announced as a single concatenated message; very long
  digests may be truncated by the TTS provider.
- `allow_http=true` bypasses the scheme restriction but not the IP-class check;
  it is intended for local development with a HA instance on the same machine
  behind a self-signed cert, not for production.
