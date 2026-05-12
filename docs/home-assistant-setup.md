# Home Assistant setup

This guide explains how to connect AskRex Assistant to a local Home Assistant instance.

AskRex uses two pieces of Home Assistant information:

1. the Home Assistant base URL
2. a long-lived access token

Do not commit real tokens to this repository, screenshots, issues, or pull requests.

## Prerequisites

- Home Assistant is running and reachable from the machine running AskRex.
- You can open the Home Assistant web UI in a browser.
- You have a Home Assistant user account that can create long-lived access tokens.
- AskRex is installed and has local config files copied from the examples.

## Common local URLs

Try one of these in a browser first:

```text
http://homeassistant.local:8123
http://homeassistant:8123
http://<LAN-IP-address>:8123
https://<your-home-assistant-hostname>
```

Use the URL that works from the same computer where AskRex runs.

## Create a long-lived access token

In Home Assistant:

1. Open your user profile.
2. Find **Long-lived access tokens**.
3. Create a new token, for example `AskRex local integration`.
4. Copy it once and store it in your local `.env` file.

Treat this token like a password.

## Configure AskRex

AskRex keeps non-secret settings in `config/rex_config.json` and secrets in `.env`.

1. Copy the example config if you have not already done so:

```bash
cp -n config/rex_config.example.json config/rex_config.json
```

2. Set the Home Assistant base URL in `config/rex_config.json`:

```json
{
  "home_assistant": {
    "base_url": "http://homeassistant.local:8123",
    "verify_ssl": true,
    "timeout": 10.0
  }
}
```

If you use a local HTTP URL, keep the scheme as `http://`. If you use HTTPS with a self-signed certificate and connection tests fail because of certificate validation, set `verify_ssl` to `false` only for that trusted local environment.

3. Store the token in `.env`:

```bash
HA_TOKEN=your-long-lived-access-token
```

If `.env` does not exist yet, create it in the repository root. Do not commit it.

## Verify the connection

Start AskRex, open the Electron GUI, and go to the Home Assistant page. Use the connection test from that page.

A successful test means AskRex can reach the configured Home Assistant URL and authenticate with the token.

## Troubleshooting

### `homeassistant.local` does not resolve

Some networks do not resolve mDNS names. Use the Home Assistant machine's LAN IP address instead:

```text
http://192.168.1.50:8123
```

### Connection refused or timeout

Check that Home Assistant is running, port `8123` is open, and both devices are on the same network or VPN.

### Invalid token or unauthorized

Create a new long-lived access token, update `HA_TOKEN` in `.env`, and restart AskRex.

### HTTPS certificate error

For a trusted local self-signed setup, set `home_assistant.verify_ssl` to `false` in `config/rex_config.json`. Keep it `true` when using a valid certificate.

### URL works in one browser but not from AskRex

Make sure AskRex is running on the same machine or network path you used for the browser test. Docker, WSL, VPNs, and guest networks can change what hostnames and LAN IPs are reachable.

## Related files

- `config/rex_config.example.json` — shows the `home_assistant.base_url`, `verify_ssl`, and `timeout` fields.
- `.env` — stores `HA_TOKEN` locally.
- `README.md` — quick start and current Home Assistant status.
