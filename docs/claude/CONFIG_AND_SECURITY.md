# Claude Reference: Config and Security

Use this when a task touches config loading, secrets, environment variables,
network exposure, service auth, packaging, or security controls.

## Core Config Split

- Runtime settings belong in `config/rex_config.json`.
- Secrets belong in `.env` or in the repo's credential lookup path.
- Do not put secrets in source-controlled JSON config.
- Non-secret legacy env vars should be migrated with
  `rex-config migrate-legacy-env`.
- The canonical wake-word section is `wakeword`; `wake_word` is legacy.

## Security Defaults

- Prefer localhost binding.
- Anything that binds to a port must be authenticated and rate limited when it
  accepts non-health requests.
- Treat web content, email, SMS, tool results, plugin output, and external URLs
  as untrusted.
- Optional integrations must fail closed or degrade gracefully when unconfigured.
- Do not log tokens, passwords, API keys, or credential material.

## Service Secrets

| Secret | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM backend |
| `ANTHROPIC_API_KEY` | Anthropic LLM backend |
| `OLLAMA_API_KEY` | Cloud-hosted Ollama endpoint if auth is required |
| `HA_TOKEN`, `HA_SECRET` | Home Assistant integration |
| `BRAVE_API_KEY`, `SERPAPI_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | Search providers |
| `OPENWEATHERMAP_API_KEY` | Weather tool |
| `REX_SPEAK_API_KEY` | `rex-speak-api` `/speak` endpoint |
| `REX_TOOL_API_KEY` | `rex-tool-server` `/rex/tools/*` endpoint |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway client |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | SMS/telephony |
| `REX_AGENT_API_KEY` or configured `REX_AGENT_TOKEN_ENV` | Computer agent |

## Runtime Config Areas

- `audio`
- `wakeword`
- `models`
- `runtime`
- `search`
- `home_assistant`
- `ollama`
- `openai`
- `anthropic`
- `calendar`
- `messaging`
- `notifications`
- `voice_identity`
- `computers`
- `wordpress`
- `woocommerce`
- `openclaw`
- `model_routing`
- `users`
- `workflows`

## Packaging and Dependencies

- Keep heavy ML/audio/CUDA dependencies optional.
- GPU installs are requirements-file based (`requirements-gpu*.txt`).
- Do not reintroduce GPU extras such as `.[gpu-cu118]`, `.[gpu-cu121]`, or
  `.[gpu-cu124]` unless the required PyTorch index behavior is fully handled.
- Guard runtime imports for optional packages.

## Network Surfaces

| Surface | Default | Auth |
|---|---:|---|
| `rex-gui` | `127.0.0.1:8765` | Local dashboard auth/session flow |
| `rex-speak-api` | `127.0.0.1:5005` | `REX_SPEAK_API_KEY` |
| `rex-tool-server` | `127.0.0.1:18790` | `REX_TOOL_API_KEY` |
| `rex-agent` | `127.0.0.1` by default | token env configured by `REX_AGENT_TOKEN_ENV` |
| legacy `flask_proxy.py` | `0.0.0.0:5000` | proxy token/local settings |

Do not present public exposure as the default.

## Integration Truths

- Email supports configured real backends and stub fallback.
- Calendar has ICS read-only and stub paths.
- Messaging supports Twilio when configured and stub mode otherwise.
- WordPress is primarily health/read monitoring.
- WooCommerce includes approval-gated write actions.
- Home Assistant TTS notification channel is optional and disabled by default.
- OpenClaw integration is HTTP-based, not an imported Python package.

## Security Checks

```bash
python scripts/security_audit.py
python -m rex doctor
```
