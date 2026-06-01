# Claude Reference: Config and Security

Use this when a task touches config loading, secrets, environment variables,
network exposure, service auth, packaging, or security controls.

## Core Config Split

- Runtime settings belong in `config/rex_config.json`.
- Secrets belong in `.env` only — never in `rex_config.json` or GUI settings files.
- Do not put secrets in source-controlled JSON config.
- Non-secret legacy env vars should be migrated with
  `rex-config migrate-legacy-env`.
- The canonical wake-word section is `wakeword`; `wake_word` is legacy.

## Canonical Secret-Loading Path

**Secrets must come from `.env` only.** Never from `rex_config.json` or
`config/gui_settings.json`.

### Python backend
- All secrets are loaded via `os.getenv("KEY_NAME")` at import/init time.
- `ha_token` → `os.getenv("HA_TOKEN")`; `ha_secret` → `os.getenv("HA_SECRET")`
- `REX_JWT_SECRET` → loaded by `rex/auth.py:get_jwt_secret()`, raises
  `RuntimeError` if unset — there is no fallback.
- `rex_config.json` must never contain token or credential values; it may
  contain credential *reference names* (e.g. `"token_ref": "ha:tts_token"`)
  that the credential lookup layer resolves against the environment.

### Electron GUI (`gui/src/main/index.ts`)
- HA token save path: `saveHomeAssistantCredentials` → `writeEnvKey('HA_TOKEN', token)`
- HA token read path: `readSavedHomeAssistantCredentials` → `readEnvFile().HA_TOKEN`
- Integration settings save path: `rex:setSettings('integrations', ...)` strips
  `haToken` before writing to `gui_settings.json`; if non-empty, writes to `.env`.
- `config/gui_settings.json` stores non-secret UI state only (URLs, providers,
  display preferences). It must not contain credential values.

### Known residual secrets in gui_settings.json (migration backlog)
The following fields are currently written to `gui_settings.json` and should
be migrated to `.env` in a follow-up story:
- `telegramBotToken` → target env var: `TELEGRAM_BOT_TOKEN`
- `emailClientSecret` / `calendarClientSecret` — OAuth app credentials;
  lower risk (app-identity, not user-token), but should move to `.env`
  long-term.

## Migration Path for Existing Users

If you have secrets stored in `rex_config.json` or `gui_settings.json`:

1. **HA token in `rex_config.json` (`home_assistant.token`):**
   Remove the `"token"` key from the `home_assistant` section.
   Add `HA_TOKEN=<your-token>` to `.env`.
   Rex reads the token exclusively from `os.getenv("HA_TOKEN")`.

2. **HA token in `gui_settings.json` (`integrations.haToken`):**
   The GUI no longer reads or writes `haToken` to `gui_settings.json`.
   Ensure `HA_TOKEN` is set in `.env`. Remove the `haToken` field from
   `gui_settings.json` if present.

3. **JWT secret anywhere other than `.env`:**
   Add `REX_JWT_SECRET=<your-secret>` to `.env`.
   Generate a new value with:
   `python -c "import secrets; print(secrets.token_hex(32))"`

4. **Twilio credentials:**
   Move `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`
   to `.env`. Remove from any JSON config file.

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
