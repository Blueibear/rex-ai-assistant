# Credential Management

## Implementation Status: Implemented in current alpha builds

The credential manager is implemented and used by the current integration backends in the repo's alpha state.

| Feature | Status |
|---------|--------|
| Environment variable loading | Implemented |
| Config file loading (`config/credentials.json`) | Implemented |
| Token expiration tracking | Implemented |
| Runtime credential updates | Implemented |
| Refresh handler registration | Implemented |
| Token masking in logs | Implemented |

Rex uses a centralized credential vault to manage API tokens and secrets for various services. This document explains how credentials are loaded, configured, and used throughout the system.

## Overview

The `CredentialManager` provides:
- Secure storage and retrieval of API tokens
- Lazy loading from environment variables and config files
- Token expiration tracking
- Runtime credential updates
- Extensible refresh handlers for OAuth tokens

## Where Credentials Are Loaded From

Credentials are loaded from two sources, in order:

1. **Environment Variables** (loaded first)
   - Prefixed with `REX_` (e.g., `REX_EMAIL_TOKEN`)
   - Standard variable names also supported (e.g., `OPENAI_API_KEY`)

2. **Configuration File** (overrides environment)
   - Location: `config/credentials.json`
   - JSON format with optional metadata

Environment variables are recommended for secrets in production. The config file is useful for local development and testing.

## Environment Variable Mapping

The following services are mapped by default:

| Service Name      | Environment Variable      |
|-------------------|---------------------------|
| `email`           | `REX_EMAIL_TOKEN`         |
| `calendar`        | `REX_CALENDAR_TOKEN`      |
| `home_assistant`  | `REX_HA_TOKEN` or `HA_TOKEN` |
| `brave`           | `REX_BRAVE_API_KEY` or `BRAVE_API_KEY` |
| `openai`          | `OPENAI_API_KEY`          |
| `ollama`          | `OLLAMA_API_KEY`          |
| `serpapi`         | `REX_SERPAPI_API_KEY`     |
| `github`          | `REX_GITHUB_TOKEN`        |
| `speak`           | `REX_SPEAK_API_KEY`       |

## Configuration File Format

Create `config/credentials.json` with the following format:

### Simple Format (token only)

```json
{
  "credentials": {
    "email": "your-email-token-here",
    "calendar": "your-calendar-token-here"
  }
}
```

### Full Format (with metadata)

```json
{
  "credentials": {
    "email": {
      "token": "your-email-token-here",
      "expires_at": "2026-12-31T23:59:59Z",
      "scopes": ["read", "send"]
    },
    "calendar": {
      "token": "your-calendar-token-here",
      "scopes": ["read", "write"]
    }
  }
}
```

### Fields

| Field       | Type          | Description                              |
|-------------|---------------|------------------------------------------|
| `token`     | string        | The secret token value (required)        |
| `expires_at`| ISO 8601 date | When the token expires (optional)        |
| `scopes`    | string[]      | Permission scopes (optional)             |

## Using Credentials in Code

### Getting a Token

```python
from rex.credentials import get_credential_manager

manager = get_credential_manager()

# Get a token (returns None if not found)
token = manager.get_token("email")

# Check if token exists and is valid
if manager.has_token("email"):
    # Token is available and not expired
    pass
```

### Setting Tokens at Runtime

```python
from datetime import datetime, timedelta, timezone
from rex.credentials import get_credential_manager

manager = get_credential_manager()

# Set a simple token
manager.set_token("custom_service", "my-token")

# Set a token with expiration
expires = datetime.now(timezone.utc) + timedelta(hours=1)
manager.set_token(
    "oauth_service",
    "access-token",
    expires_at=expires,
    scopes=["read", "write"]
)
```

### Adding Custom Credential Mappings

```python
from rex.credentials import get_credential_manager

manager = get_credential_manager()

# Map a new service to an environment variable
manager.add_credential_mapping("my_service", "MY_SERVICE_TOKEN")

# Now REX_MY_SERVICE_TOKEN will be loaded for "my_service"
```

## Token Refresh

The credential manager supports token refresh through registered handlers:

```python
from rex.credentials import get_credential_manager

manager = get_credential_manager()

def refresh_oauth_token(current_token: str) -> str:
    # Call your OAuth provider to get a new token
    new_token = call_oauth_refresh(current_token)
    return new_token

# Register the refresh handler
manager.register_refresh_handler("oauth_service", refresh_oauth_token)

# Now expired tokens will be auto-refreshed
token = manager.get_token("oauth_service", auto_refresh=True)
```

If no refresh handler is registered and you call `refresh_token()`, a `CredentialRefreshError` will be raised.

## Security Considerations

### Token Masking

Tokens are automatically masked in logs and debug output:

```python
from rex.credentials import mask_token

# Shows only first and last 4 characters
masked = mask_token("supersecrettoken123")  # "supe...123"
```

The `Credential` class's `__repr__` method automatically masks tokens.

### Best Practices

1. **Never commit secrets to version control**
   - Add `config/credentials.json` to `.gitignore`
   - Use environment variables in production

2. **Use environment variables for production**
   - Set via `.env` file (not committed)
   - Set via deployment configuration
   - Set via secrets manager (AWS Secrets Manager, etc.)

3. **Rotate tokens regularly**
   - Implement refresh handlers for OAuth tokens
   - Set appropriate expiration dates

4. **Limit token scopes**
   - Request only necessary permissions
   - Track scopes in credential metadata

## Credential Model

The `Credential` dataclass holds credential information:

```python
@dataclass
class Credential:
    name: str                          # Service name
    token: str                         # Secret token value
    expires_at: datetime | None = None # Expiration time (UTC)
    scopes: list[str] = []             # Permission scopes
    source: str = "unknown"            # env, config, or runtime
```

### Checking Expiration

```python
credential = manager.get_credential("oauth_service")
if credential and credential.is_expired():
    # Token needs refresh
    manager.refresh_token("oauth_service")
```

## Reloading Credentials

To reload credentials from environment and config:

```python
from rex.credentials import get_credential_manager

manager = get_credential_manager()
manager.reload()  # Reloads from env and config, preserves runtime tokens
```

## Integration with Tool Registry

The credential manager integrates with the tool registry to verify that required credentials are available before tool execution:

```python
from rex.openclaw.tool_registry import get_tool_registry

registry = get_tool_registry()

# Check if all required credentials are available
all_available, missing = registry.check_credentials("send_email")
if not all_available:
    print(f"Missing credentials: {missing}")
```

See [tools.md](tools.md) for more information about the tool registry.

## Troubleshooting

### Token Not Found

1. Check the environment variable is set correctly
2. Verify the variable name matches the expected mapping
3. Check if `config/credentials.json` exists and is valid JSON

### Token Expired

1. Check the `expires_at` field in your configuration
2. Implement and register a refresh handler
3. Manually update the token via `set_token()`

### Configuration File Not Loading

1. Verify the file exists at `config/credentials.json`
2. Check JSON syntax is valid
3. Ensure the file has correct permissions
