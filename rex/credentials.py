"""Credential vault for secure token handling.

This module provides a centralized credential management system that:
- Loads credentials from environment variables based on a configurable mapping
- Optionally loads overrides from a JSON config file (config/credentials.json)
- Provides methods to get, set, and refresh tokens
- Ensures secrets are never logged in full (masks all but first/last 4 chars)
- Checks token expiry and supports refresh stubs

Credentials are loaded lazily and can be refreshed at runtime.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default environment variable prefix for Rex credentials
ENV_PREFIX = "REX_"

# Default mapping of service names to environment variable names
# Format: service_name -> env_var_name (without prefix)
DEFAULT_CREDENTIAL_MAPPING: dict[str, str] = {
    "email": "EMAIL_TOKEN",
    "calendar": "CALENDAR_TOKEN",
    "home_assistant": "HA_TOKEN",
    "brave": "BRAVE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "serpapi": "SERPAPI_API_KEY",
    "github": "GITHUB_TOKEN",
    "speak": "SPEAK_API_KEY",
}

# Default path for credential config file
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "credentials.json"


def mask_token(token: str | None, *, visible_chars: int = 4) -> str:
    """Mask a token for safe display, showing only first and last N characters.

    Args:
        token: The token to mask. If None or empty, returns "[empty]".
        visible_chars: Number of characters to show at start and end.

    Returns:
        Masked token string, e.g., "abc1...xyz9" or "[empty]".
    """
    if not token:
        return "[empty]"
    if len(token) <= visible_chars * 2:
        return "*" * len(token)
    return f"{token[:visible_chars]}...{token[-visible_chars:]}"


@dataclass
class Credential:
    """A credential containing a token and optional metadata.

    Attributes:
        name: The service name this credential is for.
        token: The secret token value.
        expires_at: Optional expiration datetime (UTC).
        scopes: Optional list of permission scopes.
        source: Where the credential was loaded from (env, config, runtime).
    """

    name: str
    token: str
    expires_at: datetime | None = None
    scopes: list[str] = field(default_factory=list)
    source: str = "unknown"

    def is_expired(self) -> bool:
        """Check if the credential has expired.

        Returns:
            True if expires_at is set and is in the past, False otherwise.
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def __repr__(self) -> str:
        """Safe representation that masks the token."""
        return (
            f"Credential(name={self.name!r}, token={mask_token(self.token)!r}, "
            f"expires_at={self.expires_at!r}, scopes={self.scopes!r}, source={self.source!r})"
        )


class CredentialRefreshError(Exception):
    """Raised when credential refresh fails or is not implemented."""

    def __init__(self, service: str, message: str) -> None:
        self.service = service
        self.message = message
        super().__init__(f"Failed to refresh credential for '{service}': {message}")


class CredentialManager:
    """Central manager for loading and accessing credentials.

    The CredentialManager provides a unified interface for accessing secrets:
    1. First checks environment variables based on a configurable mapping
    2. Then loads overrides from a JSON config file if present
    3. Supports runtime token updates via set_token()
    4. Provides refresh stubs for future OAuth integration

    Example:
        >>> manager = CredentialManager()
        >>> token = manager.get_token("email")
        >>> if token is None:
        ...     print("Email token not configured")
    """

    def __init__(
        self,
        *,
        credential_mapping: dict[str, str] | None = None,
        config_path: Path | str | None = None,
        env_prefix: str = ENV_PREFIX,
        refresh_handlers: dict[str, Callable[[str], str]] | None = None,
    ) -> None:
        """Initialize the credential manager.

        Args:
            credential_mapping: Mapping of service names to env var names.
                Defaults to DEFAULT_CREDENTIAL_MAPPING.
            config_path: Path to JSON config file for overrides.
                Defaults to config/credentials.json.
            env_prefix: Prefix for environment variables (default "REX_").
            refresh_handlers: Optional dict of service_name -> refresh callable.
                Each callable takes the current token and returns a new one.
        """
        self._mapping = credential_mapping or DEFAULT_CREDENTIAL_MAPPING.copy()
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._env_prefix = env_prefix
        self._refresh_handlers = refresh_handlers or {}
        self._credentials: dict[str, Credential] = {}
        self._loaded = False
        self._config_invalid = False

    def _load_from_env(self) -> None:
        """Load credentials from environment variables."""
        for service_name, env_var in self._mapping.items():
            # Try with prefix first, then without
            full_var = f"{self._env_prefix}{env_var}"
            token = os.getenv(full_var)
            if token is None:
                # Try without prefix (for standard vars like OPENAI_API_KEY)
                token = os.getenv(env_var)

            if token:
                self._credentials[service_name] = Credential(
                    name=service_name,
                    token=token,
                    source="env",
                )
                logger.debug("Loaded credential for %s from environment", service_name)

    def _load_from_config(self) -> None:
        """Load credentials from JSON config file if it exists."""
        self._config_invalid = False
        if not self._config_path.exists():
            logger.debug("Credential config file not found at %s", self._config_path)
            return

        try:
            with open(self._config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load credential config: %s", e)
            self._config_invalid = True
            self._credentials.clear()
            return

        if not isinstance(config, dict):
            logger.warning("Invalid credential config format (expected dict)")
            self._config_invalid = True
            self._credentials.clear()
            return

        credentials_section = config.get("credentials", config)
        if not isinstance(credentials_section, dict):
            logger.warning("Invalid credentials section format")
            self._config_invalid = True
            self._credentials.clear()
            return

        for service_name, cred_data in credentials_section.items():
            if isinstance(cred_data, str):
                # Simple format: just the token
                token = cred_data
                expires_at = None
                scopes: list[str] = []
            elif isinstance(cred_data, dict):
                # Full format with metadata
                token = cred_data.get("token")
                if not token:
                    continue
                expires_at_str = cred_data.get("expires_at")
                expires_at = None
                if expires_at_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                    except ValueError:
                        logger.warning("Invalid expires_at format for %s", service_name)
                scopes = cred_data.get("scopes", [])
                if not isinstance(scopes, list):
                    scopes = []
            else:
                continue

            # Config overrides env
            self._credentials[service_name] = Credential(
                name=service_name,
                token=token,
                expires_at=expires_at,
                scopes=scopes,
                source="config",
            )
            logger.debug("Loaded credential for %s from config", service_name)

    def _ensure_loaded(self) -> None:
        """Ensure credentials are loaded (lazy loading)."""
        if not self._loaded:
            self._load_from_env()
            self._load_from_config()
            if self._config_invalid:
                self._credentials.clear()
            self._loaded = True

    def reload(self) -> None:
        """Reload all credentials from environment and config.

        This clears the cache and reloads from all sources.
        Runtime credentials set via set_token() will be preserved.
        """
        runtime_creds = {
            name: cred for name, cred in self._credentials.items() if cred.source == "runtime"
        }
        self._credentials.clear()
        self._loaded = False
        self._ensure_loaded()
        # Restore runtime credentials
        self._credentials.update(runtime_creds)
        logger.info("Reloaded credentials")

    def get_credential(self, service_name: str) -> Credential | None:
        """Get the full credential object for a service.

        Args:
            service_name: Name of the service (e.g., "email", "openai").

        Returns:
            Credential object or None if not found.
        """
        self._ensure_loaded()
        return self._credentials.get(service_name)

    def get_token(self, service_name: str, *, auto_refresh: bool = True) -> str | None:
        """Get the token for a service.

        If the token is expired and auto_refresh is True, attempts to refresh it.

        Args:
            service_name: Name of the service (e.g., "email", "openai").
            auto_refresh: Whether to attempt refresh if token is expired.

        Returns:
            Token string or None if not found or refresh failed.
        """
        credential = self.get_credential(service_name)
        if credential is None:
            return None

        if credential.is_expired() and auto_refresh:
            try:
                new_token = self.refresh_token(service_name)
                return new_token
            except CredentialRefreshError:
                logger.warning("Token for %s is expired and refresh failed", service_name)
                return None

        return credential.token

    def set_token(
        self,
        service_name: str,
        token: str,
        *,
        expires_at: datetime | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        """Set or update a token at runtime.

        Args:
            service_name: Name of the service.
            token: The new token value.
            expires_at: Optional expiration datetime (UTC).
            scopes: Optional list of permission scopes.
        """
        self._ensure_loaded()
        self._credentials[service_name] = Credential(
            name=service_name,
            token=token,
            expires_at=expires_at,
            scopes=scopes or [],
            source="runtime",
        )
        logger.debug("Set credential for %s at runtime", service_name)

    def refresh_token(self, service_name: str) -> str:
        """Refresh a token for a service.

        If a refresh handler is registered for the service, it will be called.
        Otherwise, raises CredentialRefreshError.

        Args:
            service_name: Name of the service to refresh.

        Returns:
            The new token value.

        Raises:
            CredentialRefreshError: If refresh fails or no handler is registered.
        """
        if service_name not in self._refresh_handlers:
            raise CredentialRefreshError(
                service_name,
                "No refresh handler registered. Token refresh not implemented for this service.",
            )

        credential = self.get_credential(service_name)
        current_token = credential.token if credential else ""

        try:
            handler = self._refresh_handlers[service_name]
            new_token = handler(current_token)
        except Exception as e:
            raise CredentialRefreshError(service_name, str(e)) from e

        # Update stored credential
        self.set_token(
            service_name,
            new_token,
            expires_at=credential.expires_at if credential else None,
            scopes=credential.scopes if credential else None,
        )
        logger.info("Refreshed token for %s", service_name)
        return new_token

    def register_refresh_handler(
        self,
        service_name: str,
        handler: Callable[[str], str],
    ) -> None:
        """Register a refresh handler for a service.

        Args:
            service_name: Name of the service.
            handler: Callable that takes current token and returns new token.
        """
        self._refresh_handlers[service_name] = handler
        logger.debug("Registered refresh handler for %s", service_name)

    def list_services(self) -> list[str]:
        """List all services that have credentials configured.

        Returns:
            List of service names with available credentials.
        """
        self._ensure_loaded()
        return list(self._credentials.keys())

    def has_token(self, service_name: str) -> bool:
        """Check if a token is available for a service.

        Args:
            service_name: Name of the service.

        Returns:
            True if token exists and is not expired.
        """
        credential = self.get_credential(service_name)
        if credential is None:
            return False
        return not credential.is_expired()

    def add_credential_mapping(self, service_name: str, env_var: str) -> None:
        """Add a custom credential mapping.

        Args:
            service_name: Name of the service.
            env_var: Environment variable name (without prefix).
        """
        self._mapping[service_name] = env_var
        # If already loaded, check for this new mapping
        if self._loaded:
            token = os.getenv(f"{self._env_prefix}{env_var}") or os.getenv(env_var)
            if token and service_name not in self._credentials:
                self._credentials[service_name] = Credential(
                    name=service_name,
                    token=token,
                    source="env",
                )

    def get_credential_info(self, service_name: str) -> dict[str, Any] | None:
        """Get credential information without exposing the full token.

        Args:
            service_name: Name of the service.

        Returns:
            Dict with credential info (masked token) or None.
        """
        credential = self.get_credential(service_name)
        if credential is None:
            return None
        return {
            "name": credential.name,
            "token_preview": mask_token(credential.token),
            "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
            "scopes": credential.scopes,
            "source": credential.source,
            "is_expired": credential.is_expired(),
        }


# Global credential manager instance
_credential_manager: CredentialManager | None = None


def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance.

    Creates a new instance if one doesn't exist.

    Returns:
        The global CredentialManager instance.
    """
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager


def set_credential_manager(manager: CredentialManager) -> None:
    """Set the global credential manager instance.

    Args:
        manager: The CredentialManager instance to use globally.
    """
    global _credential_manager
    _credential_manager = manager


__all__ = [
    "Credential",
    "CredentialManager",
    "CredentialRefreshError",
    "get_credential_manager",
    "set_credential_manager",
    "mask_token",
    "DEFAULT_CREDENTIAL_MAPPING",
]
