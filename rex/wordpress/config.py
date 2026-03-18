"""Pydantic v2 config models for the ``wordpress.sites[]`` section of rex_config.json.

Example config fragment::

    {
      "wordpress": {
        "sites": [
          {
            "id": "myblog",
            "base_url": "https://example.com",
            "enabled": true,
            "auth_method": "application_password",
            "credential_ref": "wp:myblog",
            "timeout_seconds": 15
          }
        ]
      }
    }

Security notes
--------------
- ``credential_ref`` is a CredentialManager lookup key.  The actual secret
  is **never** stored in config; it must live in ``.env`` or
  ``config/credentials.json``.
- For ``auth_method="application_password"`` and ``auth_method="basic"``,
  the credential string must be in ``"username:password"`` format.
- ``base_url`` must be an http(s) URL.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

SUPPORTED_AUTH_METHODS = frozenset({"none", "application_password", "basic"})


class WordPressSiteConfig(BaseModel):
    """Configuration for a single WordPress site."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique site identifier (e.g. 'myblog')")
    base_url: str = Field(
        ...,
        description=(
            "Base URL of the WordPress site "
            "(e.g. 'https://example.com'). Must be http or https. "
            "Trailing slashes are stripped automatically."
        ),
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Whether this site is active.  Disabled sites are hidden from "
            "normal list output and cannot be targeted by CLI commands."
        ),
    )
    auth_method: str = Field(
        default="none",
        description=(
            "Authentication method: "
            "'none' (public API only), "
            "'application_password' (WP Application Passwords, HTTP Basic Auth), "
            "'basic' (plain HTTP Basic Auth)."
        ),
    )
    credential_ref: str = Field(
        default="",
        description=(
            "CredentialManager lookup key for authentication.  "
            "Ignored when auth_method='none'.  "
            "The credential value must be 'username:password' format.  "
            "Store the value in .env or config/credentials.json — never here."
        ),
    )
    timeout_seconds: int = Field(
        default=15,
        ge=1,
        le=120,
        description="HTTP request timeout in seconds (1–120).",
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"base_url must use http or https scheme, got: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("base_url must include a host (netloc)")
        return v.rstrip("/")

    @field_validator("auth_method")
    @classmethod
    def _validate_auth_method(cls, v: str) -> str:
        if v not in SUPPORTED_AUTH_METHODS:
            raise ValueError(
                f"auth_method must be one of {sorted(SUPPORTED_AUTH_METHODS)}, got: {v!r}"
            )
        return v

    @property
    def needs_credential(self) -> bool:
        """Return True if this auth method requires a credential."""
        return self.auth_method in ("application_password", "basic")


class WordPressConfig(BaseModel):
    """Top-level WordPress configuration holding all site entries."""

    model_config = ConfigDict(extra="forbid")

    sites: list[WordPressSiteConfig] = Field(
        default_factory=list,
        description="List of configured WordPress sites.",
    )

    def get_site(self, site_id: str) -> WordPressSiteConfig | None:
        """Return the config for *site_id*, or ``None`` if not found."""
        for site in self.sites:
            if site.id == site_id:
                return site
        return None

    def list_enabled(self) -> list[WordPressSiteConfig]:
        """Return only enabled sites."""
        return [s for s in self.sites if s.enabled]

    def list_all(self) -> list[WordPressSiteConfig]:
        """Return all sites including disabled ones."""
        return list(self.sites)


def load_wordpress_config(raw_config: dict[str, Any]) -> WordPressConfig:
    """Parse the ``wordpress`` section from the merged runtime config.

    Args:
        raw_config: The full runtime config dict (e.g. from ``load_config()``).

    Returns:
        A :class:`WordPressConfig` model.  If the ``wordpress`` key is absent
        or malformed, an empty config with no sites is returned.
    """
    section = raw_config.get("wordpress")
    if not section or not isinstance(section, dict):
        return WordPressConfig()
    sites = section.get("sites")
    if not sites or not isinstance(sites, list):
        return WordPressConfig()
    return cast(WordPressConfig, WordPressConfig.model_validate({"sites": sites}))


__all__ = [
    "SUPPORTED_AUTH_METHODS",
    "WordPressConfig",
    "WordPressSiteConfig",
    "load_wordpress_config",
]
