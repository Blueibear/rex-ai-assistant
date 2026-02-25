"""High-level WordPress service used by the CLI (Cycle 6.1).

This module bridges config + credentials + the HTTP client into a single,
easy-to-use facade for CLI commands.

Usage example::

    svc = get_wordpress_service()
    result = svc.health("myblog")
    print(result)
"""

from __future__ import annotations

import logging
from typing import Any

from rex.wordpress.client import WordPressClient, WPHealthResult
from rex.wordpress.config import WordPressConfig, WordPressSiteConfig, load_wordpress_config

logger = logging.getLogger(__name__)


class WordPressSiteNotFoundError(Exception):
    """Raised when a requested site ID is not in the config."""


class WordPressSiteDisabledError(Exception):
    """Raised when a requested site is disabled."""


class WordPressMissingCredentialError(Exception):
    """Raised when a required credential is not configured."""


class WordPressService:
    """Facade over config, credentials, and the WordPress HTTP client.

    Parameters
    ----------
    wp_config:
        Parsed :class:`~rex.wordpress.config.WordPressConfig`.
    credential_manager:
        A :class:`~rex.credentials.CredentialManager` instance used to
        resolve credentials from ``credential_ref`` values.
    """

    def __init__(
        self,
        wp_config: WordPressConfig,
        credential_manager: Any,
    ) -> None:
        self._config = wp_config
        self._creds = credential_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self, site_id: str) -> WPHealthResult:
        """Run a health check against the specified WordPress site.

        Args:
            site_id: The ``id`` field from config.

        Returns:
            :class:`~rex.wordpress.client.WPHealthResult`.

        Raises:
            :class:`WordPressSiteNotFoundError`: If the site ID is unknown.
            :class:`WordPressSiteDisabledError`: If the site is disabled.
            :class:`WordPressMissingCredentialError`: If a required
                credential is not configured.
        """
        client = self._make_client(site_id)
        return client.health()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_site(self, site_id: str) -> WordPressSiteConfig:
        """Resolve and validate a site config entry.

        Raises:
            :class:`WordPressSiteNotFoundError`: unknown ID.
            :class:`WordPressSiteDisabledError`: site is disabled.
        """
        cfg = self._config.get_site(site_id)
        if cfg is None:
            raise WordPressSiteNotFoundError(
                f"No WordPress site with id {site_id!r} found in config. "
                "Add it to the wordpress.sites[] section of rex_config.json."
            )
        if not cfg.enabled:
            raise WordPressSiteDisabledError(
                f"WordPress site {site_id!r} is disabled. "
                "Set enabled=true in config/rex_config.json to use it."
            )
        return cfg

    def _resolve_auth(self, cfg: WordPressSiteConfig) -> tuple[str, str] | None:
        """Resolve HTTP Basic Auth credentials for *cfg*.

        Returns ``None`` when ``auth_method="none"``.

        Raises:
            :class:`WordPressMissingCredentialError`: If the credential is
                not found or is not in ``"username:password"`` format.
        """
        if not cfg.needs_credential:
            return None

        if not cfg.credential_ref:
            raise WordPressMissingCredentialError(
                f"WordPress site {cfg.id!r} has auth_method={cfg.auth_method!r} "
                "but no credential_ref is set in config."
            )

        raw = self._creds.get_token(cfg.credential_ref)
        if not raw:
            raise WordPressMissingCredentialError(
                f"Credential {cfg.credential_ref!r} for WordPress site {cfg.id!r} "
                "is not configured.  Set it in .env or config/credentials.json."
            )

        if ":" not in raw:
            raise WordPressMissingCredentialError(
                f"Credential {cfg.credential_ref!r} for WordPress site {cfg.id!r} "
                "must be in 'username:password' format."
            )

        username, password = raw.split(":", 1)
        return (username, password)

    def _make_client(self, site_id: str) -> WordPressClient:
        """Build a :class:`WordPressClient` for the given site.

        Raises:
            :class:`WordPressSiteNotFoundError`, :class:`WordPressSiteDisabledError`,
            :class:`WordPressMissingCredentialError`.
        """
        cfg = self._resolve_site(site_id)
        auth = self._resolve_auth(cfg)
        return WordPressClient(
            base_url=cfg.base_url,
            auth=auth,
            timeout=cfg.timeout_seconds,
            site_id=cfg.id,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_service: WordPressService | None = None


def get_wordpress_service() -> WordPressService:
    """Return the module-level :class:`WordPressService` singleton.

    Config is loaded from ``rex_config.json`` and credentials from the
    global :func:`~rex.credentials.get_credential_manager`.
    """
    global _service  # noqa: PLW0603
    if _service is None:
        from rex.config_manager import load_config
        from rex.credentials import get_credential_manager

        raw = load_config()
        wp_config = load_wordpress_config(raw)
        _service = WordPressService(
            wp_config=wp_config,
            credential_manager=get_credential_manager(),
        )
    return _service


__all__ = [
    "WordPressMissingCredentialError",
    "WordPressService",
    "WordPressSiteDisabledError",
    "WordPressSiteNotFoundError",
    "get_wordpress_service",
]
