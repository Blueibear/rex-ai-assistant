"""HTTP client for the WordPress REST API (read-only, Cycle 6.1).

API endpoints used
------------------
- ``GET /wp-json``
    Public site info.  Returns ``{"name": ..., "namespaces": [...], ...}``.

- ``GET /wp-json/wp/v2/users/me``
    Authenticated user check.  Returns ``{"id": ..., "name": ...}`` on
    success; 401 when credentials are invalid.

Security notes
--------------
- Credentials are passed as HTTP Basic Auth.  The raw credential string
  (``username:password``) is **never** logged.
- Only the site ``id`` (label) appears in log output.
- ``requests`` is used when available (it is a base dependency of Rex).

Dependencies
------------
Uses ``requests`` (already in ``[project.dependencies]``).
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Accept header sent with every request.
_ACCEPT_JSON = {"Accept": "application/json"}

# Keys expected in a valid /wp-json response (best-effort WP detection).
_WP_API_KEYS = frozenset({"name", "namespaces", "routes", "authentication"})


@dataclass
class WPHealthResult:
    """Result of a WordPress health check."""

    ok: bool
    reachable: bool = False
    wp_detected: bool = False
    auth_ok: bool | None = None  # None means auth was not attempted
    site_name: str = ""
    site_url: str = ""
    error: str | None = None


class WordPressClient:
    """Minimal read-only HTTP client for the WordPress REST API.

    Parameters
    ----------
    base_url:
        Base URL of the WordPress site (e.g. ``"https://example.com"``).
        Trailing slash is stripped internally.
    auth:
        ``(username, password)`` tuple for HTTP Basic Auth, or ``None`` for
        unauthenticated access.
    timeout:
        Request timeout in seconds.
    site_id:
        Human-readable label used in log messages (never the credential).
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth: tuple[str, str] | None = None,
        timeout: int = 15,
        site_id: str = "",
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._auth = auth
        self._timeout = timeout
        self._label = site_id or self._base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> WPHealthResult:
        """Check that the site is reachable and looks like WordPress.

        Calls ``GET /wp-json`` (always public).  If auth is configured,
        also calls ``GET /wp-json/wp/v2/users/me`` to verify credentials.

        Returns:
            :class:`WPHealthResult` describing the outcome.
        """
        url = f"{self._base_url}/wp-json"
        logger.debug("WP health check for %s", self._label)
        try:
            data = self._get(url, auth=None)
        except Exception as exc:  # noqa: BLE001
            message = _safe_error_message(exc)
            logger.warning("WP health check failed for %s: %s", self._label, message)
            return WPHealthResult(ok=False, reachable=False, error=message)

        if not isinstance(data, dict):
            return WPHealthResult(
                ok=False,
                reachable=True,
                error="Unexpected response format from /wp-json",
            )

        wp_detected = bool(_WP_API_KEYS & data.keys())
        site_name = str(data.get("name", ""))
        site_url = str(data.get("url", ""))

        result = WPHealthResult(
            ok=True,
            reachable=True,
            wp_detected=wp_detected,
            site_name=site_name,
            site_url=site_url,
        )

        if self._auth is not None:
            result.auth_ok = self._check_auth()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_auth(self) -> bool:
        """Call ``GET /wp-json/wp/v2/users/me`` and return True on success."""
        url = f"{self._base_url}/wp-json/wp/v2/users/me"
        logger.debug("WP auth check for %s", self._label)
        try:
            data = self._get(url, auth=self._auth)
            return isinstance(data, dict) and "id" in data
        except Exception as exc:  # noqa: BLE001
            # 401/403 surface here — treat as auth failure, not a hard error.
            logger.debug("WP auth check failed for %s: %s", self._label, type(exc).__name__)
            return False

    def _get(self, url: str, *, auth: tuple[str, str] | None) -> Any:
        """Perform a GET request and return parsed JSON.

        Args:
            url: Full URL to fetch.
            auth: Basic-auth tuple or ``None``.

        Returns:
            Parsed JSON (dict or list).

        Raises:
            requests.HTTPError: On 4xx/5xx responses.
            requests.ConnectionError: If the host is unreachable.
            ValueError: If the response is not valid JSON.
        """
        import requests  # noqa: PLC0415  (base dep, always available)  # type: ignore[import-untyped]

        resp = requests.get(
            url,
            headers=_ACCEPT_JSON,
            auth=auth,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()


def _validate_base_url(base_url: str) -> str:
    """Validate and normalize base_url for safe read-only requests."""
    parsed = urlparse(base_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"base_url must use http or https scheme, got: {scheme!r}")
    if not parsed.netloc:
        raise ValueError("base_url must include a host (netloc)")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include embedded credentials")
    _validate_remote_host(parsed.hostname)
    return base_url.rstrip("/")


def _validate_remote_host(hostname: str | None) -> None:
    """Reject localhost/private/reserved targets to reduce SSRF risk."""
    if not hostname:
        raise ValueError("base_url is missing a hostname")

    lowered = hostname.strip().lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        raise ValueError("base_url host must not be localhost or local network")

    try:
        addresses = {ai[4][0] for ai in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve base_url host: {hostname}") from exc

    for addr in addresses:
        ip = ip_address(addr)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("base_url host resolves to a local or reserved address")


def _safe_error_message(exc: Exception) -> str:
    """Return a non-sensitive error message for CLI/logging output."""
    import requests  # noqa: PLC0415

    if isinstance(exc, requests.Timeout):
        return "Request timed out"
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        return f"HTTP error from WordPress API (status={status_code})"
    if isinstance(exc, requests.RequestException):
        return "Request to WordPress API failed"
    if isinstance(exc, ValueError):
        return str(exc)
    return "Unexpected error while querying WordPress API"


__all__ = [
    "WPHealthResult",
    "WordPressClient",
]
