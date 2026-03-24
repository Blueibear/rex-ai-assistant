"""OpenClaw HTTP client — shared transport for all OpenClaw API calls.

All authentication, retry logic, timeouts, and error handling are
centralised here so individual bridge/adapter modules stay thin.

Usage
-----
::

    from rex.openclaw.http_client import get_openclaw_client

    client = get_openclaw_client(config)
    if client is not None:
        result = client.post("/v1/chat/completions", json={...})
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout

from rex.openclaw.errors import OpenClawAPIError, OpenClawAuthError, OpenClawConnectionError

logger = logging.getLogger(__name__)

# Sentinel so get_openclaw_client() is a proper singleton per config combination.
_CLIENT_CACHE: dict[str, OpenClawClient] = {}


class OpenClawClient:
    """HTTP client for the OpenClaw gateway.

    Args:
        base_url: Gateway base URL, e.g. ``"http://127.0.0.1:18789"``.
        auth_token: Bearer token for ``Authorization`` header.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries on 429 / 5xx responses.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self.timeout = timeout
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return self.base_url + ("" if path.startswith("/") else "/") + path

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path)
        attempt = 0

        while True:
            attempt += 1
            logger.debug("%s %s (attempt %d)", method.upper(), path, attempt)
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    timeout=self.timeout,
                )
            except (RequestsConnectionError, Timeout) as exc:
                logger.warning("OpenClaw connection error on %s %s: %s", method.upper(), url, exc)
                raise OpenClawConnectionError(url, exc) from exc

            status = response.status_code
            logger.debug("%s %s -> %d", method.upper(), path, status)

            if status == 401:
                logger.warning("OpenClaw auth error (401) on %s", url)
                raise OpenClawAuthError(url)

            # Retry on 429 and 5xx.
            if status == 429 or status >= 500:
                if attempt > self.max_retries:
                    body = response.text or ""
                    logger.warning(
                        "OpenClaw API error %d on %s after %d attempts",
                        status,
                        path,
                        attempt,
                    )
                    raise OpenClawAPIError(status, body)

                # Respect Retry-After for 429; use exponential backoff otherwise.
                if status == 429 and "Retry-After" in response.headers:
                    wait = float(response.headers["Retry-After"])
                else:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s, …

                logger.warning(
                    "OpenClaw %d on %s — retrying in %.1fs (attempt %d/%d)",
                    status,
                    path,
                    wait,
                    attempt,
                    self.max_retries,
                )
                time.sleep(wait)
                continue

            if status >= 400:
                body = response.text or ""
                raise OpenClawAPIError(status, body)

            # Success — return parsed JSON (or empty dict for 204 etc.)
            if response.content:
                result: dict[str, Any] = response.json()
                return result
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a POST request and return the parsed JSON response."""
        return self._request("POST", path, json=json)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a GET request and return the parsed JSON response."""
        return self._request("GET", path, params=params)

    def patch(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a PATCH request and return the parsed JSON response."""
        return self._request("PATCH", path, json=json)

    def delete(self, path: str) -> dict[str, Any]:
        """Send a DELETE request and return the parsed JSON response."""
        return self._request("DELETE", path)


def get_openclaw_client(config: Any) -> OpenClawClient | None:
    """Return a shared :class:`OpenClawClient` for *config*, or ``None``.

    Returns ``None`` when ``config.openclaw_gateway_url`` is empty/missing
    so callers can use a simple ``if client:`` guard without branching on
    configuration availability.

    The client is cached per ``(base_url, token)`` pair so that calling this
    function multiple times with the same config returns the same instance.

    Args:
        config: An :class:`rex.config.AppConfig` instance (or any object
            with ``openclaw_gateway_url``, ``openclaw_gateway_token``,
            ``openclaw_gateway_timeout``, and ``openclaw_gateway_max_retries``
            attributes).
    """
    url: str = getattr(config, "openclaw_gateway_url", "") or ""
    if not url.strip():
        return None

    token: str = getattr(config, "openclaw_gateway_token", "") or ""
    timeout: int = int(getattr(config, "openclaw_gateway_timeout", 30))
    max_retries: int = int(getattr(config, "openclaw_gateway_max_retries", 3))

    cache_key = f"{url}::{token}"
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = OpenClawClient(
            base_url=url,
            auth_token=token,
            timeout=timeout,
            max_retries=max_retries,
        )

    return _CLIENT_CACHE[cache_key]


__all__ = [
    "OpenClawClient",
    "get_openclaw_client",
]
