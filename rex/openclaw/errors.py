"""OpenClaw-specific exception types.

All exceptions inherit from :class:`rex.assistant_errors.AssistantError`
so they integrate with Rex's existing error-handling infrastructure.
"""

from __future__ import annotations

from rex.assistant_errors import AssistantError


class OpenClawConnectionError(AssistantError):
    """Raised when a connection to the OpenClaw gateway cannot be established.

    Args:
        url: The URL that was unreachable.
        cause: The underlying exception.
    """

    def __init__(self, url: str, cause: Exception) -> None:
        super().__init__(f"Cannot connect to OpenClaw gateway at {url!r}: {cause}")
        self.url = url
        self.cause = cause


class OpenClawAuthError(AssistantError):
    """Raised when the OpenClaw gateway returns HTTP 401 Unauthorized."""

    def __init__(self, url: str = "") -> None:
        msg = "OpenClaw gateway authentication failed (401)"
        if url:
            msg += f" — {url}"
        super().__init__(msg)
        self.url = url


class OpenClawAPIError(AssistantError):
    """Raised when the OpenClaw gateway returns an unexpected HTTP error
    after all retries are exhausted.

    Args:
        status: HTTP status code returned by the gateway.
        body: Response body (may be empty).
    """

    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"OpenClaw API error {status}: {body[:200]}")
        self.status = status
        self.body = body


__all__ = [
    "OpenClawConnectionError",
    "OpenClawAuthError",
    "OpenClawAPIError",
]
