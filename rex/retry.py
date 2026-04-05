"""Retry utilities for transient failures."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that indicate a transient error and should be retried.
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 503})

# HTTP status codes that indicate a permanent client error and must NOT be retried.
_NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403})


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior."""

    attempts: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 10.0
    retry_exceptions: tuple[type[Exception], ...] = (Exception,)


@dataclass
class RetryConfig:
    """HTTP-aware retry configuration for LLM provider calls.

    Attributes:
        max_attempts: Maximum number of total attempts (including the first).
            Default: 3.
        base_delay: Initial sleep duration in seconds before the first retry.
            Each subsequent retry doubles the delay (exponential backoff).
            Default: 1.0.
    """

    max_attempts: int = 3
    base_delay: float = 1.0


class NonRetryableError(Exception):
    """Raised when an HTTP status code is not eligible for retry (400, 401, 403).

    Wraps the original exception so the caller can inspect it.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _get_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from a provider exception, if available."""
    # OpenAI, Anthropic, and many HTTP libs expose .status_code directly.
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    # Some libraries expose .response with a .status_code attribute.
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_transient(exc: BaseException) -> bool:
    """Return True when the exception represents a transient, retryable failure."""
    code = _get_status_code(exc)
    if code is not None:
        return code in _TRANSIENT_STATUS_CODES
    # Treat generic timeout / connection errors as transient even without a code.
    exc_type_name = type(exc).__name__.lower()
    return any(
        token in exc_type_name
        for token in ("timeout", "connection", "network", "serviceunavailable")
    )


def _is_non_retryable(exc: BaseException) -> bool:
    """Return True when the exception represents a permanent client error."""
    code = _get_status_code(exc)
    return code is not None and code in _NON_RETRYABLE_STATUS_CODES


def with_retry(
    fn: Callable[[], T],
    config: RetryConfig | None = None,
) -> T:
    """Call *fn* and retry on transient errors with exponential back-off.

    Transient errors (HTTP 429, 503, timeout / connection exceptions) are
    retried up to ``config.max_attempts`` times total.  The delay between
    consecutive attempts starts at ``config.base_delay`` seconds and doubles
    on each retry.

    Non-retryable errors (HTTP 400, 401, 403) are re-raised immediately
    without any retry, wrapped in :class:`NonRetryableError`.

    Args:
        fn: Zero-argument callable to invoke.
        config: :class:`RetryConfig` instance.  A default instance is used
            when *config* is ``None``.

    Returns:
        The return value of *fn* on success.

    Raises:
        NonRetryableError: When *fn* raises with a non-retryable HTTP status.
        Exception: The last exception from *fn* when all retries are exhausted.
    """
    cfg = config or RetryConfig()
    delay = cfg.base_delay
    last_exc: BaseException | None = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if _is_non_retryable(exc):
                code = _get_status_code(exc)
                raise NonRetryableError(
                    f"Non-retryable error (HTTP {code}): {exc}", status_code=code
                ) from exc

            last_exc = exc

            if attempt >= cfg.max_attempts:
                # All attempts consumed — re-raise the original exception.
                raise

            if _is_transient(exc):
                logger.warning(
                    "Transient error on attempt %d/%d (sleeping %.1fs): %s",
                    attempt,
                    cfg.max_attempts,
                    delay,
                    exc,
                )
            else:
                logger.warning(
                    "Unexpected error on attempt %d/%d (sleeping %.1fs): %s",
                    attempt,
                    cfg.max_attempts,
                    delay,
                    exc,
                )

            jitter = random.uniform(0, delay * 0.1)
            time.sleep(delay + jitter)
            delay *= 2.0

    # Should never be reached, but satisfies the type checker.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("with_retry: exhausted attempts without result")  # pragma: no cover


def retry_call(
    func: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Execute a callable with retry and exponential backoff.

    Args:
        func: Callable to execute.
        policy: RetryPolicy configuration.
        on_retry: Optional callback invoked before sleeping.

    Returns:
        Result of the callable.

    Raises:
        Exception if all retries fail.
    """
    policy = policy or RetryPolicy()
    attempt = 0
    backoff = policy.initial_backoff_seconds

    while True:
        try:
            return func()
        except policy.retry_exceptions as exc:
            attempt += 1
            if attempt >= policy.attempts:
                raise
            sleep_for = min(backoff, policy.max_backoff_seconds)
            sleep_for += random.uniform(0, sleep_for * 0.1)
            if on_retry:
                on_retry(attempt, exc, sleep_for)
            else:
                logger.warning(
                    "Retrying after error (attempt %d/%d): %s",
                    attempt,
                    policy.attempts,
                    exc,
                )
            time.sleep(sleep_for)
            backoff *= policy.backoff_multiplier


__all__ = [
    "RetryConfig",
    "RetryPolicy",
    "NonRetryableError",
    "retry_call",
    "with_retry",
]
