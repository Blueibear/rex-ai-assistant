"""Retry utilities for transient failures."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior."""

    attempts: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 10.0
    retry_exceptions: tuple[type[Exception], ...] = (Exception,)


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


__all__ = ["RetryPolicy", "retry_call"]
