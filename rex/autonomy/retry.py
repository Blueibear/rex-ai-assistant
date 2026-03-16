"""Retry logic with exponential backoff for the Rex autonomy engine.

:func:`retry_step` wraps a single tool-call callable and retries it
automatically when the error is classified as transient (network blips,
rate-limits, temporary unavailability).  Non-transient errors are
re-raised immediately so the plan runner can handle them (e.g. by
triggering the replanner).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transient-error classification
# ---------------------------------------------------------------------------

#: Exception types that are always considered transient.
_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)

#: HTTP status codes (as strings) that indicate transient server-side errors.
_TRANSIENT_HTTP_CODES: frozenset[str] = frozenset({"429", "503"})


def is_transient(exc: BaseException) -> bool:
    """Return ``True`` if *exc* represents a transient (retryable) failure.

    The following are considered transient:

    - :exc:`TimeoutError` and :exc:`ConnectionError` subclasses.
    - Any exception whose string representation contains ``"429"`` or
      ``"503"`` (HTTP rate-limit and service-unavailable responses).

    Args:
        exc: The exception to classify.

    Returns:
        ``True`` if the error is transient, ``False`` otherwise.
    """
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    msg = str(exc)
    return any(code in msg for code in _TRANSIENT_HTTP_CODES)


# ---------------------------------------------------------------------------
# retry_step
# ---------------------------------------------------------------------------


def retry_step(
    step_fn: Callable[[], str],
    step_id: str,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Execute *step_fn* with exponential backoff on transient failures.

    Attempts *step_fn()* up to *max_attempts* times.  After each transient
    failure (except the last attempt), waits ``base_delay * 2^(attempt-1)``
    seconds before retrying.  Non-transient errors are re-raised on the
    first occurrence.

    Args:
        step_fn: Zero-argument callable that performs the step and returns
            a result string.
        step_id: Human-readable step identifier used in log messages.
        max_attempts: Maximum total attempts.  Defaults to ``3``.
        base_delay: Initial delay in seconds before the first retry.
            Subsequent delays double (exponential backoff).
            Defaults to ``1.0``.

    Returns:
        The string result returned by *step_fn* on success.

    Raises:
        Exception: The last transient exception if all attempts are
            exhausted, or the first non-transient exception immediately.

    Log format (DEBUG):
        ``Retrying step {id}, attempt {n}/{max}, delay {delay}s``
    """
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return step_fn()
        except Exception as exc:  # noqa: BLE001
            if not is_transient(exc):
                # Non-transient — do not retry; let the runner handle it.
                raise

            last_exc = exc

            if attempt < max_attempts:
                delay = base_delay * (2.0 ** (attempt - 1))
                logger.debug(
                    "Retrying step %s, attempt %d/%d, delay %ss",
                    step_id,
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)

    # All attempts exhausted — re-raise the last transient exception.
    assert last_exc is not None  # guaranteed: loop ran at least once
    raise last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "retry_step",
    "is_transient",
]
