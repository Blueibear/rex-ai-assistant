"""Utilities for user-friendly dependency and connection error messages.

Provides helpers to:
- Format ImportError with human-readable messages and install hints (AC1).
- Detect network connection errors for friendly LM Studio messages (AC2).
- Query debug mode so callers can decide whether to show tracebacks (AC3).
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Debug mode detection
# ---------------------------------------------------------------------------


def is_debug_mode() -> bool:
    """Return True when REX_DEBUG env var is set to a truthy value.

    Callers use this to decide whether to show full tracebacks.
    """
    return os.environ.get("REX_DEBUG", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Friendly ImportError formatting
# ---------------------------------------------------------------------------


def friendly_import_error(
    package_name: str,
    install_cmd: str,
    exc: ImportError,
) -> ImportError:
    """Return a new ImportError with a user-friendly message and install hint.

    Args:
        package_name: Display name of the missing package (e.g. ``"openai"``).
        install_cmd:  Full install command (e.g. ``"pip install openai"``).
        exc:          The original ImportError to chain.

    Returns:
        A new :class:`ImportError` with a message like::

            Rex requires 'openai' for this feature but it is not installed.
            Install it with: pip install openai
    """
    msg = (
        f"Rex requires '{package_name}' for this feature but it is not installed.\n"
        f"Install it with: {install_cmd}"
    )
    new_exc = ImportError(msg)
    new_exc.__cause__ = exc
    return new_exc


# ---------------------------------------------------------------------------
# Connection error detection
# ---------------------------------------------------------------------------

# Names of exception types that represent a failed TCP/HTTP connection.
_CONNECTION_ERROR_TYPE_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectionError",
        "ConnectTimeout",
        "RemoteDisconnected",
        "NewConnectionError",
        "MaxRetryError",
    }
)


def is_connection_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a network connection failure.

    Handles both stdlib errors (``ConnectionRefusedError``, ``OSError``) and
    the error types used by third-party HTTP libraries (``httpx``,
    ``requests``) without importing those libraries.
    """
    if isinstance(exc, (ConnectionRefusedError, ConnectionError, BrokenPipeError)):
        return True
    # OSError errno 111 is ECONNREFUSED on Linux; check the family broadly.
    if isinstance(exc, OSError) and exc.errno is not None:
        return True
    type_name = type(exc).__name__
    if type_name in _CONNECTION_ERROR_TYPE_NAMES:
        return True
    # Check cause chain — httpx sometimes wraps errors
    if exc.__cause__ is not None and is_connection_error(exc.__cause__):
        return True
    return False


# ---------------------------------------------------------------------------
# Typed exception for LM Studio connectivity
# ---------------------------------------------------------------------------


class LMStudioConnectionError(RuntimeError):
    """Raised when Rex cannot reach the LM Studio API server.

    The message always follows the pattern::

        Rex can't reach LM Studio at <url>. Is LM Studio running?
    """


__all__ = [
    "LMStudioConnectionError",
    "friendly_import_error",
    "is_connection_error",
    "is_debug_mode",
]
