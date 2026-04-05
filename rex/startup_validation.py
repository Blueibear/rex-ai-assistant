"""Startup environment variable validation for Rex.

Validates required and optional environment variables before any other
initialization, so misconfigured deployments fail loudly with clear
error messages rather than silently misbehaving.

Usage::

    from rex.startup_validation import check_startup_env

    # Call early in main() before any other initialization
    check_startup_env()  # uses built-in _ENV_SPECS, exits 1 on failure

Custom specs can be passed::

    from rex.startup_validation import EnvVarSpec, check_startup_env, _validate_int

    check_startup_env([
        EnvVarSpec("MY_REQUIRED_KEY", required=True),
        EnvVarSpec("MY_PORT", required=False, validator=_validate_int),
    ])
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------


def _validate_int(value: str) -> str | None:
    """Return None if *value* is a valid integer string, else an error string."""
    try:
        int(value)
        return None
    except (ValueError, TypeError):
        return f"expected integer, got {value!r}"


def _validate_url(value: str) -> str | None:
    """Return None if *value* is a valid http/https URL, else an error string."""
    try:
        parsed = urlparse(value)
    except Exception:  # noqa: BLE001
        return f"cannot be parsed as a URL: {value!r}"
    if parsed.scheme not in ("http", "https"):
        return f"expected http/https URL, got {value!r}"
    if not parsed.netloc:
        return f"expected http/https URL, got {value!r}"
    return None


# ---------------------------------------------------------------------------
# EnvVarSpec
# ---------------------------------------------------------------------------


@dataclass
class EnvVarSpec:
    """Specification for a single environment variable.

    Attributes:
        name: Environment variable name (e.g. ``"REX_SPEAK_PORT"``).
        required: Whether the variable *must* be present.  Optional by default.
        validator: Optional callable that receives the string value and returns
            ``None`` on success or a non-empty error string on failure.
        description: Human-readable description (used in error messages).
    """

    name: str
    required: bool = False
    validator: Callable[[str], str | None] | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in specs for known Rex env vars
# ---------------------------------------------------------------------------

_ENV_SPECS: list[EnvVarSpec] = [
    EnvVarSpec(
        "REX_SPEAK_PORT",
        required=False,
        validator=_validate_int,
        description="Port for the TTS speak API server",
    ),
    EnvVarSpec(
        "OPENAI_BASE_URL",
        required=False,
        validator=_validate_url,
        description="Custom base URL for the OpenAI-compatible API",
    ),
    EnvVarSpec(
        "HA_BASE_URL",
        required=False,
        validator=_validate_url,
        description="Home Assistant base URL",
    ),
    EnvVarSpec(
        "OLLAMA_BASE_URL",
        required=False,
        validator=_validate_url,
        description="Ollama API base URL",
    ),
]


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------


def validate_startup_env(
    specs: list[EnvVarSpec] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Validate environment variables against *specs*.

    Args:
        specs: List of :class:`EnvVarSpec` to validate.  When ``None``, the
            built-in :data:`_ENV_SPECS` list is used.
        env: Mapping to use instead of :data:`os.environ`.  Useful for testing.

    Returns:
        A list of human-readable error strings.  Empty when all checks pass.
    """
    if specs is None:
        specs = _ENV_SPECS
    if env is None:
        env = dict(os.environ)

    errors: list[str] = []

    for spec in specs:
        value = env.get(spec.name)
        if not value:
            # Treat missing (None) and empty string the same: not provided.
            if spec.required:
                errors.append(f"Required environment variable {spec.name!r} is not set.")
        else:
            if spec.validator is not None:
                err = spec.validator(value)
                if err:
                    errors.append(f"Environment variable {spec.name!r} is invalid: {err}.")

    return errors


def check_startup_env(
    specs: list[EnvVarSpec] | None = None,
    *,
    env: dict[str, str] | None = None,
    exit_on_error: bool = True,
) -> list[str]:
    """Validate environment variables and optionally exit on failure.

    This is the primary entry point.  Call it as early as possible in
    ``main()`` before any other initialization.

    Args:
        specs: Specs to validate (defaults to built-in :data:`_ENV_SPECS`).
        env: Override environment mapping (defaults to :data:`os.environ`).
        exit_on_error: When ``True`` (the default) and validation fails, print
            all errors to stderr and call :func:`sys.exit(1)`.

    Returns:
        List of error strings.  Empty when all checks pass.
    """
    errors = validate_startup_env(specs, env=env)
    if errors and exit_on_error:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    return errors


__all__ = [
    "EnvVarSpec",
    "validate_startup_env",
    "check_startup_env",
    "_validate_int",
    "_validate_url",
    "_ENV_SPECS",
]
