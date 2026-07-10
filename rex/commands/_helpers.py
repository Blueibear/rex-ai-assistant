"""Shared helpers for Rex CLI command modules.

Extracted verbatim from ``rex/cli.py`` (US-REM-027).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta


def _get_version() -> str:
    """Return the current Rex version."""
    try:
        from rex.contracts.version import CONTRACT_VERSION

        return CONTRACT_VERSION
    except ImportError:
        return "0.1.0"


def _load_email_config_safe():
    """Load EmailConfig from rex_config.json, returning None on failure."""
    try:
        from rex.config_manager import load_config as _load_json_config
        from rex.email_backends.account_config import load_email_config

        raw_config = _load_json_config()
        return load_email_config(raw_config)
    except Exception:
        return None


def _load_email_resolver_safe():
    """Load the per-user EmailAccountResolver, returning None on failure."""
    try:
        from rex.email_accounts import EmailAccountResolver

        return EmailAccountResolver.load()
    except Exception:
        return None


def _resolve_cli_user(args: argparse.Namespace) -> str | None:
    """Resolve active user context for commands that accept ``--user``."""
    from rex.identity import resolve_active_user

    explicit_user = getattr(args, "user", None)
    try:
        from rex.config_manager import load_config as _load_json_config

        config = _load_json_config()
    except Exception:
        config = None
    return resolve_active_user(explicit_user, config=config)


def _parse_ttl(ttl_str: str) -> timedelta | None:
    """Parse TTL string like '7d', '24h', '30m', '1w', '10s' into timedelta."""
    ttl_str = ttl_str.strip().lower()
    if not ttl_str:
        return None

    try:
        if ttl_str.endswith("w"):
            return timedelta(weeks=int(ttl_str[:-1]))
        if ttl_str.endswith("d"):
            return timedelta(days=int(ttl_str[:-1]))
        if ttl_str.endswith("h"):
            return timedelta(hours=int(ttl_str[:-1]))
        if ttl_str.endswith("m"):
            return timedelta(minutes=int(ttl_str[:-1]))
        if ttl_str.endswith("s"):
            return timedelta(seconds=int(ttl_str[:-1]))
        return timedelta(days=int(ttl_str))
    except ValueError:
        return None


def _parse_datetime_strict(dt_str: str) -> datetime:
    """
    Parse a datetime string in common formats and return a timezone-aware datetime.

    Accepted examples:
      - 2026-01-29 14:30
      - 2026-01-29 14:30:00
      - 2026/01/29 14:30
      - ISO-8601 (datetime.fromisoformat compatible)

    If no timezone is provided, local timezone is assumed.
    """
    dt_str = dt_str.strip()
    if not dt_str:
        raise ValueError("Datetime cannot be empty")

    # First, try ISO parsing (supports "YYYY-MM-DD HH:MM[:SS]" and true ISO forms)
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt
    except ValueError:
        pass

    # Then, try a few explicit formats
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        except ValueError:
            continue

    raise ValueError("Invalid datetime format. Use YYYY-MM-DD HH:MM (or ISO-8601).")
