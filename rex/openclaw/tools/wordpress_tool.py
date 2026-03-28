"""OpenClaw tool adapter — wordpress_health_check.

Wraps Rex's :class:`~rex.wordpress.service.WordPressService` to expose
WordPress health-check queries for registration with OpenClaw's tool system.

WordPress integration is **read-only** in the current Rex implementation
(``GET /wp-json`` and ``GET /wp-json/wp/v2/users/me`` only).  No write
operations are available.  This tool reflects that boundary.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :func:`wp_health_check` callable works
independently of OpenClaw.

Typical usage::

    from rex.openclaw.tools.wordpress_tool import wp_health_check, register

    result = wp_health_check("myblog")
    # {'ok': True, 'site_name': 'My Blog', 'site_url': '...', 'error': None}

    register()   # no-op if openclaw not installed
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Tool name used when registering with OpenClaw.
TOOL_NAME = "wordpress_health_check"

#: Human-readable description forwarded to OpenClaw's tool registry.
TOOL_DESCRIPTION = (
    "Check that a configured WordPress site is reachable and credentials are valid. "
    'Args: {"site_id": "myblog"}'
)


def _get_wordpress_service() -> Any:
    """Return the WordPress service singleton (injectable in tests)."""
    from rex.wordpress.service import get_wordpress_service

    return get_wordpress_service()


def wp_health_check(
    site_id: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a health check against a configured WordPress site.

    Calls ``GET /wp-json`` (always public) and, if credentials are
    configured, ``GET /wp-json/wp/v2/users/me`` to verify auth.

    .. note::
        The WordPress integration is read-only.  No write operations
        are available from this tool.

    Args:
        site_id: The site identifier from ``wordpress.sites[].id``
            in ``rex_config.json``.
        context: Optional ambient context dict (unused; reserved for future
            injection).

    Returns:
        A dict with keys:
        - ``ok`` (bool): True on success.
        - ``site_name`` (str): Human-readable site name, if available.
        - ``site_url`` (str): Site URL, if available.
        - ``wp_detected`` (bool): True if response looks like WordPress.
        - ``auth_ok`` (bool | None): Credential check result; None if no auth configured.
        - ``error`` (str | None): Error message on failure.
    """
    try:
        service = _get_wordpress_service()
        result = service.health(site_id)
        return {
            "ok": result.ok,
            "site_name": result.site_name,
            "site_url": result.site_url,
            "wp_detected": result.wp_detected,
            "auth_ok": result.auth_ok,
            "error": result.error,
        }
    except Exception as exc:
        logger.warning("[WP tool] Health check failed for site %r: %s", site_id, exc)
        return {
            "ok": False,
            "site_name": "",
            "site_url": "",
            "wp_detected": False,
            "auth_ok": None,
            "error": str(exc),
        }
