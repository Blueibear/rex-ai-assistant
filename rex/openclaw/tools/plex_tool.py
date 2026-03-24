"""OpenClaw Plex tools — US-P5-017.

Exposes four Rex Plex capabilities as OpenClaw tools:

* ``plex_search``   — search the Plex library by keyword
* ``plex_play``     — start playback on a Plex client
* ``plex_pause``    — pause playback on a Plex client
* ``plex_stop``     — stop playback on a Plex client

Each function is a thin wrapper around :mod:`rex.plex_client`.  All network
I/O stays inside that module; these functions only translate arguments and
return value shapes.

Typical usage::

    from rex.openclaw.tools.plex_tool import plex_search, plex_play, register

    results = plex_search("Breaking Bad")
    plex_play("my-client-id", rating_key=results["results"][0]["rating_key"])
    handles = register()
"""

from __future__ import annotations

import logging
from typing import Any

from rex.plex_client import get_plex_client

logger = logging.getLogger(__name__)

TOOL_NAMES = ("plex_search", "plex_play", "plex_pause", "plex_stop")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def plex_search(
    query: str,
    limit: int = 20,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search the Plex library for *query*.

    Args:
        query: Search string.
        limit: Maximum number of results to return (default 20).
        context: Optional ambient context dict (unused, reserved for future use).

    Returns:
        ``{"ok": True, "results": [...], "error": None}`` on success, or
        ``{"ok": False, "results": [], "error": "<message>"}`` on failure.
        Each result dict has keys: ``rating_key``, ``title``, ``media_type``,
        ``year``, ``summary``, ``duration_ms``.
    """
    client = get_plex_client()
    if client is None:
        return {"ok": False, "results": [], "error": "Plex client not configured"}
    if not client.enabled:
        return {
            "ok": False,
            "results": [],
            "error": "Plex client not enabled (missing URL or token)",
        }
    try:
        items = client.search(query, limit=limit)
        return {
            "ok": True,
            "results": [
                {
                    "rating_key": item.rating_key,
                    "title": item.title,
                    "media_type": item.media_type,
                    "year": item.year,
                    "summary": item.summary,
                    "duration_ms": item.duration_ms,
                }
                for item in items
            ],
            "error": None,
        }
    except Exception as exc:
        logger.warning("plex_search failed: %s", exc)
        return {"ok": False, "results": [], "error": str(exc)}


def plex_play(
    client_id: str,
    rating_key: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start playback on Plex client *client_id*.

    Args:
        client_id: Identifier of the Plex player (e.g. machine identifier).
        rating_key: Optional media item key returned by :func:`plex_search`.
            When provided, playback of that specific item is started.
        context: Optional ambient context dict (unused).

    Returns:
        ``{"ok": True, "error": None}`` on success, or
        ``{"ok": False, "error": "<message>"}`` on failure.
    """
    client = get_plex_client()
    if client is None:
        return {"ok": False, "error": "Plex client not configured"}
    if not client.enabled:
        return {"ok": False, "error": "Plex client not enabled (missing URL or token)"}
    try:
        success = client.play(client_id, rating_key=rating_key)
        if success:
            return {"ok": True, "error": None}
        return {"ok": False, "error": "Plex play command returned failure"}
    except Exception as exc:
        logger.warning("plex_play failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def plex_pause(
    client_id: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pause playback on Plex client *client_id*.

    Args:
        client_id: Identifier of the Plex player.
        context: Optional ambient context dict (unused).

    Returns:
        ``{"ok": True, "error": None}`` on success, or
        ``{"ok": False, "error": "<message>"}`` on failure.
    """
    client = get_plex_client()
    if client is None:
        return {"ok": False, "error": "Plex client not configured"}
    if not client.enabled:
        return {"ok": False, "error": "Plex client not enabled (missing URL or token)"}
    try:
        success = client.pause(client_id)
        if success:
            return {"ok": True, "error": None}
        return {"ok": False, "error": "Plex pause command returned failure"}
    except Exception as exc:
        logger.warning("plex_pause failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def plex_stop(
    client_id: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stop playback on Plex client *client_id*.

    Args:
        client_id: Identifier of the Plex player.
        context: Optional ambient context dict (unused).

    Returns:
        ``{"ok": True, "error": None}`` on success, or
        ``{"ok": False, "error": "<message>"}`` on failure.
    """
    client = get_plex_client()
    if client is None:
        return {"ok": False, "error": "Plex client not configured"}
    if not client.enabled:
        return {"ok": False, "error": "Plex client not enabled (missing URL or token)"}
    try:
        success = client.stop(client_id)
        if success:
            return {"ok": True, "error": None}
        return {"ok": False, "error": "Plex stop command returned failure"}
    except Exception as exc:
        logger.warning("plex_stop failed: %s", exc)
        return {"ok": False, "error": str(exc)}


