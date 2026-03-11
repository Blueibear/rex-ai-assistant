"""Plex Media Server API client for Rex."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

try:
    import requests as _requests_module
except ImportError as _exc:
    _requests_module = None  # type: ignore[assignment]
    _REQUESTS_IMPORT_ERROR: Exception | None = _exc
else:
    _REQUESTS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

_PLEX_AVAILABLE = _requests_module is not None


def _require_requests() -> None:
    if _requests_module is None:
        raise RuntimeError(
            "The Plex client requires the 'requests' package. "
            "Install with: pip install requests"
        ) from _REQUESTS_IMPORT_ERROR


@dataclass
class PlexLibrary:
    """Represents a Plex library section."""

    library_id: str
    title: str
    library_type: str
    count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlexMediaItem:
    """Represents a single media item in Plex."""

    rating_key: str
    title: str
    media_type: str
    year: int | None = None
    summary: str = ""
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PlexConnectionError(Exception):
    """Raised when the Plex server is unreachable."""


class PlexAuthError(Exception):
    """Raised when Plex authentication fails."""


class PlexClient:
    """Thin HTTP client for the Plex Media Server API."""

    def __init__(
        self,
        base_url: str = "",
        token: str = "",
        *,
        session: Any = None,
    ) -> None:
        _require_requests()
        self._base_url = base_url.rstrip("/")
        self._token = token
        if session is not None:
            self._session = session
        else:
            self._session = _requests_module.Session()
        self._session.headers.update(
            {
                "X-Plex-Token": self._token,
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Issue a GET request and return parsed JSON."""
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=10)
        except Exception as exc:
            raise PlexConnectionError(f"Cannot reach Plex at {url}: {exc}") from exc
        if resp.status_code == 401:
            raise PlexAuthError("Invalid or missing Plex token")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return True when both a URL and token are configured."""
        return bool(self._base_url) and bool(self._token)

    def ping(self) -> bool:
        """Return True if the Plex server responds to a health check."""
        try:
            self._get("/identity")
            return True
        except PlexConnectionError:
            return False
        except PlexAuthError:
            return False
        except Exception:
            return False

    def get_libraries(self) -> list[PlexLibrary]:
        """Return all library sections from the Plex server."""
        data = self._get("/library/sections")
        sections = (
            data.get("MediaContainer", {}).get("Directory") or []
        )
        libraries: list[PlexLibrary] = []
        for sec in sections:
            libraries.append(
                PlexLibrary(
                    library_id=str(sec.get("key", "")),
                    title=sec.get("title", ""),
                    library_type=sec.get("type", ""),
                    count=int(sec.get("count", 0)),
                    metadata=sec,
                )
            )
        return libraries

    def search(self, query: str, *, limit: int = 20) -> list[PlexMediaItem]:
        """Search the Plex library for *query* and return matching items."""
        if not query:
            return []
        data = self._get("/search", params={"query": query, "limit": limit})
        items: list[PlexMediaItem] = []
        for entry in data.get("MediaContainer", {}).get("Metadata") or []:
            items.append(
                PlexMediaItem(
                    rating_key=str(entry.get("ratingKey", "")),
                    title=entry.get("title", ""),
                    media_type=entry.get("type", ""),
                    year=entry.get("year"),
                    summary=entry.get("summary", ""),
                    duration_ms=entry.get("duration"),
                    metadata=entry,
                )
            )
        return items


# ---------------------------------------------------------------------------
# Global singleton helpers
# ---------------------------------------------------------------------------

_client: PlexClient | None = None


def get_plex_client() -> PlexClient | None:
    """Return the global PlexClient, or None if not configured."""
    return _client


def set_plex_client(client: PlexClient | None) -> None:
    """Replace the global PlexClient (used for testing)."""
    global _client
    _client = client


def init_plex_client(base_url: str, token: str) -> PlexClient:
    """Create and store a global PlexClient instance."""
    global _client
    _client = PlexClient(base_url=base_url, token=token)
    return _client
