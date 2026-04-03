"""Response cache for repeated factual queries (US-LAT-004).

Caches LLM responses keyed on normalized message text.  Cache entries expire
after a configurable TTL (default 5 minutes).  Cache lookup is bypassed for
messages that reference time-sensitive intents or invoke tools.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

# Patterns that indicate a time-sensitive query — bypass cache on match.
_TIME_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bright now\b",
        r"\bcurrent(ly)?\b",
        r"\btoday\b",
        r"\btonight\b",
        r"\bthis (morning|afternoon|evening|week|month|year)\b",
        r"\bjust (now|happened|said)\b",
        r"\blatest\b",
        r"\brecently\b",
        r"\bweather\b",
        r"\btemperature\b",
        r"\bforecast\b",
        r"\btime (is it|now)\b",
        r"\bwhat time\b",
        r"\btoday.s (date|schedule|calendar)\b",
        r"\bstock(s| price)\b",
        r"\bnews\b",
        r"\bscore(s)?\b",
    ]
]

# Keywords that indicate a tool-invoking query — bypass cache on match.
_TOOL_KEYWORDS: list[str] = [
    "email",
    "calendar",
    "schedule",
    "meeting",
    "appointment",
    "send",
    "home assistant",
    "light",
    "thermostat",
    "remind",
    "reminder",
    "alarm",
    "timer",
    "shopping list",
    "add to list",
    "search",
]

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _is_time_sensitive(text: str) -> bool:
    """Return True if *text* references time-sensitive information."""
    for pattern in _TIME_SENSITIVE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _is_tool_invoking(text: str) -> bool:
    """Return True if *text* likely triggers tool dispatch."""
    lower = text.lower()
    return any(kw in lower for kw in _TOOL_KEYWORDS)


def _should_bypass(text: str) -> bool:
    """Return True when the cache should be skipped for *text*."""
    return _is_time_sensitive(text) or _is_tool_invoking(text)


@dataclass
class _CacheEntry:
    response: str
    expires_at: float


class ResponseCache:
    """Thread-safe LRU-ish response cache with TTL expiry.

    Parameters
    ----------
    ttl:
        Seconds before a cached entry expires.  Default: 300 (5 minutes).
    max_size:
        Maximum number of entries to keep.  Oldest entries are evicted when
        the cache is full.  Default: 256.
    """

    def __init__(self, ttl: float = 300.0, max_size: int = 256) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._store: dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, message: str) -> str | None:
        """Return a cached response for *message*, or None on miss/bypass/expiry."""
        if _should_bypass(message):
            logger.debug("[cache] bypass: %r", message[:60])
            return None

        key = _normalize(message)
        now = time.monotonic()

        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.expires_at <= now:
                if entry is not None:
                    del self._store[key]
                self._misses += 1
                logger.debug(
                    "[cache] miss (total hits=%d misses=%d): %r",
                    self._hits,
                    self._misses,
                    key[:60],
                )
                return None
            self._hits += 1
            logger.debug(
                "[cache] hit (total hits=%d misses=%d): %r",
                self._hits,
                self._misses,
                key[:60],
            )
            return entry.response

    def put(self, message: str, response: str) -> None:
        """Store *response* for *message* unless the query should be bypassed."""
        if _should_bypass(message):
            return

        key = _normalize(message)
        expires_at = time.monotonic() + self._ttl

        with self._lock:
            # Evict oldest entry when at capacity (simple FIFO eviction).
            if key not in self._store and len(self._store) >= self._max_size:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = _CacheEntry(response=response, expires_at=expires_at)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return the fraction of lookups that were cache hits (0.0–1.0)."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Module-level helpers exposed for tests
    # ------------------------------------------------------------------

    @staticmethod
    def should_bypass(message: str) -> bool:
        """Public wrapper around the module-level bypass predicate."""
        return _should_bypass(message)


__all__ = ["ResponseCache"]
