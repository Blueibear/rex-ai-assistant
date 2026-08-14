"""Identity-safe bounded cache primitives for deterministic context artifacts."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Generic, TypeVar

from rex.identity import validate_user_id
from rex.runtime.turn import TurnScope

T = TypeVar("T")


class ContextCacheCategory(StrEnum):
    """Fixed metric categories; never derived from user-controlled content."""

    PRIVATE = "private_context"
    HOUSEHOLD = "household_context"


@dataclass(frozen=True, slots=True)
class ContextCacheVersions:
    """Content-free revision tokens that define cache validity."""

    identity: str
    policy: str
    permission: str
    model: str
    capability: str
    config: str
    memory: str
    prompt_template: str

    def __post_init__(self) -> None:
        for field_name in (
            "identity",
            "policy",
            "permission",
            "model",
            "capability",
            "config",
            "memory",
            "prompt_template",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} revision must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())


@dataclass(frozen=True, slots=True)
class ContextCacheKey:
    """Validated cache partition plus all deterministic revision tokens."""

    scope: TurnScope
    user_id: str | None
    versions: ContextCacheVersions

    def __post_init__(self) -> None:
        scope = TurnScope(self.scope)
        object.__setattr__(self, "scope", scope)
        if scope is TurnScope.USER:
            if self.user_id is None:
                raise ValueError("private context cache keys require a user_id")
            object.__setattr__(self, "user_id", validate_user_id(self.user_id))
        elif self.user_id is not None:
            raise ValueError("household context cache keys must not carry a private user_id")

    @classmethod
    def private(cls, user_id: str, versions: ContextCacheVersions) -> ContextCacheKey:
        return cls(scope=TurnScope.USER, user_id=user_id, versions=versions)

    @classmethod
    def household(cls, versions: ContextCacheVersions) -> ContextCacheKey:
        return cls(scope=TurnScope.HOUSEHOLD, user_id=None, versions=versions)

    @property
    def category(self) -> ContextCacheCategory:
        return (
            ContextCacheCategory.PRIVATE
            if self.scope is TurnScope.USER
            else ContextCacheCategory.HOUSEHOLD
        )


@dataclass(frozen=True, slots=True)
class ContextCacheMetrics:
    """Content-free operational metrics for one fixed cache category."""

    hits: int = 0
    misses: int = 0
    builds: int = 0
    evictions: int = 0
    build_seconds: float = 0.0
    entries: int = 0


@dataclass(slots=True)
class _MutableMetrics:
    hits: int = 0
    misses: int = 0
    builds: int = 0
    evictions: int = 0
    build_seconds: float = 0.0


class ContextArtifactCache(Generic[T]):
    """Thread-safe bounded LRU cache for immutable deterministic artifacts."""

    def __init__(self, *, max_entries: int = 128) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._store: OrderedDict[ContextCacheKey, T] = OrderedDict()
        self._metrics = {
            ContextCacheCategory.PRIVATE: _MutableMetrics(),
            ContextCacheCategory.HOUSEHOLD: _MutableMetrics(),
        }
        self._lock = Lock()

    def get_or_build(self, key: ContextCacheKey, builder: Callable[[], T]) -> T:
        """Return a cached artifact or build it without holding the cache lock."""
        with self._lock:
            if key in self._store:
                value = self._store.pop(key)
                self._store[key] = value
                self._metrics[key.category].hits += 1
                return value
            self._metrics[key.category].misses += 1

        started = time.perf_counter()
        value = builder()
        elapsed = time.perf_counter() - started

        with self._lock:
            metrics = self._metrics[key.category]
            metrics.builds += 1
            metrics.build_seconds += elapsed
            if key in self._store:
                self._store.pop(key)
            while len(self._store) >= self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                self._metrics[evicted_key.category].evictions += 1
            self._store[key] = value
        return value

    def metrics_snapshot(self) -> dict[str, ContextCacheMetrics]:
        """Return immutable category metrics without exposing cache keys or values."""
        with self._lock:
            entry_counts = dict.fromkeys(ContextCacheCategory, 0)
            for key in self._store:
                entry_counts[key.category] += 1
            return {
                category.value: ContextCacheMetrics(
                    hits=metrics.hits,
                    misses=metrics.misses,
                    builds=metrics.builds,
                    evictions=metrics.evictions,
                    build_seconds=metrics.build_seconds,
                    entries=entry_counts[category],
                )
                for category, metrics in self._metrics.items()
            }

    def clear(self) -> None:
        """Drop cached values and reset content-free metrics."""
        with self._lock:
            self._store.clear()
            self._metrics = {
                ContextCacheCategory.PRIVATE: _MutableMetrics(),
                ContextCacheCategory.HOUSEHOLD: _MutableMetrics(),
            }


__all__ = [
    "ContextArtifactCache",
    "ContextCacheCategory",
    "ContextCacheKey",
    "ContextCacheMetrics",
    "ContextCacheVersions",
]
