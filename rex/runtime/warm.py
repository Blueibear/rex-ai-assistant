"""Bounded lifecycle management for heavyweight local runtime components."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any


class WarmState(StrEnum):
    """Public lifecycle states for a managed warm component."""

    COLD = "cold"
    LOADING = "loading"
    WARM = "warm"
    DEGRADED = "degraded"
    EVICTED = "evicted"
    ERROR = "error"


@dataclass(frozen=True)
class WarmComponentSpec:
    """Static lifecycle policy for one heavyweight component."""

    name: str
    loader: Callable[[], object]
    unloader: Callable[[object], None] | None = None
    fallback: Callable[[], object] | None = None
    estimated_cost_mb: float = 0.0
    idle_timeout_s: float = 0.0


@dataclass(frozen=True)
class WarmComponentStatus:
    """Content-free diagnostic state for one managed component."""

    name: str
    state: WarmState
    estimated_cost_mb: float
    load_count: int
    error_type: str | None = None
    active_leases: int = 0


@dataclass
class _WarmEntry:
    spec: WarmComponentSpec
    diagnostic_name: str
    state: WarmState = WarmState.COLD
    value: object | None = None
    last_used: float = 0.0
    load_count: int = 0
    error_type: str | None = None
    ref_count: int = 0
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock)
    use_lock: threading.Lock = field(default_factory=threading.Lock)


class WarmLease:
    def __init__(
        self,
        manager: WarmRuntimeManager,
        name: str,
        value: object,
        use_lock: threading.Lock,
        *,
        retained: bool,
        transient_unloader: Callable[[object], None] | None = None,
    ) -> None:
        self._manager = manager
        self._name = name
        self.value = value
        self._use_lock = use_lock
        self._retained = retained
        self._transient_unloader = transient_unloader
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if self._retained:
                self._manager._release(self._name)
            elif self._transient_unloader is not None:
                self._transient_unloader(self.value)
        finally:
            self._use_lock.release()

    def __enter__(self) -> object:
        return self.value

    def __exit__(self, *_args: object) -> None:
        self.release()


class WarmRuntimeManager:
    """Own heavyweight process-local components under explicit resource bounds."""

    def __init__(
        self,
        *,
        max_cost_mb: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_cost_mb < 0:
            raise ValueError("max_cost_mb must be non-negative")
        self.max_cost_mb = float(max_cost_mb)
        self._clock = clock
        self._entries: dict[str, _WarmEntry] = {}
        self._lock = threading.RLock()

    def set_budget(self, max_cost_mb: float) -> None:
        """Apply a new cache-accounting ceiling after evicting unused LRU entries."""
        if max_cost_mb < 0:
            raise ValueError("max_cost_mb must be non-negative")
        requested = float(max_cost_mb)
        while True:
            with self._lock:
                if self._accounted_cost() <= requested:
                    self.max_cost_mb = requested
                    return
                victim = self._lru_evictable_locked(exclude=None)
                if victim is None:
                    raise MemoryError("warm runtime budget cannot shrink below active leases")
                victim_name = victim.spec.name
            if not self.evict(victim_name):
                continue

    def set_idle_timeout(self, idle_timeout_s: float) -> None:
        """Apply the authoritative idle policy to all registered components."""
        if idle_timeout_s < 0:
            raise ValueError("idle_timeout_s must be non-negative")
        requested = float(idle_timeout_s)
        with self._lock:
            for entry in self._entries.values():
                entry.spec = replace(entry.spec, idle_timeout_s=requested)

    def register(self, spec: WarmComponentSpec) -> None:
        if not spec.name.strip():
            raise ValueError("warm component name must not be empty")
        if spec.estimated_cost_mb < 0 or spec.idle_timeout_s < 0:
            raise ValueError("warm component cost and idle timeout must be non-negative")
        with self._lock:
            if spec.name in self._entries:
                raise ValueError(f"warm component already registered: {spec.name}")
            self._entries[spec.name] = _WarmEntry(
                spec=spec,
                diagnostic_name=_diagnostic_component_name(spec.name),
            )

    def register_if_absent(self, spec: WarmComponentSpec) -> None:
        """Register *spec* once while rejecting conflicting resource policy."""
        with self._lock:
            existing = self._entries.get(spec.name)
            if existing is None:
                self.register(spec)
                return
            if (
                existing.spec.estimated_cost_mb != spec.estimated_cost_mb
                or existing.spec.idle_timeout_s != spec.idle_timeout_s
            ):
                raise ValueError(f"conflicting warm component policy: {spec.name}")

    def peek(self, name: str) -> object | None:
        """Return an already-loaded value without loading or changing LRU state."""
        with self._lock:
            return self._entry(name).value

    def warm(self, name: str) -> bool:
        """Load *name* only when it can remain inside the retained cache budget."""
        self.evict_idle()
        with self._lock:
            entry = self._entry(name)
            lifecycle_lock = entry.lifecycle_lock
        with lifecycle_lock:
            with self._lock:
                if self._is_resident(entry):
                    entry.last_used = self._clock()
                    return True
            if not self._reserve_load(name, entry):
                return False
            self._load_reserved_entry(entry)
            return True

    def get(self, name: str) -> object:
        """Load/return a component for lifecycle access; inference should use acquire()."""
        self.evict_idle()
        return self._get_or_load(name)

    def _get_or_load(self, name: str) -> object:
        with self._lock:
            entry = self._entry(name)
            lifecycle_lock = entry.lifecycle_lock
        with lifecycle_lock:
            with self._lock:
                if self._is_resident(entry):
                    entry.last_used = self._clock()
                    assert entry.value is not None
                    return entry.value
            if self._reserve_load(name, entry):
                return self._load_reserved_entry(entry)
            return self._load_uncached_entry(entry)

    def acquire(self, name: str) -> WarmLease:
        """Serialize component use and return a retained or transient lease."""
        # Sweep before taking this component's use lock. Calling get() after taking
        # it could make an expired component try to evict itself and deadlock.
        self.evict_idle()
        with self._lock:
            entry = self._entry(name)
            use_lock = entry.use_lock
        use_lock.acquire()
        try:
            value = self._get_or_load(name)
            with self._lock:
                retained = self._is_resident(entry) and entry.value is value
                if retained:
                    entry.ref_count += 1
            return WarmLease(
                self,
                name,
                value,
                use_lock,
                retained=retained,
                transient_unloader=None if retained else entry.spec.unloader,
            )
        except BaseException:
            use_lock.release()
            raise

    def _reserve_load(self, name: str, entry: _WarmEntry) -> bool:
        additional = entry.spec.estimated_cost_mb
        if additional > self.max_cost_mb:
            return False
        while True:
            with self._lock:
                if self._accounted_cost() + additional <= self.max_cost_mb:
                    entry.state = WarmState.LOADING
                    entry.error_type = None
                    return True
                victim = self._lru_evictable_locked(exclude=name)
                if victim is None:
                    return False
                victim_name = victim.spec.name
            self.evict(victim_name)

    def _load_reserved_entry(self, entry: _WarmEntry) -> object:
        try:
            value = entry.spec.loader()
        except Exception as exc:
            if entry.spec.fallback is None:
                with self._lock:
                    entry.load_count += 1
                    entry.error_type = type(exc).__name__
                    entry.state = WarmState.ERROR
                    entry.value = None
                raise
            try:
                value = entry.spec.fallback()
            except Exception as fallback_exc:
                with self._lock:
                    entry.load_count += 1
                    entry.error_type = type(fallback_exc).__name__
                    entry.state = WarmState.ERROR
                    entry.value = None
                raise
            with self._lock:
                entry.load_count += 1
                entry.error_type = type(exc).__name__
                entry.value = value
                entry.last_used = self._clock()
                entry.state = WarmState.DEGRADED
            return value
        with self._lock:
            entry.value = value
            entry.last_used = self._clock()
            entry.load_count += 1
            entry.error_type = None
            entry.state = WarmState.WARM
        return value

    def _load_uncached_entry(self, entry: _WarmEntry) -> object:
        try:
            value = entry.spec.loader()
        except Exception as exc:
            if entry.spec.fallback is None:
                with self._lock:
                    entry.load_count += 1
                    entry.error_type = type(exc).__name__
                    entry.state = WarmState.ERROR
                raise
            try:
                value = entry.spec.fallback()
            except Exception as fallback_exc:
                with self._lock:
                    entry.load_count += 1
                    entry.error_type = type(fallback_exc).__name__
                    entry.state = WarmState.ERROR
                raise
            with self._lock:
                entry.load_count += 1
                entry.error_type = type(exc).__name__
                entry.state = WarmState.DEGRADED
            return value
        with self._lock:
            entry.load_count += 1
            entry.error_type = None
            entry.state = WarmState.COLD
        return value

    @staticmethod
    def _is_resident(entry: _WarmEntry) -> bool:
        return entry.value is not None and entry.state in {
            WarmState.WARM,
            WarmState.DEGRADED,
        }

    def _accounted_cost(self) -> float:
        return sum(
            entry.spec.estimated_cost_mb
            for entry in self._entries.values()
            if entry.state is WarmState.LOADING or self._is_resident(entry)
        )

    def _lru_evictable_locked(self, *, exclude: str | None) -> _WarmEntry | None:
        candidates = [
            entry
            for name, entry in self._entries.items()
            if name != exclude and self._is_resident(entry) and entry.ref_count == 0
        ]
        return min(candidates, key=lambda item: item.last_used) if candidates else None

    def _release(self, name: str) -> None:
        with self._lock:
            entry = self._entry(name)
            if entry.ref_count > 0:
                entry.ref_count -= 1
                entry.last_used = self._clock()

    def status(self, name: str) -> WarmComponentStatus:
        with self._lock:
            entry = self._entry(name)
            return self._status(entry)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_cost_mb": self.max_cost_mb,
                "estimated_cost_mb": self._accounted_cost(),
                "components": [
                    self._status(entry).__dict__.copy()
                    for _, entry in sorted(self._entries.items())
                ],
            }

    def evict(self, name: str) -> bool:
        return self._evict(name)

    def _evict(self, name: str, *, idle_check_at: float | None = None) -> bool:
        with self._lock:
            entry = self._entry(name)
            if entry.ref_count > 0:
                return False
            use_lock = entry.use_lock
            lifecycle_lock = entry.lifecycle_lock
        with use_lock:
            with lifecycle_lock:
                with self._lock:
                    if entry.ref_count > 0 or not self._is_resident(entry):
                        return False
                    if idle_check_at is not None and not self._is_idle_at(entry, idle_check_at):
                        return False
                    value = entry.value
                if entry.spec.unloader is not None and value is not None:
                    try:
                        entry.spec.unloader(value)
                    except Exception as exc:
                        with self._lock:
                            entry.error_type = type(exc).__name__
                        raise
                with self._lock:
                    entry.value = None
                    entry.state = WarmState.EVICTED
                    entry.last_used = 0.0
                    entry.error_type = None
                return True

    @staticmethod
    def _is_idle_at(entry: _WarmEntry, now: float) -> bool:
        return (
            entry.spec.idle_timeout_s > 0
            and entry.ref_count == 0
            and now - entry.last_used >= entry.spec.idle_timeout_s
        )

    def evict_idle(self) -> list[str]:
        now = self._clock()
        with self._lock:
            candidates = [
                name
                for name, entry in self._entries.items()
                if self._is_resident(entry) and self._is_idle_at(entry, now)
            ]
        return [name for name in candidates if self._evict(name, idle_check_at=now)]

    def close(self) -> None:
        with self._lock:
            names = list(self._entries)
        for name in names:
            self.evict(name)

    def _entry(self, name: str) -> _WarmEntry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown warm component: {name}") from exc

    def _status(self, entry: _WarmEntry) -> WarmComponentStatus:
        active_cost = entry.spec.estimated_cost_mb if self._is_resident(entry) else 0.0
        return WarmComponentStatus(
            name=entry.diagnostic_name,
            state=entry.state,
            estimated_cost_mb=active_cost,
            load_count=entry.load_count,
            error_type=entry.error_type,
            active_leases=entry.ref_count,
        )


def _diagnostic_component_name(name: str) -> str:
    """Return a content-free identifier even when callers supply an unsafe name."""
    kind, separator, digest = name.partition(":")
    if (
        separator
        and kind.replace("_", "").replace("-", "").isalnum()
        and len(digest) == 12
        and all(character in "0123456789abcdef" for character in digest.lower())
    ):
        return name
    safe_digest = hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"component:{safe_digest}"


_GLOBAL_LOCK = threading.RLock()
_GLOBAL_POLICY_CONDITION = threading.Condition(_GLOBAL_LOCK)
_GLOBAL_MANAGER: WarmRuntimeManager | None = None
_DEFAULT_MAX_COST_MB = 6144.0
_DEFAULT_IDLE_TIMEOUT_S = 900.0
_GLOBAL_IDLE_TIMEOUT_S = _DEFAULT_IDLE_TIMEOUT_S
_GLOBAL_POLICY_ESTABLISHED = False
_GLOBAL_POLICY_CONFIGURING = False


def _numeric_setting(settings: object | None, name: str, default: float) -> float:
    if settings is None:
        return default
    value = getattr(settings, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(0.0, float(value))


def warm_component_key(kind: str, *parts: object) -> str:
    """Return a stable diagnostic-safe key without exposing local paths/model IDs."""
    material = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{kind}:{digest}"


def _apply_global_policy(manager: WarmRuntimeManager, settings: object) -> float:
    requested_cost = _numeric_setting(settings, "warm_runtime_max_cost_mb", _DEFAULT_MAX_COST_MB)
    requested_idle = _numeric_setting(
        settings, "warm_runtime_idle_timeout_s", _DEFAULT_IDLE_TIMEOUT_S
    )
    if manager.max_cost_mb != requested_cost:
        manager.set_budget(requested_cost)
    manager.set_idle_timeout(requested_idle)
    return requested_idle


def get_global_warm_runtime(settings: object | None = None) -> WarmRuntimeManager:
    """Return the process manager; the first settings-bearing access establishes policy."""
    global _GLOBAL_IDLE_TIMEOUT_S, _GLOBAL_MANAGER
    global _GLOBAL_POLICY_CONFIGURING, _GLOBAL_POLICY_ESTABLISHED

    with _GLOBAL_POLICY_CONDITION:
        while _GLOBAL_POLICY_CONFIGURING:
            _GLOBAL_POLICY_CONDITION.wait()
        if _GLOBAL_MANAGER is None:
            requested_cost = _numeric_setting(
                settings, "warm_runtime_max_cost_mb", _DEFAULT_MAX_COST_MB
            )
            _GLOBAL_IDLE_TIMEOUT_S = _numeric_setting(
                settings, "warm_runtime_idle_timeout_s", _DEFAULT_IDLE_TIMEOUT_S
            )
            _GLOBAL_MANAGER = WarmRuntimeManager(max_cost_mb=requested_cost)
            _GLOBAL_POLICY_ESTABLISHED = settings is not None
            return _GLOBAL_MANAGER
        if settings is None or _GLOBAL_POLICY_ESTABLISHED:
            return _GLOBAL_MANAGER
        # A caller may deliberately tune a provisional manager directly (for
        # tests/operator control) before an application wrapper is built. Treat
        # a non-default budget as established rather than silently replacing it.
        if _GLOBAL_MANAGER.max_cost_mb != _DEFAULT_MAX_COST_MB:
            _GLOBAL_POLICY_ESTABLISHED = True
            return _GLOBAL_MANAGER
        manager = _GLOBAL_MANAGER
        _GLOBAL_POLICY_CONFIGURING = True

    try:
        requested_idle = _apply_global_policy(manager, settings)
    except BaseException:
        with _GLOBAL_POLICY_CONDITION:
            _GLOBAL_POLICY_CONFIGURING = False
            _GLOBAL_POLICY_CONDITION.notify_all()
        raise

    with _GLOBAL_POLICY_CONDITION:
        _GLOBAL_IDLE_TIMEOUT_S = requested_idle
        _GLOBAL_POLICY_ESTABLISHED = True
        _GLOBAL_POLICY_CONFIGURING = False
        _GLOBAL_POLICY_CONDITION.notify_all()
        return manager


def configure_global_warm_runtime(settings: object) -> WarmRuntimeManager:
    """Explicitly replace application resource policy on the process-local manager."""
    global _GLOBAL_IDLE_TIMEOUT_S, _GLOBAL_POLICY_CONFIGURING, _GLOBAL_POLICY_ESTABLISHED
    manager = get_global_warm_runtime(settings)
    with _GLOBAL_POLICY_CONDITION:
        while _GLOBAL_POLICY_CONFIGURING:
            _GLOBAL_POLICY_CONDITION.wait()
        _GLOBAL_POLICY_CONFIGURING = True
    try:
        requested_idle = _apply_global_policy(manager, settings)
    except BaseException:
        with _GLOBAL_POLICY_CONDITION:
            _GLOBAL_POLICY_CONFIGURING = False
            _GLOBAL_POLICY_CONDITION.notify_all()
        raise
    with _GLOBAL_POLICY_CONDITION:
        _GLOBAL_IDLE_TIMEOUT_S = requested_idle
        _GLOBAL_POLICY_ESTABLISHED = True
        _GLOBAL_POLICY_CONFIGURING = False
        _GLOBAL_POLICY_CONDITION.notify_all()
    return manager


def default_idle_timeout(settings: object | None = None) -> float:
    if settings is not None:
        return _numeric_setting(settings, "warm_runtime_idle_timeout_s", _DEFAULT_IDLE_TIMEOUT_S)
    with _GLOBAL_LOCK:
        return _GLOBAL_IDLE_TIMEOUT_S


def reset_global_warm_runtime() -> None:
    """Close and clear the process-local manager (tests and controlled shutdown only)."""
    global _GLOBAL_IDLE_TIMEOUT_S, _GLOBAL_MANAGER
    global _GLOBAL_POLICY_CONFIGURING, _GLOBAL_POLICY_ESTABLISHED

    with _GLOBAL_POLICY_CONDITION:
        while _GLOBAL_POLICY_CONFIGURING:
            _GLOBAL_POLICY_CONDITION.wait()
        manager = _GLOBAL_MANAGER
        _GLOBAL_MANAGER = None
        _GLOBAL_IDLE_TIMEOUT_S = _DEFAULT_IDLE_TIMEOUT_S
        _GLOBAL_POLICY_ESTABLISHED = False
        _GLOBAL_POLICY_CONFIGURING = False
        _GLOBAL_POLICY_CONDITION.notify_all()
    if manager is not None:
        manager.close()


__all__ = [
    "WarmComponentSpec",
    "WarmComponentStatus",
    "WarmLease",
    "WarmRuntimeManager",
    "WarmState",
    "configure_global_warm_runtime",
    "default_idle_timeout",
    "get_global_warm_runtime",
    "reset_global_warm_runtime",
    "warm_component_key",
]
