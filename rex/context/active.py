"""Typed, bounded, user-scoped references for conversational follow-ups."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import TypeAlias

from rex.context.source_policy import ContextSourcePolicyStore
from rex.identity import validate_user_id

_DOMAIN_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,255}$")
_PAYLOAD_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,199}$")
_MAX_PAYLOAD_ITEMS = 12
_MAX_PAYLOAD_STRING = 256
_MAX_SOURCE_IDS = 8
_MAX_REFS_PER_USER = 32

ScalarValue: TypeAlias = str | int | float | bool | None

_DOMAIN_HINTS: dict[str, frozenset[str]] = {
    "media": frozenset({"music", "song", "track", "media", "speaker", "volume"}),
    "timekeeping": frozenset({"timer", "timers", "alarm", "alarms", "countdown", "snooze"}),
    "document": frozenset({"document", "file", "upload", "attachment"}),
    "location": frozenset({"location", "where", "nearby"}),
}
_REFERENTIAL_WORDS = frozenset({"it", "this", "that", "them", "one", "ones", "there"})
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_:@.-]+")


def _validate_domain(value: str) -> str:
    if not isinstance(value, str) or not _DOMAIN_PATTERN.fullmatch(value):
        raise ValueError("active context domain is invalid")
    return value


def _validate_key(value: str) -> str:
    if not isinstance(value, str) or not _KEY_PATTERN.fullmatch(value):
        raise ValueError("active context key is invalid")
    return value


def _validate_source_id(value: str) -> str:
    if not isinstance(value, str) or not _SOURCE_ID_PATTERN.fullmatch(value):
        raise ValueError("active context source ID is invalid")
    return value


def _validate_timestamp(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("active context expiry is invalid")
    normalized = float(value)
    if normalized < 0 or not math.isfinite(normalized):
        raise ValueError("active context expiry is invalid")
    return normalized


def _normalize_scalar(value: object) -> ScalarValue:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("active context payload is invalid")
        return value
    if isinstance(value, str) and 0 < len(value) <= _MAX_PAYLOAD_STRING:
        if any(ord(character) < 32 and character not in "\t" for character in value):
            raise ValueError("active context payload is invalid")
        return value
    raise ValueError("active context payload must contain bounded scalar values")


def _normalize_payload(payload: Mapping[str, object]) -> Mapping[str, ScalarValue]:
    if not isinstance(payload, Mapping) or len(payload) > _MAX_PAYLOAD_ITEMS:
        raise ValueError("active context payload is invalid")
    normalized: dict[str, ScalarValue] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not _PAYLOAD_KEY_PATTERN.fullmatch(key):
            raise ValueError("active context payload key is invalid")
        normalized[key] = _normalize_scalar(value)
    return MappingProxyType(normalized)


def _validate_revision(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(character.isspace() for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("active context revision is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ActiveContextRef:
    """Minimal expiring state used to resolve one user's conversational reference."""

    domain: str
    key: str
    owner_user_id: str
    payload: Mapping[str, ScalarValue]
    source_ids: tuple[str, ...]
    revision: str
    expires_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _validate_domain(self.domain))
        object.__setattr__(self, "key", _validate_key(self.key))
        object.__setattr__(self, "owner_user_id", validate_user_id(self.owner_user_id))
        object.__setattr__(self, "payload", _normalize_payload(self.payload))
        if not isinstance(self.source_ids, tuple) or len(self.source_ids) > _MAX_SOURCE_IDS:
            raise ValueError("active context source IDs are invalid")
        sources = tuple(dict.fromkeys(_validate_source_id(item) for item in self.source_ids))
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "revision", _validate_revision(self.revision))
        object.__setattr__(self, "expires_at", _validate_timestamp(self.expires_at))


@dataclass(frozen=True, slots=True)
class ReferenceResolution:
    """Outcome of resolving an utterance against one user's active references."""

    ref: ActiveContextRef | None
    reason: str
    candidates: tuple[ActiveContextRef, ...] = ()


class ActiveContextStore:
    """Thread-safe owner-partitioned active references with expiry and revocation checks."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        source_policy_store: ContextSourcePolicyStore | None = None,
        max_refs_per_user: int = _MAX_REFS_PER_USER,
    ) -> None:
        if (
            isinstance(max_refs_per_user, bool)
            or not isinstance(max_refs_per_user, int)
            or max_refs_per_user <= 0
        ):
            raise ValueError("max_refs_per_user must be positive")
        self._clock = clock
        self._source_policy_store = source_policy_store or ContextSourcePolicyStore()
        self._max_refs_per_user = max_refs_per_user
        self._refs: dict[str, dict[tuple[str, str], ActiveContextRef]] = {}
        self._lock = RLock()

    def revision_for_sources(self, user_id: str, source_ids: tuple[str, ...]) -> str:
        """Return a content-free revision for source-backed references."""
        user = validate_user_id(user_id)
        if not source_ids:
            raise ValueError("source_ids must not be empty")
        entries: list[dict[str, object]] = []
        for source_id in dict.fromkeys(_validate_source_id(item) for item in source_ids):
            policy = self._source_policy_store.get(source_id, subject_user_id=user)
            if policy is None or not self._source_policy_store.is_context_eligible(
                source_id,
                subject_user_id=user,
                requester_user_id=user,
            ):
                raise PermissionError("active context source is not currently authorized")
            entries.append(
                {
                    "source_id": source_id,
                    "policy_revision": policy.policy_revision,
                    "audience_scope": policy.audience_scope.value,
                    "context_enabled": policy.context_enabled,
                }
            )
        encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sources:" + hashlib.sha256(encoded).hexdigest()

    def _is_current(self, ref: ActiveContextRef, *, now: float) -> bool:
        if now >= ref.expires_at:
            return False
        if not ref.source_ids:
            return True
        try:
            revision = self.revision_for_sources(ref.owner_user_id, ref.source_ids)
        except (PermissionError, ValueError):
            return False
        return revision == ref.revision

    def put(self, ref: ActiveContextRef) -> None:
        """Store one bounded reference after checking any source-backed revision."""
        if not isinstance(ref, ActiveContextRef):
            raise TypeError("ref must be an ActiveContextRef")
        now = _validate_timestamp(self._clock())
        if ref.expires_at <= now:
            raise ValueError("active context reference is already expired")
        if ref.source_ids:
            current = self.revision_for_sources(ref.owner_user_id, ref.source_ids)
            if current != ref.revision:
                raise PermissionError("active context source revision is stale")
        owner = ref.owner_user_id
        with self._lock:
            partition = self._refs.setdefault(owner, {})
            partition[(ref.domain, ref.key)] = ref
            self._trim_partition(partition, now=now)

    def _trim_partition(
        self,
        partition: dict[tuple[str, str], ActiveContextRef],
        *,
        now: float,
    ) -> None:
        stale = [key for key, ref in partition.items() if not self._is_current(ref, now=now)]
        for key in stale:
            partition.pop(key, None)
        if len(partition) <= self._max_refs_per_user:
            return
        ordered = sorted(partition.items(), key=lambda item: item[1].expires_at)
        for key, _ref in ordered[: len(partition) - self._max_refs_per_user]:
            partition.pop(key, None)

    def get(self, user_id: str, domain: str, key: str) -> ActiveContextRef | None:
        """Return an unexpired current reference owned by the requesting user."""
        owner = validate_user_id(user_id)
        lookup = (_validate_domain(domain), _validate_key(key))
        now = _validate_timestamp(self._clock())
        with self._lock:
            partition = self._refs.get(owner)
            if partition is None:
                return None
            ref = partition.get(lookup)
            if ref is None:
                return None
            if not self._is_current(ref, now=now):
                partition.pop(lookup, None)
                return None
            return ref

    def list_for_user(
        self,
        user_id: str,
        *,
        domains: tuple[str, ...] | None = None,
    ) -> tuple[ActiveContextRef, ...]:
        """Return current references for one user, optionally restricted by domain."""
        owner = validate_user_id(user_id)
        allowed = None if domains is None else {_validate_domain(item) for item in domains}
        now = _validate_timestamp(self._clock())
        with self._lock:
            partition = self._refs.get(owner, {})
            self._trim_partition(partition, now=now)
            refs = tuple(
                ref for ref in partition.values() if allowed is None or ref.domain in allowed
            )
        return tuple(sorted(refs, key=lambda ref: (ref.domain, ref.key)))

    def remove(self, user_id: str, domain: str, key: str) -> bool:
        """Remove one user-owned reference without affecting other users or domains."""
        owner = validate_user_id(user_id)
        lookup = (_validate_domain(domain), _validate_key(key))
        with self._lock:
            partition = self._refs.get(owner)
            if partition is None:
                return False
            return partition.pop(lookup, None) is not None

    def invalidate_source(self, source_id: str) -> int:
        """Immediately remove every reference derived from a revoked source."""
        source = _validate_source_id(source_id)
        removed = 0
        with self._lock:
            for partition in self._refs.values():
                matching = [key for key, ref in partition.items() if source in ref.source_ids]
                for key in matching:
                    partition.pop(key, None)
                    removed += 1
        return removed

    @staticmethod
    def _tokens(utterance: str) -> tuple[str, ...]:
        if not isinstance(utterance, str):
            raise TypeError("utterance must be a string")
        return tuple(token.casefold() for token in _TOKEN_PATTERN.findall(utterance))

    @staticmethod
    def _hinted_domains(tokens: tuple[str, ...], domains: set[str]) -> set[str]:
        words = set(tokens)
        hinted: set[str] = set()
        for domain in domains:
            hints = _DOMAIN_HINTS.get(domain, frozenset())
            if words & hints:
                hinted.add(domain)
        return hinted

    @staticmethod
    def _explicit_matches(
        refs: tuple[ActiveContextRef, ...],
        utterance: str,
    ) -> tuple[ActiveContextRef, ...]:
        lowered = utterance.casefold()
        return tuple(ref for ref in refs if ref.key.casefold() in lowered)

    @staticmethod
    def _resolution_for_candidates(
        candidates: tuple[ActiveContextRef, ...],
    ) -> ReferenceResolution:
        if len(candidates) == 1:
            return ReferenceResolution(candidates[0], "resolved", candidates)
        if len(candidates) > 1:
            return ReferenceResolution(None, "ambiguous", candidates)
        return ReferenceResolution(None, "not_found", ())

    def resolve(
        self,
        user_id: str,
        utterance: str,
        candidate_domains: tuple[str, ...],
    ) -> ReferenceResolution:
        """Resolve a referential utterance without guessing across ambiguous active state."""
        owner = validate_user_id(user_id)
        domains = tuple(dict.fromkeys(_validate_domain(item) for item in candidate_domains))
        if not domains:
            return ReferenceResolution(None, "not_found", ())
        tokens = self._tokens(utterance)
        refs = self.list_for_user(owner, domains=domains)

        explicit = self._explicit_matches(refs, utterance)
        if explicit:
            return self._resolution_for_candidates(explicit)

        hinted_domains = self._hinted_domains(tokens, set(domains))
        if hinted_domains:
            hinted = tuple(ref for ref in refs if ref.domain in hinted_domains)
            return self._resolution_for_candidates(hinted)

        referential = bool(set(tokens) & _REFERENTIAL_WORDS)
        if not referential:
            return ReferenceResolution(None, "not_referential", ())
        return self._resolution_for_candidates(refs)


_DEFAULT_STORE: ActiveContextStore | None = None
_DEFAULT_STORE_LOCK = RLock()


def get_active_context_store() -> ActiveContextStore:
    """Return the process-wide canonical active-reference store."""
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = ActiveContextStore()
        return _DEFAULT_STORE


__all__ = [
    "ActiveContextRef",
    "ActiveContextStore",
    "ReferenceResolution",
    "get_active_context_store",
]
