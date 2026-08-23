"""User-scoped contextual-source policy and content-free revisions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from rex.identity import validate_user_id
from rex.runtime_paths import household_data_path, user_data_path

_SCHEMA_VERSION = 1
_SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,199}$")
_PARTITION_LOCKS_GUARD = threading.Lock()
_PARTITION_LOCKS: dict[Path, threading.RLock] = {}


class ContextSourceType(StrEnum):
    """Canonical classes of data that may contribute situational context."""

    INTEGRATION = "integration"
    UPLOAD = "upload"
    LOCATION = "location"
    MEMORY = "memory"
    CAPABILITY = "capability"


class AudienceScope(StrEnum):
    """Who may receive a source as contextual input."""

    PRIVATE = "private"
    HOUSEHOLD = "household"


class DisclosurePolicy(StrEnum):
    """Separate disclosure boundary for source-derived information."""

    OWNER_ONLY = "owner_only"
    HOUSEHOLD = "household"
    EXPLICIT_GRANT = "explicit_grant"


def _validate_source_id(source_id: str) -> str:
    if not isinstance(source_id, str) or not _SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError("Context source ID is invalid")
    return source_id


def _partition_lock(path: Path) -> threading.RLock:
    resolved = path.resolve(strict=False)
    with _PARTITION_LOCKS_GUARD:
        lock = _PARTITION_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _PARTITION_LOCKS[resolved] = lock
        return lock


@dataclass(frozen=True, slots=True)
class ContextSourcePolicy:
    """Persisted policy metadata for one contextual source."""

    source_id: str
    source_type: ContextSourceType
    owner_user_id: str | None
    audience_scope: AudienceScope
    context_enabled: bool
    disclosure_policy: DisclosurePolicy
    policy_revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _validate_source_id(self.source_id))
        object.__setattr__(self, "source_type", ContextSourceType(self.source_type))
        if self.owner_user_id is not None:
            object.__setattr__(self, "owner_user_id", validate_user_id(self.owner_user_id))
        object.__setattr__(self, "audience_scope", AudienceScope(self.audience_scope))
        object.__setattr__(self, "disclosure_policy", DisclosurePolicy(self.disclosure_policy))
        if not isinstance(self.context_enabled, bool):
            raise ValueError("context_enabled must be boolean")
        if isinstance(self.policy_revision, bool) or not isinstance(self.policy_revision, int):
            raise ValueError("policy_revision must be an integer")
        if self.policy_revision < 0:
            raise ValueError("policy_revision must not be negative")
        if self.owner_user_id is None and self.audience_scope is AudienceScope.PRIVATE:
            raise ValueError("private context sources require an owner")


def _default_policy_values(
    source_type: ContextSourceType,
    audience_scope: AudienceScope | None,
    context_enabled: bool | None,
    disclosure_policy: DisclosurePolicy | None,
) -> tuple[AudienceScope, bool, DisclosurePolicy]:
    source_type = ContextSourceType(source_type)
    audience = audience_scope or AudienceScope.PRIVATE
    if context_enabled is None:
        enabled = source_type not in {ContextSourceType.UPLOAD, ContextSourceType.LOCATION}
    else:
        enabled = context_enabled
    if disclosure_policy is None:
        if source_type is ContextSourceType.LOCATION:
            disclosure = DisclosurePolicy.EXPLICIT_GRANT
        elif audience is AudienceScope.HOUSEHOLD:
            disclosure = DisclosurePolicy.HOUSEHOLD
        else:
            disclosure = DisclosurePolicy.OWNER_ONLY
    else:
        disclosure = DisclosurePolicy(disclosure_policy)
    return audience, enabled, disclosure


def _policy_to_dict(policy: ContextSourcePolicy) -> dict[str, Any]:
    return {
        "source_id": policy.source_id,
        "source_type": policy.source_type.value,
        "owner_user_id": policy.owner_user_id,
        "audience_scope": policy.audience_scope.value,
        "context_enabled": policy.context_enabled,
        "disclosure_policy": policy.disclosure_policy.value,
        "policy_revision": policy.policy_revision,
    }


def _policy_from_dict(payload: object) -> ContextSourcePolicy:
    if not isinstance(payload, dict):
        raise ValueError("Context source policy entry is malformed")
    expected = {
        "source_id",
        "source_type",
        "owner_user_id",
        "audience_scope",
        "context_enabled",
        "disclosure_policy",
        "policy_revision",
    }
    if set(payload) != expected:
        raise ValueError("Context source policy entry has invalid schema")
    return ContextSourcePolicy(
        source_id=payload["source_id"],
        source_type=payload["source_type"],
        owner_user_id=payload["owner_user_id"],
        audience_scope=payload["audience_scope"],
        context_enabled=payload["context_enabled"],
        disclosure_policy=payload["disclosure_policy"],
        policy_revision=payload["policy_revision"],
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ContextSourcePolicyStore:
    """Persist contextual-use policy separately from source contents."""

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else None

    def _path(self, partition_user_id: str | None) -> Path:
        if partition_user_id is None:
            if self._root is not None:
                return self._root / "household" / "context" / "source_policy.json"
            return household_data_path("context", "source_policy.json")
        owner = validate_user_id(partition_user_id)
        if self._root is not None:
            return self._root / owner / "context" / "source_policy.json"
        return user_data_path(owner, "context", "source_policy.json")

    @staticmethod
    def _partition_user_for_policy(policy: ContextSourcePolicy) -> str | None:
        """Choose the revision partition from audience, not source ownership."""
        if policy.audience_scope is AudienceScope.HOUSEHOLD:
            return None
        if policy.owner_user_id is None:
            raise ValueError("private context sources require an owner")
        return policy.owner_user_id

    def _read_partition(
        self, partition_user_id: str | None
    ) -> tuple[int, tuple[ContextSourcePolicy, ...]]:
        path = self._path(partition_user_id)
        if not path.exists():
            return 0, ()
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Context source policy store is unreadable") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"version", "owner_user_id", "revision", "policies"}
            or payload.get("version") != _SCHEMA_VERSION
            or payload.get("owner_user_id") != partition_user_id
            or isinstance(payload.get("revision"), bool)
            or not isinstance(payload.get("revision"), int)
            or payload["revision"] < 0
            or not isinstance(payload.get("policies"), list)
        ):
            raise ValueError("Context source policy store has invalid ownership or schema")
        policies = tuple(_policy_from_dict(item) for item in payload["policies"])
        source_ids: set[str] = set()
        for policy in policies:
            if partition_user_id is None:
                if policy.audience_scope is not AudienceScope.HOUSEHOLD:
                    raise ValueError("Household policy partition contains a private source")
            elif (
                policy.owner_user_id != partition_user_id
                or policy.audience_scope is not AudienceScope.PRIVATE
            ):
                raise ValueError("Private policy partition has invalid ownership or audience")
            if policy.source_id in source_ids:
                raise ValueError("Context source policies must have unique source IDs")
            source_ids.add(policy.source_id)
        return payload["revision"], policies

    def _write_partition(
        self,
        partition_user_id: str | None,
        revision: int,
        policies: tuple[ContextSourcePolicy, ...],
    ) -> None:
        payload = {
            "version": _SCHEMA_VERSION,
            # Schema v1 keeps this key for compatibility; the value identifies
            # the revision partition, not necessarily the source's true owner.
            "owner_user_id": partition_user_id,
            "revision": revision,
            "policies": [_policy_to_dict(policy) for policy in policies],
        }
        _atomic_write(self._path(partition_user_id), payload)

    def _remove_stale_copy(
        self,
        partition_user_id: str | None,
        *,
        source_id: str,
        owner_user_id: str | None,
    ) -> None:
        path = self._path(partition_user_id)
        with _partition_lock(path):
            revision, policies = self._read_partition(partition_user_id)
            remaining = tuple(
                item
                for item in policies
                if not (item.source_id == source_id and item.owner_user_id == owner_user_id)
            )
            if len(remaining) == len(policies):
                return
            self._write_partition(partition_user_id, revision + 1, remaining)

    def put(self, policy: ContextSourcePolicy) -> ContextSourcePolicy:
        """Create or replace a policy in the partition implied by its audience."""
        if not isinstance(policy, ContextSourcePolicy):
            raise TypeError("policy must be a ContextSourcePolicy")
        partition_user = self._partition_user_for_policy(policy)
        opposite_partition = None if partition_user is not None else policy.owner_user_id
        if opposite_partition != partition_user:
            self._remove_stale_copy(
                opposite_partition,
                source_id=policy.source_id,
                owner_user_id=policy.owner_user_id,
            )
        path = self._path(partition_user)
        with _partition_lock(path):
            revision, existing = self._read_partition(partition_user)
            new_revision = revision + 1
            saved = replace(policy, policy_revision=new_revision)
            policies = tuple(item for item in existing if item.source_id != saved.source_id) + (
                saved,
            )
            policies = tuple(sorted(policies, key=lambda item: item.source_id))
            self._write_partition(partition_user, new_revision, policies)
        return saved

    def register_source(
        self,
        source_id: str,
        source_type: ContextSourceType,
        *,
        owner_user_id: str | None,
        audience_scope: AudienceScope | None = None,
        context_enabled: bool | None = None,
        disclosure_policy: DisclosurePolicy | None = None,
    ) -> ContextSourcePolicy:
        """Register source metadata using privacy-preserving type defaults."""
        source_type = ContextSourceType(source_type)
        if owner_user_id is not None:
            owner_user_id = validate_user_id(owner_user_id)
        elif audience_scope is AudienceScope.PRIVATE:
            raise ValueError("private context sources require an owner")
        elif audience_scope is None:
            audience_scope = AudienceScope.HOUSEHOLD
        audience, enabled, disclosure = _default_policy_values(
            source_type,
            audience_scope,
            context_enabled,
            disclosure_policy,
        )
        return self.put(
            ContextSourcePolicy(
                source_id=_validate_source_id(source_id),
                source_type=source_type,
                owner_user_id=owner_user_id,
                audience_scope=audience,
                context_enabled=enabled,
                disclosure_policy=disclosure,
            )
        )

    def _get_from_partition(
        self, source_id: str, owner_user_id: str | None
    ) -> ContextSourcePolicy | None:
        _revision, policies = self._read_partition(owner_user_id)
        return next((policy for policy in policies if policy.source_id == source_id), None)

    def get(self, source_id: str, *, subject_user_id: str) -> ContextSourcePolicy | None:
        """Return policy metadata relevant to one subject, never source content."""
        source_id = _validate_source_id(source_id)
        subject = validate_user_id(subject_user_id)
        policy = self._get_from_partition(source_id, subject)
        if policy is not None:
            return policy
        return self._get_from_partition(source_id, None)

    def list_for_user(self, user_id: str) -> tuple[ContextSourcePolicy, ...]:
        """Return content-free private and household policy metadata visible to one user."""
        user = validate_user_id(user_id)
        _private_revision, private = self._read_partition(user)
        _household_revision, household = self._read_partition(None)
        policies = tuple(private) + tuple(household)
        return tuple(
            sorted(
                policies,
                key=lambda policy: (
                    policy.source_id,
                    policy.owner_user_id or "",
                    policy.audience_scope.value,
                ),
            )
        )

    def _set_context_enabled_in_partition(
        self,
        partition_user_id: str | None,
        *,
        owner_user_id: str,
        source_id: str,
        enabled: bool,
    ) -> ContextSourcePolicy | None:
        path = self._path(partition_user_id)
        with _partition_lock(path):
            revision, policies = self._read_partition(partition_user_id)
            current = next(
                (
                    item
                    for item in policies
                    if item.source_id == source_id and item.owner_user_id == owner_user_id
                ),
                None,
            )
            if current is None:
                return None
            new_revision = revision + 1
            updated = replace(current, context_enabled=enabled, policy_revision=new_revision)
            new_policies = tuple(
                updated if item.source_id == source_id else item for item in policies
            )
            self._write_partition(partition_user_id, new_revision, new_policies)
            return updated

    def set_context_enabled(
        self,
        owner_user_id: str,
        source_id: str,
        enabled: bool,
    ) -> ContextSourcePolicy:
        """Change contextual eligibility for an owner's private or shared source."""
        owner = validate_user_id(owner_user_id)
        source_id = _validate_source_id(source_id)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        for partition_user in (owner, None):
            updated = self._set_context_enabled_in_partition(
                partition_user,
                owner_user_id=owner,
                source_id=source_id,
                enabled=enabled,
            )
            if updated is not None:
                return updated
        raise KeyError("Context source policy was not found")

    def is_context_eligible(
        self,
        source_id: str,
        *,
        subject_user_id: str,
        requester_user_id: str,
    ) -> bool:
        """Check contextual-use authority before any source retrieval."""
        subject = validate_user_id(subject_user_id)
        requester = validate_user_id(requester_user_id)
        policy = self.get(source_id, subject_user_id=subject)
        if policy is None or not policy.context_enabled:
            return False
        if policy.audience_scope is AudienceScope.PRIVATE:
            return policy.owner_user_id == requester == subject
        # Household contextual eligibility is audience-wide. The true owner is
        # retained for disclosure/mutation checks but does not make the shared
        # source private to that owner.
        return policy.audience_scope is AudienceScope.HOUSEHOLD

    def revision_for_user(self, user_id: str) -> str:
        """Return a content-free token covering private and household policy state."""
        user = validate_user_id(user_id)
        private_revision, _private = self._read_partition(user)
        household_revision, _household = self._read_partition(None)
        payload = json.dumps(
            {
                "private_revision": private_revision,
                "household_revision": household_revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AudienceScope",
    "ContextSourcePolicy",
    "ContextSourcePolicyStore",
    "ContextSourceType",
    "DisclosurePolicy",
]
