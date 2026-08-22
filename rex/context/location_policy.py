"""Owner-scoped location assistance and person-specific sharing policy."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from rex.context.source_policy import (
    AudienceScope,
    ContextSourcePolicyStore,
    ContextSourceType,
    DisclosurePolicy,
)
from rex.identity import validate_user_id
from rex.runtime_paths import user_data_path

_SCHEMA_VERSION = 1
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[Path, threading.RLock] = {}


class LocationUsePurpose(StrEnum):
    """Bounded reasons Rex may request an authorized user's location."""

    TOOL_CONTEXT = "tool_context"
    EXPLICIT_REQUEST = "explicit_request"
    PROACTIVE_RULE = "proactive_rule"
    DISCLOSURE = "disclosure"


@dataclass(frozen=True, slots=True)
class LocationGrants:
    """Persisted grants owned and mutable only by one Rex user."""

    owner_user_id: str
    revision: int = 0
    location_assist: bool = False
    shared_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_user_id", validate_user_id(self.owner_user_id))
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("location grant revision is invalid")
        if not isinstance(self.location_assist, bool):
            raise ValueError("location_assist must be boolean")
        recipients = tuple(sorted({validate_user_id(item) for item in self.shared_with}))
        object.__setattr__(self, "shared_with", recipients)


@dataclass(frozen=True, slots=True)
class PrivateLocation:
    """Recent private location observation for one authenticated user."""

    city: str | None = None
    timezone: str | None = None
    lat: float | None = None
    lon: float | None = None

    def __post_init__(self) -> None:
        if self.city is not None and (not isinstance(self.city, str) or not self.city.strip()):
            raise ValueError("city must be a non-empty string")
        if self.timezone is not None and (
            not isinstance(self.timezone, str) or not self.timezone.strip()
        ):
            raise ValueError("timezone must be a non-empty string")
        if self.lat is not None and not isinstance(self.lat, (int, float)):
            raise ValueError("lat must be numeric")
        if self.lon is not None and not isinstance(self.lon, (int, float)):
            raise ValueError("lon must be numeric")
        if self.city is None and self.timezone is None and self.lat is None and self.lon is None:
            raise ValueError("location observation must contain at least one value")


@dataclass(frozen=True, slots=True)
class LocationDisclosureResult:
    """Disclosure result that never exposes location on denied paths."""

    allowed: bool
    location: PrivateLocation | None
    message: str


def _lock_for(path: Path) -> threading.RLock:
    resolved = path.resolve(strict=False)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[resolved] = lock
        return lock


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
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


class LocationGrantStore:
    """Persist owner-controlled assistance and recipient-specific share grants."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        source_policy_store: ContextSourcePolicyStore | None = None,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._source_policy_store = source_policy_store or ContextSourcePolicyStore()

    def _path(self, owner_user_id: str) -> Path:
        owner = validate_user_id(owner_user_id)
        if self._root is not None:
            return self._root / owner / "context" / "location_grants.json"
        return user_data_path(owner, "context", "location_grants.json")

    def _read(self, owner_user_id: str) -> LocationGrants:
        owner = validate_user_id(owner_user_id)
        path = self._path(owner)
        if not path.exists():
            return LocationGrants(owner_user_id=owner)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Location grant store is unreadable") from exc
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"version", "owner_user_id", "revision", "location_assist", "shared_with"}
            or payload.get("version") != _SCHEMA_VERSION
            or payload.get("owner_user_id") != owner
            or not isinstance(payload.get("shared_with"), list)
        ):
            raise ValueError("Location grant store has invalid ownership or schema")
        return LocationGrants(
            owner_user_id=payload["owner_user_id"],
            revision=payload["revision"],
            location_assist=payload["location_assist"],
            shared_with=tuple(payload["shared_with"]),
        )

    def _write(self, grants: LocationGrants) -> None:
        payload: dict[str, object] = {
            "version": _SCHEMA_VERSION,
            "owner_user_id": grants.owner_user_id,
            "revision": grants.revision,
            "location_assist": grants.location_assist,
            "shared_with": list(grants.shared_with),
        }
        _atomic_write(self._path(grants.owner_user_id), payload)

    @staticmethod
    def _require_owner(owner_user_id: str, actor_user_id: str) -> str:
        owner = validate_user_id(owner_user_id)
        actor = validate_user_id(actor_user_id)
        if owner != actor:
            raise PermissionError("owner authorization required")
        return owner

    def _sync_source_policy(self, grants: LocationGrants) -> None:
        self._source_policy_store.register_source(
            f"location:{grants.owner_user_id}",
            ContextSourceType.LOCATION,
            owner_user_id=grants.owner_user_id,
            audience_scope=AudienceScope.PRIVATE,
            context_enabled=grants.location_assist,
            disclosure_policy=DisclosurePolicy.EXPLICIT_GRANT,
        )

    def get(self, owner_user_id: str) -> LocationGrants:
        """Return content-free grants for one owner."""
        owner = validate_user_id(owner_user_id)
        path = self._path(owner)
        with _lock_for(path):
            return self._read(owner)

    def set_assist(
        self,
        *,
        owner_user_id: str,
        enabled: bool,
        actor_user_id: str,
    ) -> LocationGrants:
        """Set private location assistance; only the owner may mutate it."""
        owner = self._require_owner(owner_user_id, actor_user_id)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        path = self._path(owner)
        with _lock_for(path):
            current = self._read(owner)
            saved = replace(current, revision=current.revision + 1, location_assist=enabled)
            self._write(saved)
        self._sync_source_policy(saved)
        return saved

    def set_share(
        self,
        *,
        owner_user_id: str,
        recipient_user_id: str,
        enabled: bool,
        actor_user_id: str,
    ) -> LocationGrants:
        """Set one recipient-specific disclosure grant; only the owner may mutate it."""
        owner = self._require_owner(owner_user_id, actor_user_id)
        recipient = validate_user_id(recipient_user_id)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        path = self._path(owner)
        with _lock_for(path):
            current = self._read(owner)
            recipients = set(current.shared_with)
            if enabled:
                recipients.add(recipient)
            else:
                recipients.discard(recipient)
            saved = replace(
                current,
                revision=current.revision + 1,
                shared_with=tuple(sorted(recipients)),
            )
            self._write(saved)
        self._sync_source_policy(saved)
        return saved

    def is_assist_enabled(self, owner_user_id: str) -> bool:
        return self.get(owner_user_id).location_assist

    def can_share(self, owner_user_id: str, recipient_user_id: str) -> bool:
        recipient = validate_user_id(recipient_user_id)
        return recipient in self.get(owner_user_id).shared_with


LocationProvider = Callable[[str, LocationUsePurpose], PrivateLocation | None]


class LocationContextService:
    """Resolve private location only after the owner's grants permit the use."""

    def __init__(
        self,
        *,
        grant_store: LocationGrantStore,
        location_provider: LocationProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_location_age_seconds: float = 300.0,
    ) -> None:
        if max_location_age_seconds <= 0:
            raise ValueError("max_location_age_seconds must be positive")
        self._grant_store = grant_store
        self._location_provider = location_provider
        self._clock = clock
        self._max_location_age_seconds = float(max_location_age_seconds)
        self._recent: dict[str, tuple[float, PrivateLocation]] = {}

    def seed_private_location(
        self,
        user_id: str,
        *,
        city: str | None = None,
        timezone: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> PrivateLocation:
        """Accept a trusted recent observation for one authenticated user."""
        user = validate_user_id(user_id)
        location = PrivateLocation(city=city, timezone=timezone, lat=lat, lon=lon)
        self._recent[user] = (self._clock(), location)
        return location

    def _resolve_authorized(
        self,
        user_id: str,
        purpose: LocationUsePurpose,
    ) -> PrivateLocation | None:
        cached = self._recent.get(user_id)
        if cached is not None:
            observed_at, cached_location = cached
            if self._clock() - observed_at <= self._max_location_age_seconds:
                return cached_location
            self._recent.pop(user_id, None)
        if self._location_provider is None:
            return None
        provider_location = self._location_provider(user_id, purpose)
        if provider_location is not None and not isinstance(provider_location, PrivateLocation):
            raise TypeError("location provider must return PrivateLocation or None")
        if provider_location is not None:
            self._recent[user_id] = (self._clock(), provider_location)
        return provider_location

    def get_for_assistance(
        self,
        user_id: str,
        purpose: LocationUsePurpose,
    ) -> PrivateLocation | None:
        """Return location for the owner only after explicit assistance opt-in."""
        user = validate_user_id(user_id)
        purpose = LocationUsePurpose(purpose)
        if not self._grant_store.is_assist_enabled(user):
            return None
        return self._resolve_authorized(user, purpose)

    def get_for_disclosure(
        self,
        *,
        subject_user_id: str,
        requester_user_id: str,
    ) -> LocationDisclosureResult:
        """Disclose only with owner assistance plus explicit recipient sharing."""
        subject = validate_user_id(subject_user_id)
        requester = validate_user_id(requester_user_id)
        denied_message = f"I can't share {subject.capitalize()}'s location."

        if requester != subject and not self._grant_store.can_share(subject, requester):
            return LocationDisclosureResult(False, None, denied_message)
        if not self._grant_store.is_assist_enabled(subject):
            return LocationDisclosureResult(False, None, denied_message)

        location = self._resolve_authorized(subject, LocationUsePurpose.DISCLOSURE)
        if location is None:
            return LocationDisclosureResult(
                True,
                None,
                f"I don't have a recent location for {subject.capitalize()}.",
            )
        return LocationDisclosureResult(True, location, "")


__all__ = [
    "LocationContextService",
    "LocationDisclosureResult",
    "LocationGrantStore",
    "LocationGrants",
    "LocationProvider",
    "LocationUsePurpose",
    "PrivateLocation",
]
