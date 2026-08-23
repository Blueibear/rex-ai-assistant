"""Deterministic, content-free revision snapshots for context caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rex.context.cache import ContextCacheVersions
from rex.identity import validate_user_id
from rex.runtime.turn import AuthorizationSnapshotRef, TurnScope
from rex.runtime_paths import memory_dir

if TYPE_CHECKING:
    from rex.capabilities.registry import CapabilityRegistry

CONTEXT_PROMPT_TEMPLATE_REVISION = "context-builder-v1"
_FILE_CHUNK_BYTES = 64 * 1024


def _stable_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    """Hash file contents in bounded chunks without retaining private content."""
    digest = hashlib.sha256()
    if not path.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    if not path.is_file():
        digest.update(b"not-file")
        return digest.hexdigest()
    digest.update(b"present\0")
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_FILE_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return hashlib.sha256(b"unreadable").hexdigest()
    return digest.hexdigest()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value) if isinstance(value, (int, float, bool)) else None


def _context_config_snapshot(settings: Any) -> dict[str, str | None]:
    return {
        "default_location": _optional_text(getattr(settings, "default_location", None)),
        "default_timezone": _optional_text(getattr(settings, "default_timezone", None)),
        "personality": _optional_text(getattr(settings, "personality", None)),
    }


@dataclass(frozen=True, slots=True)
class ContextCacheRequest:
    """Immutable authority/model snapshot required before cross-turn caching."""

    user_id: str | None
    scope: TurnScope
    authorization: AuthorizationSnapshotRef
    model_provider: str
    model_name: str

    def __post_init__(self) -> None:
        scope = TurnScope(self.scope)
        object.__setattr__(self, "scope", scope)
        if scope is TurnScope.USER:
            if self.user_id is None:
                raise ValueError("private context caching requires a user_id")
            object.__setattr__(self, "user_id", validate_user_id(self.user_id))
        elif self.user_id is not None:
            raise ValueError("household context caching must not carry a private user_id")
        for field_name in ("model_provider", "model_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())


def _capability_snapshot(registry: CapabilityRegistry | None) -> list[dict[str, object]]:
    if registry is None:
        from rex.capabilities.registry import get_capability_registry

        registry = get_capability_registry()
    return registry.metadata_snapshot()


def build_context_cache_versions(
    request: ContextCacheRequest,
    settings: Any,
    capability_registry: CapabilityRegistry | None = None,
    *,
    source_policy_revision: str | None = None,
) -> ContextCacheVersions:
    """Build deterministic cache revisions without exposing private source data."""
    identity_source: object = {"scope": request.scope.value, "owner": "household"}
    memory_source: object = {"scope": request.scope.value, "state": "household"}

    if request.scope is TurnScope.USER:
        assert request.user_id is not None
        from rex import memory_utils

        user_id = validate_user_id(request.user_id)
        identity_profile = memory_dir() / user_id / "core.json"
        legacy_profile = Path(memory_utils.MEMORY_ROOT) / user_id / "core.json"
        facts_file = Path(memory_utils.MEMORY_ROOT) / f"{user_id}_facts.json"
        identity_source = {
            "scope": request.scope.value,
            "owner": _stable_digest(user_id),
            "profile": _file_digest(identity_profile),
        }
        memory_source = {
            "scope": request.scope.value,
            "profile": _file_digest(legacy_profile),
            "facts": _file_digest(facts_file),
        }

    if source_policy_revision is not None:
        if not isinstance(source_policy_revision, str) or not source_policy_revision.strip():
            raise ValueError("source_policy_revision must be a non-empty string")
        source_policy_revision = source_policy_revision.strip()

    return ContextCacheVersions(
        identity=_stable_digest(identity_source),
        policy=_stable_digest(
            {
                "authorization": request.authorization.policy_ref,
                "context_sources": source_policy_revision or "unconfigured",
            }
        ),
        permission=_stable_digest(request.authorization.permission_ref),
        model=_stable_digest({"provider": request.model_provider, "model": request.model_name}),
        capability=_stable_digest(_capability_snapshot(capability_registry)),
        config=_stable_digest(_context_config_snapshot(settings)),
        memory=_stable_digest(memory_source),
        prompt_template=_stable_digest(CONTEXT_PROMPT_TEMPLATE_REVISION),
    )


__all__ = [
    "CONTEXT_PROMPT_TEMPLATE_REVISION",
    "ContextCacheRequest",
    "build_context_cache_versions",
]
