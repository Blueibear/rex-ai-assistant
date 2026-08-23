"""Owner-bound context/privacy settings shared by desktop and mobile adapters."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rex.context.location_policy import LocationGrantStore
from rex.context.source_policy import ContextSourcePolicyStore
from rex.identity import validate_user_id
from rex.runtime_paths import user_data_path

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ContextPrivacyPreferences:
    owner_user_id: str
    proactive_assistance: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_user_id", validate_user_id(self.owner_user_id))
        if not isinstance(self.proactive_assistance, bool):
            raise ValueError("proactive_assistance must be boolean")


class ContextPrivacyPreferenceStore:
    """Persist one user's proactive-assistance preference with owner-only mutation."""

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else None
        self._lock = threading.RLock()

    def _path(self, owner_user_id: str) -> Path:
        owner = validate_user_id(owner_user_id)
        if self._root is not None:
            return self._root / owner / "context_preferences.json"
        return user_data_path(owner, "context", "preferences.json")

    def get(self, owner_user_id: str) -> ContextPrivacyPreferences:
        owner = validate_user_id(owner_user_id)
        path = self._path(owner)
        with self._lock:
            if not path.exists():
                return ContextPrivacyPreferences(owner)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Context privacy preferences are unreadable") from exc
        if not isinstance(payload, dict) or payload.get("version") != _SCHEMA_VERSION:
            raise ValueError("Context privacy preferences have invalid schema")
        if payload.get("owner_user_id") != owner:
            raise ValueError("Context privacy preferences have invalid ownership")
        enabled = payload.get("proactive_assistance")
        if not isinstance(enabled, bool):
            raise ValueError("Context privacy preferences have invalid schema")
        return ContextPrivacyPreferences(owner, enabled)

    def set_proactive_assistance(
        self,
        *,
        owner_user_id: str,
        enabled: bool,
        actor_user_id: str,
    ) -> ContextPrivacyPreferences:
        owner = validate_user_id(owner_user_id)
        actor = validate_user_id(actor_user_id)
        if owner != actor:
            raise PermissionError("owner authorization required")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        saved = replace(self.get(owner), proactive_assistance=enabled)
        path = self._path(owner)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SCHEMA_VERSION,
            "owner_user_id": owner,
            "proactive_assistance": saved.proactive_assistance,
        }
        temporary: Path | None = None
        with self._lock:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(payload, handle, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return saved


class ContextPrivacyService:
    """Canonical owner-authority boundary for context/privacy settings."""

    def __init__(
        self,
        *,
        source_policy_store: ContextSourcePolicyStore,
        location_grant_store: LocationGrantStore,
        knowledge_base: Any,
        preference_store: ContextPrivacyPreferenceStore,
    ) -> None:
        self.source_policy_store = source_policy_store
        self.location_grant_store = location_grant_store
        self.knowledge_base = knowledge_base
        self.preference_store = preference_store

    @staticmethod
    def _require_owner(owner_user_id: str, actor_user_id: str) -> str:
        owner = validate_user_id(owner_user_id)
        actor = validate_user_id(actor_user_id)
        if owner != actor:
            raise PermissionError("owner authorization required")
        return owner

    @staticmethod
    def _source_payload(policy: Any, actor_user_id: str) -> dict[str, Any]:
        return {
            "source_id": policy.source_id,
            "source_type": policy.source_type.value,
            "audience_scope": policy.audience_scope.value,
            "context_enabled": policy.context_enabled,
            "disclosure_policy": policy.disclosure_policy.value,
            "mutable": policy.owner_user_id == actor_user_id,
        }

    @staticmethod
    def _upload_payload(doc: Any) -> dict[str, Any]:
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "audience_scope": doc.audience_scope,
            "context_enabled": doc.context_enabled,
        }

    def get_state(self, user_id: str) -> dict[str, Any]:
        user = validate_user_id(user_id)
        sources = [
            self._source_payload(policy, user)
            for policy in self.source_policy_store.list_for_user(user)
        ]
        uploads = [
            self._upload_payload(doc)
            for doc in self.knowledge_base.list_documents_for_user(
                user,
                context_only=False,
                limit=100,
            )
            if doc.owner_user_id == user
        ]
        grants = self.location_grant_store.get(user)
        preferences = self.preference_store.get(user)
        return {
            "sources": sources,
            "uploads": uploads,
            "location": {
                "location_assist": grants.location_assist,
                "shared_with": list(grants.shared_with),
            },
            "proactive_assistance": preferences.proactive_assistance,
        }

    def set_source_context(
        self,
        *,
        owner_user_id: str,
        source_id: str,
        enabled: bool,
        actor_user_id: str,
    ) -> dict[str, Any]:
        owner = self._require_owner(owner_user_id, actor_user_id)
        policy = self.source_policy_store.get(source_id, subject_user_id=owner)
        if policy is None or policy.owner_user_id != owner:
            raise PermissionError("owner authorization required")
        saved = self.source_policy_store.set_context_enabled(owner, source_id, enabled)
        return self._source_payload(saved, owner)

    def update_upload_policy(
        self,
        *,
        owner_user_id: str,
        doc_id: str,
        audience_scope: str,
        context_enabled: bool,
        actor_user_id: str,
    ) -> dict[str, Any]:
        owner = self._require_owner(owner_user_id, actor_user_id)
        saved = self.knowledge_base.assign_document_policy(
            doc_id,
            owner_user_id=owner,
            actor_user_id=actor_user_id,
            audience_scope=audience_scope,
            context_enabled=context_enabled,
        )
        return self._upload_payload(saved)

    def set_location_assist(
        self,
        *,
        owner_user_id: str,
        enabled: bool,
        actor_user_id: str,
    ) -> dict[str, Any]:
        owner = self._require_owner(owner_user_id, actor_user_id)
        grants = self.location_grant_store.set_assist(
            owner_user_id=owner,
            enabled=enabled,
            actor_user_id=actor_user_id,
        )
        return {
            "location_assist": grants.location_assist,
            "shared_with": list(grants.shared_with),
        }

    def set_location_share(
        self,
        *,
        owner_user_id: str,
        recipient_user_id: str,
        enabled: bool,
        actor_user_id: str,
    ) -> dict[str, Any]:
        owner = self._require_owner(owner_user_id, actor_user_id)
        grants = self.location_grant_store.set_share(
            owner_user_id=owner,
            recipient_user_id=recipient_user_id,
            enabled=enabled,
            actor_user_id=actor_user_id,
        )
        return {
            "location_assist": grants.location_assist,
            "shared_with": list(grants.shared_with),
        }

    def set_proactive_assistance(
        self,
        *,
        owner_user_id: str,
        enabled: bool,
        actor_user_id: str,
    ) -> bool:
        owner = self._require_owner(owner_user_id, actor_user_id)
        saved = self.preference_store.set_proactive_assistance(
            owner_user_id=owner,
            enabled=enabled,
            actor_user_id=actor_user_id,
        )
        return saved.proactive_assistance


_DEFAULT_SERVICE: ContextPrivacyService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def get_context_privacy_service() -> ContextPrivacyService:
    global _DEFAULT_SERVICE
    with _DEFAULT_SERVICE_LOCK:
        if _DEFAULT_SERVICE is None:
            from rex.knowledge_base import get_knowledge_base

            source_policy = ContextSourcePolicyStore()
            _DEFAULT_SERVICE = ContextPrivacyService(
                source_policy_store=source_policy,
                location_grant_store=LocationGrantStore(source_policy_store=source_policy),
                knowledge_base=get_knowledge_base(),
                preference_store=ContextPrivacyPreferenceStore(),
            )
        return _DEFAULT_SERVICE


__all__ = [
    "ContextPrivacyPreferences",
    "ContextPrivacyPreferenceStore",
    "ContextPrivacyService",
    "get_context_privacy_service",
]
