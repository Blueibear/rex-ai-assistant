from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from rex.context.source_policy import (
    AudienceScope,
    ContextSourcePolicyStore,
    ContextSourceType,
    DisclosurePolicy,
)


def test_private_source_is_filtered_before_retrieval(tmp_path) -> None:
    store = ContextSourcePolicyStore(tmp_path)
    saved = store.register_source(
        "upload:taxes",
        source_type=ContextSourceType.UPLOAD,
        owner_user_id="james",
        context_enabled=True,
    )

    assert saved.audience_scope is AudienceScope.PRIVATE
    assert (
        store.is_context_eligible("upload:taxes", subject_user_id="james", requester_user_id="cole")
        is False
    )
    assert (
        store.is_context_eligible(
            "upload:taxes", subject_user_id="james", requester_user_id="james"
        )
        is True
    )


def test_policy_change_changes_content_free_revision(tmp_path) -> None:
    store = ContextSourcePolicyStore(tmp_path)
    store.register_source(
        "calendar:main",
        source_type=ContextSourceType.INTEGRATION,
        owner_user_id="james",
    )
    before = store.revision_for_user("james")

    store.set_context_enabled("james", "calendar:main", False)
    after = store.revision_for_user("james")

    assert after != before
    assert re.fullmatch(r"[0-9a-f]{64}", before)
    assert re.fullmatch(r"[0-9a-f]{64}", after)
    assert "james" not in after
    assert "calendar" not in after


def test_source_type_defaults_are_privacy_preserving(tmp_path) -> None:
    store = ContextSourcePolicyStore(tmp_path)
    integration = store.register_source(
        "calendar:main", ContextSourceType.INTEGRATION, owner_user_id="james"
    )
    upload = store.register_source("upload:doc-1", ContextSourceType.UPLOAD, owner_user_id="james")
    location = store.register_source(
        "location:current", ContextSourceType.LOCATION, owner_user_id="james"
    )

    assert integration.context_enabled is True
    assert integration.audience_scope is AudienceScope.PRIVATE
    assert integration.disclosure_policy is DisclosurePolicy.OWNER_ONLY
    assert upload.context_enabled is False
    assert upload.audience_scope is AudienceScope.PRIVATE
    assert location.context_enabled is False
    assert location.disclosure_policy is DisclosurePolicy.EXPLICIT_GRANT


def test_household_source_is_context_eligible_for_household_requester(tmp_path) -> None:
    store = ContextSourcePolicyStore(tmp_path)
    store.register_source(
        "integration:shared-weather",
        ContextSourceType.INTEGRATION,
        owner_user_id="james",
        audience_scope=AudienceScope.HOUSEHOLD,
    )

    assert (
        store.is_context_eligible(
            "integration:shared-weather",
            subject_user_id="james",
            requester_user_id="cole",
        )
        is True
    )


def test_unknown_or_disabled_source_is_not_context_eligible(tmp_path) -> None:
    store = ContextSourcePolicyStore(tmp_path)
    store.register_source(
        "calendar:main",
        ContextSourceType.INTEGRATION,
        owner_user_id="james",
    )
    store.set_context_enabled("james", "calendar:main", False)

    assert (
        store.is_context_eligible(
            "calendar:main", subject_user_id="james", requester_user_id="james"
        )
        is False
    )
    assert (
        store.is_context_eligible(
            "missing:source", subject_user_id="james", requester_user_id="james"
        )
        is False
    )


def test_source_policy_rejects_invalid_identity_and_source_id(tmp_path) -> None:
    store = ContextSourcePolicyStore(tmp_path)

    with pytest.raises(ValueError):
        store.register_source(
            "../secret",
            ContextSourceType.INTEGRATION,
            owner_user_id="james",
        )
    with pytest.raises(ValueError):
        store.register_source(
            "calendar:main",
            ContextSourceType.INTEGRATION,
            owner_user_id="../james",
        )


def test_ownerless_private_registration_fails_closed(tmp_path) -> None:
    store = ContextSourcePolicyStore(tmp_path)

    with pytest.raises(ValueError, match="private.*owner"):
        store.register_source(
            "integration:private",
            ContextSourceType.INTEGRATION,
            owner_user_id=None,
            audience_scope=AudienceScope.PRIVATE,
        )


def test_policy_persists_across_store_instances(tmp_path) -> None:
    first = ContextSourcePolicyStore(tmp_path)
    saved = first.register_source(
        "calendar:main", ContextSourceType.INTEGRATION, owner_user_id="james"
    )

    second = ContextSourcePolicyStore(tmp_path)
    loaded = second.get("calendar:main", subject_user_id="james")

    assert loaded == saved
    assert loaded is not None
    assert loaded.policy_revision == 1


def test_concurrent_same_user_registration_does_not_lose_updates(tmp_path) -> None:
    first = ContextSourcePolicyStore(tmp_path)
    second = ContextSourcePolicyStore(tmp_path)

    def register(store: ContextSourcePolicyStore, source_id: str) -> None:
        store.register_source(
            source_id,
            ContextSourceType.INTEGRATION,
            owner_user_id="james",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(register, first, "calendar:main")
        right = pool.submit(register, second, "weather:main")
        left.result(timeout=5)
        right.result(timeout=5)

    loaded = ContextSourcePolicyStore(tmp_path)
    assert loaded.get("calendar:main", subject_user_id="james") is not None
    assert loaded.get("weather:main", subject_user_id="james") is not None
