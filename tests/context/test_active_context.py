from __future__ import annotations

import pytest

from rex.context.active import ActiveContextRef, ActiveContextStore
from rex.context.source_policy import (
    AudienceScope,
    ContextSourcePolicyStore,
    ContextSourceType,
    DisclosurePolicy,
)


def _ref(
    *,
    user: str = "james",
    domain: str = "media",
    key: str = "session-1",
    expires_at: float = 200.0,
    payload: dict[str, object] | None = None,
    source_ids: tuple[str, ...] = (),
    revision: str = "local:1",
) -> ActiveContextRef:
    return ActiveContextRef(
        domain=domain,
        key=key,
        owner_user_id=user,
        payload=payload or {"target_id": "ha:media_player.living_room"},
        source_ids=source_ids,
        revision=revision,
        expires_at=expires_at,
    )


def test_it_resolves_to_recent_media_for_same_user():
    store = ActiveContextStore(clock=lambda: 100.0)
    store.put(_ref())

    result = store.resolve("james", "pause it", candidate_domains=("media", "timekeeping"))

    assert result.ref is not None
    assert result.ref.domain == "media"
    assert result.reason == "resolved"


def test_two_equally_relevant_refs_require_clarification():
    store = ActiveContextStore(clock=lambda: 100.0)
    store.put(_ref(domain="timekeeping", key="timer-1", payload={"record_type": "timer"}))
    store.put(_ref(domain="timekeeping", key="timer-2", payload={"record_type": "timer"}))

    result = store.resolve("james", "cancel it", candidate_domains=("timekeeping",))

    assert result.ref is None
    assert result.reason == "ambiguous"
    assert {candidate.key for candidate in result.candidates} == {"timer-1", "timer-2"}


def test_cross_user_reference_is_invisible():
    store = ActiveContextStore(clock=lambda: 100.0)
    store.put(_ref(user="cole"))

    assert store.get("james", "media", "session-1") is None
    result = store.resolve("james", "pause it", candidate_domains=("media",))
    assert result.ref is None
    assert result.reason == "not_found"


def test_expired_reference_is_evicted_on_read():
    now = [100.0]
    store = ActiveContextStore(clock=lambda: now[0])
    store.put(_ref(expires_at=110.0))
    now[0] = 111.0

    assert store.get("james", "media", "session-1") is None
    assert store.resolve("james", "pause it", candidate_domains=("media",)).reason == "not_found"


def test_payload_is_bounded_to_scalar_state():
    with pytest.raises(ValueError, match="payload"):
        _ref(payload={"transcript": {"whole": "conversation"}})


def test_source_policy_revision_change_invalidates_reference(tmp_path):
    policy = ContextSourcePolicyStore(tmp_path / "policy")
    policy.register_source(
        "upload:trip",
        ContextSourceType.UPLOAD,
        owner_user_id="james",
        audience_scope=AudienceScope.PRIVATE,
        context_enabled=True,
        disclosure_policy=DisclosurePolicy.OWNER_ONLY,
    )
    store = ActiveContextStore(clock=lambda: 100.0, source_policy_store=policy)
    revision = store.revision_for_sources("james", ("upload:trip",))
    store.put(
        _ref(
            domain="document",
            key="trip",
            source_ids=("upload:trip",),
            revision=revision,
        )
    )
    assert store.get("james", "document", "trip") is not None

    policy.set_context_enabled("james", "upload:trip", False)

    assert store.get("james", "document", "trip") is None
