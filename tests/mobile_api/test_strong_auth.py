from __future__ import annotations

import copy

import pytest

from rex.mobile_api.auth import authenticate_token
from rex.mobile_api.device_proof import sign_transcript
from rex.mobile_api.strong_auth import (
    APPROVAL_TTL_SECONDS,
    CHALLENGE_TTL_SECONDS,
    MobileActionRisk,
    StrongAuthError,
    canonical_action,
    canonical_strong_auth_transcript,
    policy_for_action,
)
from tests.mobile_api.conftest import create_user, paired_login_tokens


def _principal(services, tokens):
    return authenticate_token(services, tokens["access_token"])


def _ha_payload(*, domain: str = "light", service: str = "turn_off") -> dict:
    return {
        "domain": domain,
        "service": service,
        "entity_id": f"{domain}.downstairs",
        "data": {"transition": 1},
    }


def _verified_approval(services, principal, private_key, payload):
    challenge = services.strong_auth_authority.create_challenge(
        principal,
        action_name="home_assistant_call_service",
        payload=payload,
    )
    signature = sign_transcript(private_key, canonical_strong_auth_transcript(challenge))
    approval = services.strong_auth_authority.verify_challenge(
        principal,
        challenge_id=challenge.challenge_id,
        signature_b64=signature,
    )
    return challenge, approval


def test_canonical_action_is_stable_and_ignores_only_server_transport_metadata():
    first = {
        "service": "TURN_OFF",
        "domain": "LIGHT",
        "entity_id": "light.downstairs",
        "data": {"transition": 1},
        "_user_id": "server-user-a",
        "context": {"request_id": "server-a"},
    }
    second = {
        "_user_id": "server-user-b",
        "context": {"request_id": "server-b"},
        "data": {"transition": 1},
        "domain": "light",
        "service": "turn_off",
        "entity_id": "light.downstairs",
    }
    assert (
        canonical_action("HOME_ASSISTANT_CALL_SERVICE", first)[2]
        == canonical_action("home_assistant_call_service", second)[2]
    )
    changed = copy.deepcopy(second)
    changed["data"]["transition"] = 2
    assert (
        canonical_action("home_assistant_call_service", second)[2]
        != canonical_action("home_assistant_call_service", changed)[2]
    )

    nested = copy.deepcopy(second)
    nested["data"]["context"] = "approved-value"
    nested_changed = copy.deepcopy(nested)
    nested_changed["data"]["context"] = "different-value"
    assert (
        canonical_action("home_assistant_call_service", nested)[2]
        != canonical_action("home_assistant_call_service", nested_changed)[2]
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_ha_payload(), MobileActionRisk.HIGH),
        (_ha_payload(domain="lock", service="unlock"), MobileActionRisk.CRITICAL),
        (
            _ha_payload(domain="alarm_control_panel", service="alarm_disarm"),
            MobileActionRisk.CRITICAL,
        ),
    ],
)
def test_server_owned_policy_classifies_home_mutations(payload, expected):
    policy = policy_for_action("home_assistant_call_service", payload)
    assert policy is not None
    assert policy.required_scope == "home.control"
    assert policy.risk_level is expected
    assert policy.requires_strong_auth is True


def test_music_mutation_is_not_misrepresented_as_high_risk():
    policy = policy_for_action("music_pause", {"target": "living-room"})
    assert policy is not None
    assert policy.risk_level is MobileActionRisk.MEDIUM
    assert policy.requires_strong_auth is False


def test_high_risk_approval_binds_session_device_grant_action_and_is_one_time(client, services):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    principal = _principal(services, tokens)
    payload = _ha_payload()
    challenge, approval = _verified_approval(services, principal, tokens["_private_key"], payload)

    assert challenge.user_id == principal.user_id
    assert challenge.session_id == principal.session_id
    assert challenge.device_id == principal.paired_device_id
    assert challenge.grant_id == principal.grant_id
    assert challenge.grant_version == principal.grant_version
    assert challenge.risk_level == "high"

    consumed = services.strong_auth_authority.consume_approval(
        principal,
        approval_id=approval.approval_id,
        action_name="home_assistant_call_service",
        payload=payload,
    )
    assert consumed.action_hash == approval.action_hash

    with pytest.raises(StrongAuthError, match="already used") as replay:
        services.strong_auth_authority.consume_approval(
            principal,
            approval_id=approval.approval_id,
            action_name="home_assistant_call_service",
            payload=payload,
        )
    assert replay.value.reason == "approval_replayed"

    with pytest.raises(StrongAuthError, match="already used") as verify_replay:
        services.strong_auth_authority.verify_challenge(
            principal,
            challenge_id=challenge.challenge_id,
            signature_b64=sign_transcript(
                tokens["_private_key"], canonical_strong_auth_transcript(challenge)
            ),
        )
    assert verify_replay.value.reason == "challenge_replayed"


def test_changed_action_payload_cannot_consume_approval(client, services):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    principal = _principal(services, tokens)
    payload = _ha_payload()
    _, approval = _verified_approval(services, principal, tokens["_private_key"], payload)
    changed = copy.deepcopy(payload)
    changed["service"] = "turn_on"

    with pytest.raises(StrongAuthError, match="does not match") as exc:
        services.strong_auth_authority.consume_approval(
            principal,
            approval_id=approval.approval_id,
            action_name="home_assistant_call_service",
            payload=changed,
        )
    assert exc.value.reason == "action_changed"

    # A mismatch does not consume the valid exact-action approval.
    services.strong_auth_authority.consume_approval(
        principal,
        approval_id=approval.approval_id,
        action_name="home_assistant_call_service",
        payload=payload,
    )


def test_wrong_session_cannot_verify_or_consume(client, services):
    create_user("james", "correct-horse", admin=True)
    first = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    second = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    first_principal = _principal(services, first)
    second_principal = _principal(services, second)
    payload = _ha_payload()
    challenge = services.strong_auth_authority.create_challenge(
        first_principal,
        action_name="home_assistant_call_service",
        payload=payload,
    )
    signature = sign_transcript(first["_private_key"], canonical_strong_auth_transcript(challenge))

    with pytest.raises(StrongAuthError) as wrong_verify:
        services.strong_auth_authority.verify_challenge(
            second_principal,
            challenge_id=challenge.challenge_id,
            signature_b64=signature,
        )
    assert wrong_verify.value.reason == "binding_mismatch"

    approval = services.strong_auth_authority.verify_challenge(
        first_principal,
        challenge_id=challenge.challenge_id,
        signature_b64=signature,
    )
    with pytest.raises(StrongAuthError) as wrong_consume:
        services.strong_auth_authority.consume_approval(
            second_principal,
            approval_id=approval.approval_id,
            action_name="home_assistant_call_service",
            payload=payload,
        )
    assert wrong_consume.value.reason == "binding_mismatch"


def test_expired_challenge_and_approval_fail_closed(client, services, clock):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    principal = _principal(services, tokens)
    payload = _ha_payload()
    challenge = services.strong_auth_authority.create_challenge(
        principal,
        action_name="home_assistant_call_service",
        payload=payload,
    )
    signature = sign_transcript(tokens["_private_key"], canonical_strong_auth_transcript(challenge))
    clock.advance(seconds=CHALLENGE_TTL_SECONDS)
    with pytest.raises(StrongAuthError) as expired_challenge:
        services.strong_auth_authority.verify_challenge(
            principal,
            challenge_id=challenge.challenge_id,
            signature_b64=signature,
        )
    assert expired_challenge.value.reason == "challenge_expired"

    # Fresh challenge, then let its approval expire.
    fresh = services.strong_auth_authority.create_challenge(
        principal,
        action_name="home_assistant_call_service",
        payload=payload,
    )
    approval = services.strong_auth_authority.verify_challenge(
        principal,
        challenge_id=fresh.challenge_id,
        signature_b64=sign_transcript(
            tokens["_private_key"], canonical_strong_auth_transcript(fresh)
        ),
    )
    clock.advance(seconds=APPROVAL_TTL_SECONDS)
    with pytest.raises(StrongAuthError) as expired_approval:
        services.strong_auth_authority.consume_approval(
            principal,
            approval_id=approval.approval_id,
            action_name="home_assistant_call_service",
            payload=payload,
        )
    assert expired_approval.value.reason == "approval_expired"


def test_revoked_device_invalidates_unconsumed_approval(client, services):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    principal = _principal(services, tokens)
    payload = _ha_payload()
    _, approval = _verified_approval(services, principal, tokens["_private_key"], payload)
    assert services.pairing_authority.revoke_device(
        tokens["_paired_device_id"],
        revoked_by="TEST\\DesktopOwner",
        reason="test revocation",
    )
    with pytest.raises(StrongAuthError) as revoked:
        services.strong_auth_authority.consume_approval(
            principal,
            approval_id=approval.approval_id,
            action_name="home_assistant_call_service",
            payload=payload,
        )
    assert revoked.value.reason == "binding_revoked"


def test_scope_and_non_applicable_actions_are_denied_before_challenge(client, services):
    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(client, "james", "correct-horse", scopes=["chat.send"])
    principal = _principal(services, tokens)
    with pytest.raises(StrongAuthError) as scope:
        services.strong_auth_authority.create_challenge(
            principal,
            action_name="home_assistant_call_service",
            payload=_ha_payload(),
        )
    assert scope.value.reason == "scope_denied"

    with pytest.raises(StrongAuthError) as low:
        services.strong_auth_authority.create_challenge(
            principal,
            action_name="music_pause",
            payload={"target": "living-room"},
        )
    assert low.value.reason == "strong_auth_not_applicable"


def test_strong_auth_audit_records_hashes_and_outcomes_without_payloads(client, services):
    from rex.mobile_api.db import connect

    create_user("james", "correct-horse", admin=True)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=["chat.send", "home.control"],
    )
    principal = _principal(services, tokens)
    payload = _ha_payload()
    _, approval = _verified_approval(services, principal, tokens["_private_key"], payload)
    changed = copy.deepcopy(payload)
    changed["service"] = "turn_on"
    with pytest.raises(StrongAuthError):
        services.strong_auth_authority.consume_approval(
            principal,
            approval_id=approval.approval_id,
            action_name="home_assistant_call_service",
            payload=changed,
        )
    services.strong_auth_authority.consume_approval(
        principal,
        approval_id=approval.approval_id,
        action_name="home_assistant_call_service",
        payload=payload,
    )

    conn = connect(services.db_path)
    try:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(mobile_strong_auth_audit)")
        }
        rows = conn.execute(
            "SELECT event_type, action_hash, reason FROM mobile_strong_auth_audit ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    assert "payload" not in columns
    assert "signature" not in columns
    assert [row["event_type"] for row in rows] == [
        "challenge_issued",
        "proof_verified",
        "approval_denied",
        "approval_consumed",
    ]
    assert rows[2]["reason"] == "action_changed"
    assert all(len(row["action_hash"]) == 64 for row in rows)


def test_denied_challenge_is_persistently_audited(client, services):
    from rex.mobile_api.db import connect

    create_user("james", "correct-horse")
    tokens = paired_login_tokens(client, "james", "correct-horse", scopes=["home.control"])
    principal = _principal(services, tokens)
    with pytest.raises(StrongAuthError) as denied:
        services.strong_auth_authority.create_challenge(
            principal,
            action_name="home_assistant_call_service",
            payload=_ha_payload(),
        )
    assert denied.value.reason == "scope_denied"
    conn = connect(services.db_path)
    try:
        row = conn.execute(
            "SELECT event_type, reason, action_hash FROM mobile_strong_auth_audit"
        ).fetchone()
    finally:
        conn.close()
    assert row["event_type"] == "challenge_denied"
    assert row["reason"] == "scope_denied"
    assert len(row["action_hash"]) == 64


def test_home_action_schema_is_strict_and_execution_compatible():
    valid = _ha_payload()
    normalized_hash = canonical_action(
        "home_assistant_call_service",
        {**valid, "domain": " LIGHT ", "service": " TURN_OFF "},
    )[2]
    assert normalized_hash == canonical_action("home_assistant_call_service", valid)[2]

    invalid_actions = [
        {"domain": "light", "service": "turn_off", "data": {}},
        {**valid, "unexpected": True},
        {**valid, "entity_id": "switch.downstairs"},
        {**valid, "data": "not-an-object"},
    ]
    for action in invalid_actions:
        with pytest.raises(StrongAuthError) as exc:
            canonical_action("home_assistant_call_service", action)
        assert exc.value.reason == "invalid_payload"
