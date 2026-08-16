from __future__ import annotations

from dataclasses import fields

import pytest

from rex.media.sessions import ActiveMediaSession, ActiveMediaSessionStore


def _session(user_id: str, *, updated_at: float = 1000.0) -> ActiveMediaSession:
    return ActiveMediaSession(
        user_id=user_id,
        target_id="ha:media_player.den",
        provider="ha",
        media_ref="track:1",
        updated_at=updated_at,
    )


def test_same_user_can_get_and_clear_active_session() -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    session = _session("james")

    store.set(session)

    assert store.get("james") == session
    store.clear("james")
    assert store.get("james") is None


def test_active_sessions_are_isolated_by_user() -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    james = _session("james")
    cole = ActiveMediaSession(
        user_id="cole",
        target_id="sonos:kitchen",
        provider="sonos",
        media_ref="album:2",
        updated_at=1000.0,
    )

    store.set(james)
    store.set(cole)

    assert store.get("james") == james
    assert store.get("cole") == cole
    store.clear("cole")
    assert store.get("james") == james
    assert store.get("cole") is None


def test_active_session_expires_and_is_evicted() -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    store.set(_session("james", updated_at=600.0))

    assert store.get("james") is None
    assert "james" not in store._sessions


def test_active_session_expires_at_exact_ttl_boundary() -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    store.set(_session("james", updated_at=700.0))

    assert store.get("james") is None


def test_get_accepts_explicit_time_for_deterministic_expiry() -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 5000.0)
    session = _session("james", updated_at=1000.0)
    store.set(session)

    assert store.get("james", now=1299.9) == session
    assert store.get("james", now=1300.0) is None


def test_active_session_evicts_when_now_precedes_updated_at() -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)
    session = _session("james", updated_at=1000.0)
    store.set(session)

    assert store.get("james", now=500.0) is None
    assert "james" not in store._sessions


def test_active_session_evicts_wall_clock_updated_at_with_default_clock() -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300)
    session = _session("james", updated_at=1_700_000_000.0)
    store.set(session)

    assert store.get("james") is None
    assert "james" not in store._sessions


@pytest.mark.parametrize("user_id", ["", "../cole", "two/users", "NUL"])
@pytest.mark.parametrize("method", ["set", "get", "clear"])
def test_session_operations_reject_invalid_user_ids(user_id: str, method: str) -> None:
    store = ActiveMediaSessionStore(ttl_seconds=300, clock=lambda: 1000.0)

    with pytest.raises(ValueError, match="Invalid user_id"):
        if method == "set":
            store.set(_session(user_id))
        elif method == "get":
            store.get(user_id)
        else:
            store.clear(user_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_id", ""),
        ("provider", ""),
        ("provider", "Apple Music"),
        ("media_ref", ""),
        ("media_ref", "transcript\ncontents"),
    ],
)
def test_active_session_rejects_invalid_bounded_fields(field: str, value: str) -> None:
    values: dict[str, str | float] = {
        "user_id": "james",
        "target_id": "ha:media_player.den",
        "provider": "ha",
        "media_ref": "track:1",
        "updated_at": 1000.0,
    }
    values[field] = value

    with pytest.raises(ValueError):
        ActiveMediaSession(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("updated_at", [-1.0, float("inf"), float("nan")])
def test_active_session_rejects_invalid_timestamp(updated_at: float) -> None:
    with pytest.raises(ValueError, match="updated_at"):
        _session("james", updated_at=updated_at)


@pytest.mark.parametrize("ttl_seconds", [0, -1, float("inf"), float("nan")])
def test_session_store_rejects_invalid_ttl(ttl_seconds: float) -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        ActiveMediaSessionStore(ttl_seconds=ttl_seconds)


def test_session_model_contains_only_bounded_follow_up_fields() -> None:
    assert {field.name for field in fields(ActiveMediaSession)} == {
        "user_id",
        "target_id",
        "provider",
        "media_ref",
        "updated_at",
    }

    with pytest.raises(TypeError):
        ActiveMediaSession(
            user_id="james",
            target_id="ha:media_player.den",
            provider="ha",
            media_ref="track:1",
            updated_at=1000.0,
            access_token="secret",  # type: ignore[call-arg]
        )
