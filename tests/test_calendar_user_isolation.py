"""Per-user calendar account and credential isolation tests (issue #303).

These tests reproduce and lock down the cross-user calendar defects present
on the pre-#303 implementation (they fail against that base):

1. All users shared one global calendar store: any user saw and mutated
   every other user's events.
2. Real calendar operations ran without any validated user identity.
3. The provider-API service (rex.integrations.calendar_service) served any
   caller with the global ``GOOGLE_CALENDAR_ACCESS_TOKEN`` and fell back to
   shared stub data on provider failures (fail open).
4. Full private event payloads (titles, attendees, locations, descriptions)
   were published on shared event-bus topics.
5. Account revocation had no effect in long-lived processes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.calendar_accounts import (
    CalendarAccountAccessError,
    CalendarAccountResolver,
    CalendarIdentityError,
)
from rex.calendar_service import CalendarEvent, CalendarService

ALICE = "alice"
BOB = "bob"

# Marker substring used to detect credential leakage in any output.
SECRET_MARKER = "s3cr3t-cal-t0ken"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(title: str, hours_from_now: float = 1.0) -> CalendarEvent:
    start = datetime.now(UTC) + timedelta(hours=hours_from_now)
    return CalendarEvent(
        title=title,
        start_time=start,
        end_time=start + timedelta(hours=1),
        attendees=["private@example.com"],
        location="Private location",
        description="Private description",
    )


def _ics_file(tmp_path, name: str, summary: str) -> str:
    path = tmp_path / name
    path.write_text(
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//Test//EN\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{name}-1\r\n"
        f"SUMMARY:{summary}\r\n"
        "DTSTART:20270101T100000Z\r\n"
        "DTEND:20270101T110000Z\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n",
        encoding="utf-8",
    )
    return str(path)


def _two_user_ics_resolver(tmp_path) -> CalendarAccountResolver:
    return CalendarAccountResolver.from_raw_config(
        {
            "calendar": {
                "accounts": [
                    {
                        "id": "alice-cal",
                        "provider": "ics",
                        "ics": {"source": _ics_file(tmp_path, "alice.ics", "Alice Meeting")},
                    },
                    {
                        "id": "bob-cal",
                        "provider": "ics",
                        "ics": {"source": _ics_file(tmp_path, "bob.ics", "Bob Meeting")},
                    },
                ],
            },
            "users": {
                ALICE: {"calendar_accounts": [{"account_id": "alice-cal"}]},
                BOB: {"calendar_accounts": [{"account_id": "bob-cal"}]},
            },
        }
    )


class RecordingBus:
    """Fake event bus recording (topic, payload) publishes."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload))

    def topics(self) -> list[str]:
        return [topic for topic, _ in self.published]

    def payloads_for(self, topic: str) -> list[dict[str, Any]]:
        return [payload for t, payload in self.published if t == topic]


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch):
    """Point per-user runtime calendar storage at a temp directory."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "appdata"))
    return tmp_path


# ---------------------------------------------------------------------------
# Identity validation (fails before any account/credential lookup)
# ---------------------------------------------------------------------------


class TestIdentityRequired:
    def test_missing_user_fails_closed(self):
        svc = CalendarService(mock_events=[_event("X")], owner_user_id=ALICE)
        with pytest.raises(CalendarIdentityError):
            svc.get_all_events()
        with pytest.raises(CalendarIdentityError):
            svc.list_upcoming()
        with pytest.raises(CalendarIdentityError):
            svc.get_events(datetime.now(UTC), datetime.now(UTC) + timedelta(days=1))
        with pytest.raises(CalendarIdentityError):
            svc.create_event("t", datetime.now(UTC), datetime.now(UTC) + timedelta(hours=1))
        with pytest.raises(CalendarIdentityError):
            svc.update_event("id", {"title": "x"})
        with pytest.raises(CalendarIdentityError):
            svc.delete_event("id")
        with pytest.raises(CalendarIdentityError):
            svc.find_conflicts()

    @pytest.mark.parametrize("bad", ["", "   ", "../../etc", "..", "a/b"])
    def test_invalid_and_traversal_user_ids_fail_closed(self, bad):
        svc = CalendarService(mock_events=[_event("X")], owner_user_id=ALICE)
        with pytest.raises(CalendarIdentityError):
            svc.get_all_events(user_id=bad)

    def test_missing_identity_fails_before_backend_resolution(self, tmp_path):
        """Identity errors are raised before any account/backend lookup."""
        resolver = _two_user_ics_resolver(tmp_path)
        svc = CalendarService(account_resolver=resolver)
        with pytest.raises(CalendarIdentityError):
            svc.get_all_events(user_id=None)
        # No backend was built for anyone.
        assert svc._user_backends == {}

    def test_connect_fails_closed_without_identity(self):
        svc = CalendarService(mock_events=[_event("X")], owner_user_id=ALICE)
        assert svc.connect() is False
        assert svc.connect(user_id="../evil") is False


# ---------------------------------------------------------------------------
# Store isolation (defect 1 on base: shared global store)
# ---------------------------------------------------------------------------


class TestStoreIsolation:
    def test_in_memory_events_are_owner_bound(self):
        svc = CalendarService(mock_events=[_event("Alice private event")], owner_user_id=ALICE)
        assert [e.title for e in svc.get_all_events(user_id=ALICE)] == ["Alice private event"]
        # Bob sees an isolated, empty store — never Alice's events.
        assert svc.get_all_events(user_id=BOB) == []

    def test_created_event_not_visible_to_other_user(self):
        svc = CalendarService(mock_events=[], owner_user_id=ALICE)
        svc.create_event(
            "Alice meeting",
            datetime.now(UTC) + timedelta(hours=1),
            datetime.now(UTC) + timedelta(hours=2),
            user_id=ALICE,
        )
        assert [e.title for e in svc.get_all_events(user_id=ALICE)] == ["Alice meeting"]
        assert svc.get_all_events(user_id=BOB) == []
        assert svc.list_upcoming(user_id=BOB) == []

    def test_update_cannot_cross_users(self):
        svc = CalendarService(mock_events=[], owner_user_id=ALICE)
        event = svc.create_event(
            "Alice meeting",
            datetime.now(UTC) + timedelta(hours=1),
            datetime.now(UTC) + timedelta(hours=2),
            user_id=ALICE,
        )
        assert svc.update_event(event.event_id, {"title": "Hacked"}, user_id=BOB) is None
        assert svc.get_all_events(user_id=ALICE)[0].title == "Alice meeting"

    def test_delete_cannot_cross_users(self):
        svc = CalendarService(mock_events=[], owner_user_id=ALICE)
        event = svc.create_event(
            "Alice meeting",
            datetime.now(UTC) + timedelta(hours=1),
            datetime.now(UTC) + timedelta(hours=2),
            user_id=ALICE,
        )
        assert svc.delete_event(event.event_id, user_id=BOB) is False
        assert len(svc.get_all_events(user_id=ALICE)) == 1

    def test_stub_disk_stores_are_isolated_per_user(self, isolated_storage, monkeypatch):
        """Stub-mode (no accounts configured) keeps separate per-user files."""
        empty = CalendarAccountResolver.from_raw_config({})
        svc = CalendarService(account_resolver=empty)
        svc.create_event(
            "Alice stub event",
            datetime.now(UTC) + timedelta(hours=1),
            datetime.now(UTC) + timedelta(hours=2),
            user_id=ALICE,
        )
        titles_alice = [e.title for e in svc.get_all_events(user_id=ALICE)]
        titles_bob = [e.title for e in svc.get_all_events(user_id=BOB)]
        assert "Alice stub event" in titles_alice
        assert "Alice stub event" not in titles_bob

        # A fresh service instance (fresh process) sees the same isolation.
        svc2 = CalendarService(account_resolver=empty)
        assert "Alice stub event" in [e.title for e in svc2.get_all_events(user_id=ALICE)]
        assert "Alice stub event" not in [e.title for e in svc2.get_all_events(user_id=BOB)]

    def test_explicit_account_id_in_stub_mode_is_generic_error(self):
        svc = CalendarService(mock_events=[], owner_user_id=ALICE)
        assert svc.get_all_events(user_id=ALICE) == []  # sanity: works without account
        with pytest.raises(CalendarAccountAccessError):
            svc.list_upcoming(user_id=ALICE, account_id="any-account")


# ---------------------------------------------------------------------------
# Account-backed isolation (ICS backends per user)
# ---------------------------------------------------------------------------


class TestAccountBackedIsolation:
    def test_users_get_only_their_own_ics_events(self, tmp_path):
        svc = CalendarService(account_resolver=_two_user_ics_resolver(tmp_path))
        alice_titles = [e.title for e in svc.get_all_events(user_id=ALICE)]
        bob_titles = [e.title for e in svc.get_all_events(user_id=BOB)]
        assert alice_titles == ["Alice Meeting"]
        assert bob_titles == ["Bob Meeting"]

    def test_backend_for_alice_is_not_reused_for_bob(self, tmp_path):
        svc = CalendarService(account_resolver=_two_user_ics_resolver(tmp_path))
        svc.get_all_events(user_id=ALICE)
        svc.get_all_events(user_id=BOB)
        keys = set(svc._user_backends)
        assert (ALICE, "alice-cal") in keys
        assert (BOB, "bob-cal") in keys
        assert svc._user_backends[(ALICE, "alice-cal")] is not svc._user_backends[(BOB, "bob-cal")]

    def test_same_user_backend_reuse_is_safe(self, tmp_path):
        svc = CalendarService(account_resolver=_two_user_ics_resolver(tmp_path))
        svc.get_all_events(user_id=ALICE)
        backend_first = svc._user_backends[(ALICE, "alice-cal")]
        svc.get_all_events(user_id=ALICE)
        assert svc._user_backends[(ALICE, "alice-cal")] is backend_first
        assert len([k for k in svc._user_backends if k[0] == ALICE]) == 1

    def test_explicit_foreign_account_raises_generic_error(self, tmp_path):
        svc = CalendarService(account_resolver=_two_user_ics_resolver(tmp_path))
        with pytest.raises(CalendarAccountAccessError) as foreign:
            svc.list_upcoming(user_id=ALICE, account_id="bob-cal")
        with pytest.raises(CalendarAccountAccessError) as missing:
            svc.list_upcoming(user_id=ALICE, account_id="no-such")
        assert str(foreign.value).replace("bob-cal", "X") == str(missing.value).replace(
            "no-such", "X"
        )
        # Bob's backend was never built for Alice's request.
        assert (ALICE, "bob-cal") not in svc._user_backends

    def test_unassigned_named_user_fails_closed_not_first_account(self, tmp_path):
        """A named user with no assignment gets nothing — not the global or
        first configured account."""
        svc = CalendarService(account_resolver=_two_user_ics_resolver(tmp_path))
        assert svc.get_all_events(user_id="charlie") == []
        with pytest.raises(IntegrationNotConfiguredError):
            svc.create_event(
                "x",
                datetime.now(UTC) + timedelta(hours=1),
                datetime.now(UTC) + timedelta(hours=2),
                user_id="charlie",
            )
        assert svc.connect(user_id="charlie") is False

    def test_followup_cues_read_only_own_events(self, tmp_path, monkeypatch):
        """generate_followup_cues() consumes only the requesting user's events."""
        created: list[tuple[str, str]] = []

        class FakeCueStore:
            def has_cue_for_source(self, user_id, source_type, source_id):
                return False

            def add_cue(self, *, user_id, title, **kwargs):
                created.append((user_id, title))

        import rex.cue_store as cue_store_mod

        monkeypatch.setattr(cue_store_mod, "get_cue_store", lambda: FakeCueStore())

        past = _event("Bob past event", hours_from_now=-2)
        svc = CalendarService(mock_events=[past], owner_user_id=BOB)
        # Alice generates cues: none of Bob's events may leak into her cues.
        assert svc.generate_followup_cues(ALICE) == 0
        assert created == []
        # Bob gets his own cue.
        assert svc.generate_followup_cues(BOB) == 1
        assert created == [(BOB, "Bob past event")]


# ---------------------------------------------------------------------------
# Revocation without restart
# ---------------------------------------------------------------------------


class TestRevocation:
    def test_revoking_assignment_takes_effect_without_restart(self, tmp_path, monkeypatch):
        resolver_with_alice = _two_user_ics_resolver(tmp_path)
        resolver_revoked = CalendarAccountResolver.from_raw_config(
            {
                "calendar": {
                    "accounts": [
                        {
                            "id": "bob-cal",
                            "provider": "ics",
                            "ics": {"source": _ics_file(tmp_path, "bob2.ics", "Bob Meeting")},
                        }
                    ]
                },
                "users": {BOB: {"calendar_accounts": [{"account_id": "bob-cal"}]}},
            }
        )

        state = {"stamp": 1, "resolver": resolver_with_alice}
        import rex.calendar_accounts as ca_mod

        monkeypatch.setattr(ca_mod, "config_stamp", lambda: state["stamp"])
        monkeypatch.setattr(
            CalendarAccountResolver, "load", classmethod(lambda cls: state["resolver"])
        )

        svc = CalendarService()  # long-lived, config-backed resolver
        assert [e.title for e in svc.get_all_events(user_id=ALICE)] == ["Alice Meeting"]

        # Revoke Alice's assignment; the config stamp changes.
        state["resolver"] = resolver_revoked
        state["stamp"] = 2

        assert svc.get_all_events(user_id=ALICE) == []
        with pytest.raises(IntegrationNotConfiguredError):
            svc.create_event(
                "x",
                datetime.now(UTC) + timedelta(hours=1),
                datetime.now(UTC) + timedelta(hours=2),
                user_id=ALICE,
            )


# ---------------------------------------------------------------------------
# Event-bus topics (defect 4 on base: private payloads on shared topics)
# ---------------------------------------------------------------------------

#: Fields that must never appear on shared topics.
_PRIVATE_KEYS = {"events", "event", "title", "attendees", "location", "description"}


class TestEventBusIsolation:
    def test_shared_topics_carry_only_safe_envelopes(self):
        bus = RecordingBus()
        svc = CalendarService(
            bus, mock_events=[_event("Alice secret meeting")], owner_user_id=ALICE
        )
        svc.connect(user_id=ALICE)
        svc.list_upcoming(user_id=ALICE)
        event = svc.create_event(
            "Another secret",
            datetime.now(UTC) + timedelta(hours=1),
            datetime.now(UTC) + timedelta(hours=2),
            user_id=ALICE,
        )
        svc.update_event(event.event_id, {"title": "Renamed secret"}, user_id=ALICE)
        svc.delete_event(event.event_id, user_id=ALICE)
        svc.get_events(datetime.now(UTC), datetime.now(UTC) + timedelta(days=1), user_id=ALICE)

        shared = [(topic, payload) for topic, payload in bus.published if ".user." not in topic]
        assert shared, "expected shared envelope events"
        for topic, payload in shared:
            leaked = _PRIVATE_KEYS.intersection(payload)
            assert not leaked, f"shared topic {topic} leaked private fields: {leaked}"
            blob = str(payload)
            assert "secret" not in blob.lower()
            assert "private@example.com" not in blob

    def test_private_payloads_go_to_user_scoped_topics(self):
        bus = RecordingBus()
        svc = CalendarService(
            bus, mock_events=[_event("Alice secret meeting")], owner_user_id=ALICE
        )
        svc.list_upcoming(user_id=ALICE)
        user_payloads = bus.payloads_for(f"calendar.upcoming.user.{ALICE}")
        assert user_payloads
        assert user_payloads[0]["events"][0]["title"] == "Alice secret meeting"
        # No other user's topic ever received Alice's payload.
        assert not [t for t in bus.topics() if ".user." in t and not t.endswith(f".user.{ALICE}")]


# ---------------------------------------------------------------------------
# Provider-API service (rex.integrations.calendar_service)
# ---------------------------------------------------------------------------


class TestProviderServiceIsolation:
    def _raw_config(self) -> dict:
        return {
            "calendar": {
                "provider": "google",
                "accounts": [
                    {
                        "id": "alice-google",
                        "provider": "google",
                        "credential_ref": "GOOGLE_CALENDAR_TOKEN_ALICE",
                    },
                    {
                        "id": "bob-google",
                        "provider": "google",
                        "credential_ref": "GOOGLE_CALENDAR_TOKEN_BOB",
                    },
                ],
            },
            "users": {
                ALICE: {"calendar_accounts": [{"account_id": "alice-google"}]},
                BOB: {"calendar_accounts": [{"account_id": "bob-google"}]},
            },
        }

    def test_named_user_gets_own_token_never_bobs(self, monkeypatch):
        from rex.integrations.calendar_service import create_calendar_service_for_user

        monkeypatch.setenv("GOOGLE_CALENDAR_TOKEN_ALICE", f"alice-{SECRET_MARKER}")
        monkeypatch.setenv("GOOGLE_CALENDAR_TOKEN_BOB", f"bob-{SECRET_MARKER}")
        monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", f"global-{SECRET_MARKER}")

        svc, provider = create_calendar_service_for_user(ALICE, self._raw_config())
        assert provider == "google" and svc is not None
        auth = svc._google_headers()["Authorization"]
        assert auth == f"Bearer alice-{SECRET_MARKER}"
        assert "bob-" not in auth and "global-" not in auth

    def test_named_user_never_inherits_global_env_token(self, monkeypatch):
        from rex.integrations.calendar_service import create_calendar_service_for_user

        monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", f"global-{SECRET_MARKER}")
        raw = {"calendar": {"provider": "google"}}
        svc, provider = create_calendar_service_for_user("james", raw)
        assert svc is None
        assert provider == "none"

    def test_default_profile_keeps_legacy_global_provider(self, monkeypatch):
        from rex.integrations.calendar_service import create_calendar_service_for_user

        monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", f"global-{SECRET_MARKER}")
        raw = {"calendar": {"provider": "google"}}
        svc, provider = create_calendar_service_for_user("default", raw)
        assert provider == "google" and svc is not None
        assert svc._google_headers()["Authorization"] == f"Bearer global-{SECRET_MARKER}"

    def test_identity_required_before_any_credential_read(self, monkeypatch):
        """Missing/invalid identity raises before the resolver or any
        environment token is consulted."""
        from rex.integrations import calendar_service as provider_mod

        consulted = {"resolver": False}

        class BoomResolver:
            @classmethod
            def from_raw_config(cls, raw):
                consulted["resolver"] = True
                raise AssertionError("resolver must not be consulted without identity")

        import rex.calendar_accounts as ca_mod

        monkeypatch.setattr(ca_mod, "CalendarAccountResolver", BoomResolver)
        with pytest.raises(PermissionError):
            provider_mod.create_calendar_service_for_user("", self._raw_config())
        with pytest.raises(PermissionError):
            provider_mod.create_calendar_service_for_user("../evil", self._raw_config())
        assert consulted["resolver"] is False

    def test_missing_named_token_fails_closed_no_fallback(self, monkeypatch):
        from rex.integrations.calendar_service import create_calendar_service_for_user

        monkeypatch.delenv("GOOGLE_CALENDAR_TOKEN_ALICE", raising=False)
        monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", f"global-{SECRET_MARKER}")
        svc, provider = create_calendar_service_for_user(ALICE, self._raw_config())
        assert svc is None and provider == "none"

    def test_google_failure_returns_empty_not_stub_data(self, monkeypatch):
        """Regression: base implementation returned shared stub events on
        provider errors (fail open)."""
        from rex.integrations.calendar_service import CalendarService as ProviderCalendarService

        svc = ProviderCalendarService(calendar_provider="google", access_token="bad-token")

        def boom(*args, **kwargs):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        events = svc.get_events(datetime.now(UTC), datetime.now(UTC) + timedelta(days=7))
        assert events == []

    def test_no_token_values_in_logs(self, monkeypatch, caplog):
        from rex.integrations.calendar_service import create_calendar_service_for_user

        monkeypatch.setenv("GOOGLE_CALENDAR_TOKEN_ALICE", f"alice-{SECRET_MARKER}")
        with caplog.at_level(logging.DEBUG):
            create_calendar_service_for_user(ALICE, self._raw_config())
            create_calendar_service_for_user("charlie", self._raw_config())
        assert SECRET_MARKER not in caplog.text

    def test_arbitrary_credential_reference_injection_is_ignored(self, monkeypatch):
        """Callers cannot supply their own credential reference: routing uses
        only the authorized account definition's credential_ref."""
        from rex.integrations.calendar_service import create_calendar_service_for_user

        monkeypatch.setenv("GOOGLE_CALENDAR_TOKEN_ALICE", "alice-token")
        monkeypatch.setenv("EVIL_REF", "evil-token")
        raw = self._raw_config()
        # A malicious per-user entry cannot switch the credential ref: the
        # canonical definition in calendar.accounts is authoritative.
        raw["users"][ALICE]["calendar_accounts"] = [
            {
                "account_id": "alice-google",
                "credential_ref": "EVIL_REF",
                "credentials_key": "EVIL_REF",
            }
        ]
        svc, provider = create_calendar_service_for_user(ALICE, raw)
        assert provider == "google" and svc is not None
        assert svc._google_headers()["Authorization"] == "Bearer alice-token"
