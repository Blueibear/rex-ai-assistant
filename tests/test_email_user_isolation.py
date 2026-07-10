"""Per-user email account and credential isolation tests (issue #303).

These tests reproduce and lock down the cross-user email defects:

1. A named user must never route reads or sends through another user's
   configured account or credentials (no global default / first-account
   fallback for named users).
2. A backend resolved or connected for User A must never be reused for
   User B in the same process.
3. Identity is required and validated before any credential lookup; missing,
   blank, malformed, and traversal-style identities fail closed.
4. The ``user_email_accounts`` map is the authoritative authorization map;
   an empty map means "no access", never "allow all".
5. Legacy global ``email.accounts`` entries belong only to the explicit
   ``default`` profile.
6. Stub/mock mode isolates mutable state between users.
7. No credential value ever appears in results, events, or error messages.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from rex.email_service import EmailService

ALICE = "alice"
BOB = "bob"

ALICE_HOST = "imap.alice.example.com"
BOB_HOST = "imap.bob.example.com"

ALICE_CRED_REF = "email:alice-work"
BOB_CRED_REF = "email:bob-personal"

# Marker substrings used to detect credential leakage in any output.
SECRET_MARKER = "s3cr3t-p4ss"


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _account_def(account_id: str, host_stem: str, cred_ref: str) -> dict[str, Any]:
    return {
        "id": account_id,
        "label": f"{account_id} label",
        "address": f"{account_id}@{host_stem}.example.com",
        "imap": {"host": f"imap.{host_stem}.example.com", "port": 993, "ssl": True},
        "smtp": {"host": f"smtp.{host_stem}.example.com", "port": 587, "starttls": True},
        "credential_ref": cred_ref,
    }


def _two_user_raw_config() -> dict[str, Any]:
    """Two named users, each owning exactly one configured account."""
    return {
        "email": {
            "default_account_id": "alice-work",
            "accounts": [
                _account_def("alice-work", "alice", ALICE_CRED_REF),
                _account_def("bob-personal", "bob", BOB_CRED_REF),
            ],
        },
        "users": {
            ALICE: {
                "email_accounts": [
                    {
                        "account_id": "alice-work",
                        "backend": "imap",
                        "credentials_key": ALICE_CRED_REF,
                    }
                ]
            },
            BOB: {
                "email_accounts": [
                    {
                        "account_id": "bob-personal",
                        "backend": "imap",
                        "credentials_key": BOB_CRED_REF,
                    }
                ]
            },
        },
    }


def _legacy_only_raw_config() -> dict[str, Any]:
    """Legacy global accounts with no ``users`` block at all."""
    return {
        "email": {
            "default_account_id": "legacy-main",
            "accounts": [_account_def("legacy-main", "legacy", "email:legacy-main")],
        },
    }


class FakeCredentialManager:
    """Records every credential-ref lookup and returns a marked secret."""

    def __init__(self) -> None:
        self.lookups: list[str] = []

    def get_token(self, ref: str, **_: Any) -> str:
        self.lookups.append(ref)
        return f"user-{ref}:{SECRET_MARKER}-{ref}"

    def get_credential(self, name: str) -> None:
        return None


class FakeImapBackend:
    """Stand-in for ImapSmtpEmailBackend: no network, distinct mail per host."""

    instances: list[FakeImapBackend] = []

    def __init__(
        self,
        *,
        imap_host: str,
        imap_port: int = 993,
        smtp_host: str = "",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        use_starttls: bool = True,
    ) -> None:
        self.imap_host = imap_host
        self.smtp_host = smtp_host
        self.username = username
        self.password = password
        self.sent: list[dict[str, Any]] = []
        self.connected = False
        FakeImapBackend.instances.append(self)

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def fetch_unread(self, limit: int = 10) -> list[Any]:
        return [
            SimpleNamespace(
                message_id=f"msg-{self.imap_host}-1",
                from_addr=f"sender@{self.imap_host}",
                subject=f"mail-via-{self.imap_host}",
                snippet="hello",
                received_at=datetime(2026, 7, 1, tzinfo=UTC),
                to_addrs=["me@example.com"],
                labels=["unread"],
            )
        ][:limit]

    def mark_as_read(self, message_id: str) -> bool:
        return True

    def send(self, *, from_addr: str, to_addrs: list[str], subject: str, body: str) -> Any:
        self.sent.append(
            {"from_addr": from_addr, "to_addrs": to_addrs, "subject": subject, "body": body}
        )
        return SimpleNamespace(ok=True, message_id=f"mid-{self.smtp_host}", error=None)


@pytest.fixture(autouse=True)
def _reset_fake_backend_registry():
    FakeImapBackend.instances = []
    yield
    FakeImapBackend.instances = []


@pytest.fixture()
def fake_creds() -> FakeCredentialManager:
    return FakeCredentialManager()


def _patch_env(
    monkeypatch: pytest.MonkeyPatch,
    raw_config: dict[str, Any],
) -> None:
    """Point config loading at *raw_config* and neutralize the real backend."""
    import rex.config_manager as config_manager
    import rex.email_backends.imap_smtp as imap_smtp

    monkeypatch.setattr(config_manager, "load_config", lambda *a, **k: raw_config)
    monkeypatch.setattr(imap_smtp, "ImapSmtpEmailBackend", FakeImapBackend)


def _make_service(
    monkeypatch: pytest.MonkeyPatch,
    raw_config: dict[str, Any],
    fake_creds: FakeCredentialManager,
) -> EmailService:
    _patch_env(monkeypatch, raw_config)
    svc = EmailService()
    svc.credential_manager = fake_creds
    return svc


# ---------------------------------------------------------------------------
# Defect reproduction #1: named user routed through another user's account
# ---------------------------------------------------------------------------


class TestNoForeignAccountRouting:
    def test_send_for_unassigned_user_never_uses_global_default_account(
        self, monkeypatch, fake_creds
    ):
        """A user with no assigned accounts must not send via the global default."""
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        result = svc.send(
            to="someone@example.com",
            subject="hi",
            body="hello",
            user_id="james",
        )

        assert result.get("ok") is not True
        assert ALICE_CRED_REF not in fake_creds.lookups
        assert BOB_CRED_REF not in fake_creds.lookups
        # Nothing was handed to any backend.
        assert all(not b.sent for b in FakeImapBackend.instances)

    def test_send_without_account_id_uses_requesters_own_account(self, monkeypatch, fake_creds):
        """Bob's account (not the global default alice-work) serves Bob's send."""
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        result = svc.send(
            to="someone@example.com",
            subject="hi",
            body="hello",
            user_id=BOB,
        )

        assert result.get("ok") is True
        assert ALICE_CRED_REF not in fake_creds.lookups
        assert BOB_CRED_REF in fake_creds.lookups
        sent_hosts = [b.smtp_host for b in FakeImapBackend.instances if b.sent]
        assert sent_hosts == ["smtp.bob.example.com"]

    def test_send_explicit_foreign_account_rejected_before_credential_lookup(
        self, monkeypatch, fake_creds
    ):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        with pytest.raises(PermissionError):
            svc.send(
                to="x@example.com",
                subject="s",
                body="b",
                user_id=BOB,
                account_id="alice-work",
            )
        assert fake_creds.lookups == []

    def test_fetch_unread_explicit_foreign_account_rejected(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        with pytest.raises(PermissionError):
            svc.fetch_unread(user_id=BOB, account_id="alice-work")
        assert fake_creds.lookups == []

    def test_foreign_and_nonexistent_accounts_are_indistinguishable(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        with pytest.raises(PermissionError) as foreign:
            svc.send(to="x@e.com", subject="s", body="b", user_id=BOB, account_id="alice-work")
        with pytest.raises(PermissionError) as missing:
            svc.send(to="x@e.com", subject="s", body="b", user_id=BOB, account_id="no-such-acct")

        # Same message shape modulo the requested id; neither leaks existence.
        msg_foreign = str(foreign.value).replace("alice-work", "<id>")
        msg_missing = str(missing.value).replace("no-such-acct", "<id>")
        assert msg_foreign == msg_missing


# ---------------------------------------------------------------------------
# Defect reproduction #2: backend resolved for User A reused for User B
# ---------------------------------------------------------------------------


class TestNoCrossUserBackendReuse:
    def test_backend_resolved_for_alice_not_reused_for_bob(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        alice_msgs = svc.fetch_unread(user_id=ALICE)
        bob_msgs = svc.fetch_unread(user_id=BOB)

        assert [m.subject for m in alice_msgs] == [f"mail-via-{ALICE_HOST}"]
        assert [m.subject for m in bob_msgs] == [f"mail-via-{BOB_HOST}"]

    def test_backend_credentials_looked_up_per_owner_only(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        svc.fetch_unread(user_id=ALICE)
        lookups_after_alice = list(fake_creds.lookups)
        svc.fetch_unread(user_id=BOB)

        assert lookups_after_alice == [ALICE_CRED_REF]
        assert BOB_CRED_REF in fake_creds.lookups
        # Alice's credential was never re-fetched on Bob's behalf beyond her own call.
        assert fake_creds.lookups.count(ALICE_CRED_REF) == 1

    def test_backend_instance_reused_for_same_user_and_account(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        svc.fetch_unread(user_id=ALICE)
        svc.fetch_unread(user_id=ALICE)

        alice_backends = [b for b in FakeImapBackend.instances if b.imap_host == ALICE_HOST]
        assert len(alice_backends) == 1


# ---------------------------------------------------------------------------
# Identity is required and validated before credential lookup
# ---------------------------------------------------------------------------


class TestIdentityFailClosed:
    @pytest.mark.parametrize("bad_user", [None, "", "   ", "..", "../evil", "a/b", "a\\b"])
    def test_fetch_unread_invalid_identity_fails_closed(self, monkeypatch, fake_creds, bad_user):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        with pytest.raises((PermissionError, ValueError)):
            svc.fetch_unread(user_id=bad_user)
        assert fake_creds.lookups == []

    @pytest.mark.parametrize("bad_user", [None, "", "..", "../evil"])
    def test_send_invalid_identity_fails_closed(self, monkeypatch, fake_creds, bad_user):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        with pytest.raises((PermissionError, ValueError)):
            svc.send(to="x@e.com", subject="s", body="b", user_id=bad_user)
        assert fake_creds.lookups == []

    def test_mark_as_read_requires_identity(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        with pytest.raises((PermissionError, ValueError, TypeError)):
            svc.mark_as_read("msg-1")

    def test_triage_unread_requires_identity(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        with pytest.raises((PermissionError, ValueError, TypeError)):
            svc.triage_unread()


# ---------------------------------------------------------------------------
# Authorization map semantics
# ---------------------------------------------------------------------------


class TestAuthorizationMap:
    def test_empty_user_map_is_deny_not_allow_all(self):
        """No configured scoping means no account access — never allow-all."""
        svc = EmailService()

        with pytest.raises(PermissionError):
            svc._check_account_access(ALICE, "any-account")

    def test_get_accounts_requires_valid_identity(self):
        svc = EmailService()

        with pytest.raises((PermissionError, ValueError)):
            svc.get_accounts("../evil")


# ---------------------------------------------------------------------------
# Legacy global accounts belong only to the explicit `default` profile
# ---------------------------------------------------------------------------


class TestLegacyAccountPolicy:
    def test_default_profile_keeps_legacy_account(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _legacy_only_raw_config(), fake_creds)

        msgs = svc.fetch_unread(user_id="default")

        assert [m.subject for m in msgs] == ["mail-via-imap.legacy.example.com"]
        assert fake_creds.lookups == ["email:legacy-main"]

    def test_named_user_does_not_inherit_legacy_accounts(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _legacy_only_raw_config(), fake_creds)

        msgs = svc.fetch_unread(user_id="james")

        assert msgs == []
        assert fake_creds.lookups == []

    def test_named_user_cannot_send_via_legacy_account(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _legacy_only_raw_config(), fake_creds)

        result = svc.send(to="x@e.com", subject="s", body="b", user_id="james")

        assert result.get("ok") is not True
        assert fake_creds.lookups == []

    def test_unassigned_legacy_account_stays_with_default_when_users_block_exists(
        self, monkeypatch, fake_creds
    ):
        """A users block for alice must not stop `default` from owning legacy accounts."""
        raw = _two_user_raw_config()
        raw["email"]["accounts"].append(_account_def("legacy-extra", "extra", "email:extra"))

        svc = _make_service(monkeypatch, raw, fake_creds)

        msgs = svc.fetch_unread(user_id="default", account_id="legacy-extra")
        assert [m.subject for m in msgs] == ["mail-via-imap.extra.example.com"]

        with pytest.raises(PermissionError):
            svc.fetch_unread(user_id=ALICE, account_id="legacy-extra")


# ---------------------------------------------------------------------------
# Stub/mock mode isolation between users
# ---------------------------------------------------------------------------


class TestStubModeIsolation:
    @pytest.fixture()
    def stub_service(self, tmp_path, monkeypatch) -> EmailService:
        _patch_env(monkeypatch, {})  # no accounts configured at all
        mock_file = tmp_path / "mock_emails.json"
        mock_file.write_text(
            """
            [
              {"id": "m1", "from_addr": "a@example.com", "subject": "One",
               "snippet": "one", "received_at": "2026-07-01T00:00:00+00:00",
               "labels": ["unread"]},
              {"id": "m2", "from_addr": "b@example.com", "subject": "Two",
               "snippet": "two", "received_at": "2026-07-01T01:00:00+00:00",
               "labels": ["unread"]}
            ]
            """,
            encoding="utf-8",
        )
        return EmailService(mock_data_file=mock_file)

    def test_mark_read_by_one_user_does_not_affect_another(self, stub_service):
        alice_before = stub_service.fetch_unread(user_id=ALICE)
        assert {m.id for m in alice_before} == {"m1", "m2"}

        assert stub_service.mark_as_read("m1", user_id=ALICE) is True

        alice_after = stub_service.fetch_unread(user_id=ALICE)
        bob_view = stub_service.fetch_unread(user_id=BOB)

        assert {m.id for m in alice_after} == {"m2"}
        assert {m.id for m in bob_view} == {"m1", "m2"}

    def test_summarize_requires_identity(self, stub_service):
        with pytest.raises((PermissionError, ValueError, TypeError)):
            stub_service.summarize("m1")


# ---------------------------------------------------------------------------
# No credential values in results, events, or errors
# ---------------------------------------------------------------------------


class TestNoSecretLeakage:
    def test_secrets_never_appear_in_results_events_or_errors(self, monkeypatch, fake_creds):
        published: list[tuple[str, dict[str, Any]]] = []

        class Bus:
            def publish(self, topic, payload):
                published.append((topic, payload))

        _patch_env(monkeypatch, _two_user_raw_config())
        svc = EmailService(event_bus=Bus())
        svc.credential_manager = fake_creds

        svc.fetch_unread(user_id=ALICE)
        result = svc.send(to="x@e.com", subject="s", body="b", user_id=ALICE)
        try:
            svc.send(to="x@e.com", subject="s", body="b", user_id=ALICE, account_id="bob-personal")
        except PermissionError as exc:
            error_text = str(exc)
        else:
            error_text = ""

        blob = repr(published) + repr(result) + error_text
        assert SECRET_MARKER not in blob
        assert ALICE_CRED_REF in repr(fake_creds.lookups)  # sanity: lookups happened


# ---------------------------------------------------------------------------
# Per-user defaults
# ---------------------------------------------------------------------------


class TestConnectOwnership:
    def test_connect_requires_user_when_accounts_configured(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        assert svc.connect() is False
        assert fake_creds.lookups == []

    def test_connect_resolves_only_own_account(self, monkeypatch, fake_creds):
        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)

        assert svc.connect("james") is False  # no account for james
        assert fake_creds.lookups == []

        assert svc.connect(BOB) is True
        assert fake_creds.lookups == [BOB_CRED_REF]


class TestCredentialRefIntegrity:
    def test_only_the_definition_credential_ref_is_used(self, monkeypatch, fake_creds):
        """A tampered entry credentials_key can never redirect credential lookup."""
        raw = _two_user_raw_config()
        # Malicious/typo'd config: alice's authorization entry names Bob's ref.
        raw["users"][ALICE]["email_accounts"][0]["credentials_key"] = BOB_CRED_REF

        svc = _make_service(monkeypatch, raw, fake_creds)
        svc.fetch_unread(user_id=ALICE)

        # Credential lookup uses only the authorized account definition's ref.
        assert fake_creds.lookups == [ALICE_CRED_REF]

    def test_no_api_accepts_a_credential_reference(self, monkeypatch, fake_creds):
        """send() has no parameter through which a caller can name a credential."""
        import inspect

        from rex.email_service import EmailService as Svc

        for method in (Svc.send, Svc.fetch_unread, Svc.triage_unread, Svc.mark_as_read):
            params = set(inspect.signature(method).parameters)
            assert not params & {"credential_ref", "credentials_key", "credential"}


class TestEventScoping:
    def test_private_fields_only_on_user_scoped_topic(self, monkeypatch, fake_creds):
        published: list[tuple[str, dict[str, Any]]] = []

        class Bus:
            def publish(self, topic, payload):
                published.append((topic, payload))

        _patch_env(monkeypatch, _two_user_raw_config())
        svc = EmailService(event_bus=Bus())
        svc.credential_manager = fake_creds

        svc.fetch_unread(user_id=ALICE)

        shared = [p for t, p in published if t == "email.unread"]
        private = [p for t, p in published if t == f"email.unread.user.{ALICE}"]
        assert shared and private

        # The shared topic carries only the safe envelope.
        assert "messages" not in shared[0]
        assert "emails" not in shared[0]
        assert shared[0]["user_id"] == ALICE
        # Private payload goes only to the owner-scoped topic.
        assert private[0]["messages"]
        # No payload for Alice's mail was published on Bob's topic.
        assert all(t != f"email.unread.user.{BOB}" for t, _ in published)

    def test_triage_events_are_user_scoped(self, monkeypatch, fake_creds):
        published: list[tuple[str, dict[str, Any]]] = []

        class Bus:
            def publish(self, topic, payload):
                published.append((topic, payload))

        _patch_env(monkeypatch, _two_user_raw_config())
        svc = EmailService(event_bus=Bus())
        svc.credential_manager = fake_creds

        svc.triage_unread(user_id=BOB)

        shared = [p for t, p in published if t == "email.triaged"]
        assert shared and "triaged" not in shared[0]
        assert shared[0]["user_id"] == BOB
        private = [p for t, p in published if t == f"email.triaged.user.{BOB}"]
        assert private and private[0]["triaged"]


class TestScheduledTriage:
    def test_triage_runs_per_stored_owner(self, monkeypatch):
        import rex.config_manager as config_manager
        from rex.services import _run_email_triage

        monkeypatch.setattr(config_manager, "load_config", lambda *a, **k: _two_user_raw_config())

        calls: list[str] = []

        class FakeEmail:
            def triage_unread(self, limit: int = 10, *, user_id=None, account_id=None):
                calls.append(user_id)
                return []

        _run_email_triage(FakeEmail())
        assert calls == [ALICE, BOB]

    def test_one_owner_failure_does_not_affect_others(self, monkeypatch):
        import rex.config_manager as config_manager
        from rex.services import _run_email_triage

        monkeypatch.setattr(config_manager, "load_config", lambda *a, **k: _two_user_raw_config())

        calls: list[str] = []

        class FakeEmail:
            def triage_unread(self, limit: int = 10, *, user_id=None, account_id=None):
                calls.append(user_id)
                if user_id == ALICE:
                    raise RuntimeError("alice IMAP down")
                return []

        _run_email_triage(FakeEmail())  # must not raise
        assert calls == [ALICE, BOB]

    def test_legacy_job_without_owners_runs_default_only(self, monkeypatch):
        import rex.config_manager as config_manager
        from rex.services import _run_email_triage

        monkeypatch.setattr(config_manager, "load_config", lambda *a, **k: {})

        calls: list[str] = []

        class FakeEmail:
            def triage_unread(self, limit: int = 10, *, user_id=None, account_id=None):
                calls.append(user_id)
                return []

        _run_email_triage(FakeEmail())
        assert calls == ["default"]


class TestOpenClawEmailTool:
    def test_fails_closed_without_user_id(self, monkeypatch):
        from unittest.mock import patch

        from rex.openclaw.tools.email_tool import send_email

        with patch("rex.openclaw.tools.email_tool._get_email_service") as get_svc:
            result = send_email("x@example.com", "s", "b")

        assert result["ok"] is False
        get_svc.assert_not_called()

    def test_fails_closed_on_invalid_user_id(self):
        from unittest.mock import patch

        from rex.openclaw.tools.email_tool import send_email

        with patch("rex.openclaw.tools.email_tool._get_email_service") as get_svc:
            result = send_email("x@example.com", "s", "b", _user_id="../evil")

        assert result["ok"] is False
        get_svc.assert_not_called()

    def test_cannot_send_through_foreign_account(self, monkeypatch, fake_creds):
        from unittest.mock import patch

        from rex.openclaw.tools.email_tool import send_email

        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)
        with patch("rex.openclaw.tools.email_tool._get_email_service", return_value=svc):
            result = send_email("x@example.com", "s", "b", _user_id=BOB, account_id="alice-work")

        assert result["ok"] is False
        assert fake_creds.lookups == []
        assert SECRET_MARKER not in str(result)


class TestGmailPerUserToken:
    def test_named_user_never_inherits_global_gmail_token(self, monkeypatch):
        from rex.integrations.email_service import create_email_service_for_user

        monkeypatch.setenv("GMAIL_ACCESS_TOKEN", "global-secret-token")
        raw = {"email": {"provider": "gmail"}}

        svc, provider = create_email_service_for_user("james", raw)
        assert svc is None
        assert provider == "none"

    def test_default_profile_keeps_legacy_gmail_env(self, monkeypatch):
        from rex.integrations.email_service import create_email_service_for_user

        monkeypatch.setenv("GMAIL_ACCESS_TOKEN", "global-secret-token")
        raw = {"email": {"provider": "gmail"}}

        svc, provider = create_email_service_for_user("default", raw)
        assert provider == "gmail"
        assert svc is not None
        assert svc._gmail_headers()["Authorization"] == "Bearer global-secret-token"

    def test_named_user_uses_own_token(self, monkeypatch):
        from rex.integrations.email_service import create_email_service_for_user

        monkeypatch.setenv("GMAIL_ACCESS_TOKEN", "global-secret-token")
        monkeypatch.setenv("EMAIL_ALICE_GMAIL", "alice-own-token")
        raw = {
            "users": {
                ALICE: {
                    "email_accounts": [
                        {
                            "account_id": "alice-gmail",
                            "backend": "gmail",
                            "credentials_key": "EMAIL_ALICE_GMAIL",
                        }
                    ]
                }
            }
        }

        svc, provider = create_email_service_for_user(ALICE, raw)
        assert provider == "gmail"
        assert svc is not None
        assert svc._gmail_headers()["Authorization"] == "Bearer alice-own-token"

    def test_named_user_without_token_fails_closed(self, monkeypatch):
        from rex.integrations.email_service import create_email_service_for_user

        monkeypatch.setenv("GMAIL_ACCESS_TOKEN", "global-secret-token")
        monkeypatch.delenv("EMAIL_ALICE_GMAIL", raising=False)
        raw = {
            "users": {
                ALICE: {
                    "email_accounts": [
                        {
                            "account_id": "alice-gmail",
                            "backend": "gmail",
                            "credentials_key": "EMAIL_ALICE_GMAIL",
                        }
                    ]
                }
            }
        }

        svc, provider = create_email_service_for_user(ALICE, raw)
        assert svc is None
        assert provider == "none"

    def test_invalid_identity_fails_before_token_read(self):
        import pytest as _pytest

        from rex.integrations.email_service import create_email_service_for_user

        with _pytest.raises(PermissionError):
            create_email_service_for_user("../evil", {})


class TestElectronBridgeIdentity:
    def test_explicit_user_payload_wins(self):
        from bridge import rex_email_bridge

        assert rex_email_bridge._resolve_user({"user": ALICE}) == ALICE

    def test_invalid_explicit_user_fails_closed(self):
        from bridge import rex_email_bridge

        assert rex_email_bridge._resolve_user({"user": "../evil"}) is None

    def test_list_for_named_user_without_provider_is_unconfigured(self, monkeypatch):
        import rex.config_manager as config_manager
        from bridge import rex_email_bridge

        monkeypatch.setenv("GMAIL_ACCESS_TOKEN", "global-secret-token")
        monkeypatch.setattr(
            config_manager,
            "load_config",
            lambda *a, **k: {"email": {"provider": "gmail"}},
        )

        result = rex_email_bridge._handle_list("james", 10)
        assert result == {"ok": True, "messages": [], "configured": False}

    def test_bridge_response_never_contains_credentials(self, monkeypatch):
        import rex.config_manager as config_manager
        from bridge import rex_email_bridge

        monkeypatch.setenv("GMAIL_ACCESS_TOKEN", "global-secret-token")
        monkeypatch.setattr(
            config_manager,
            "load_config",
            lambda *a, **k: {"email": {"provider": "gmail"}},
        )

        result = rex_email_bridge._handle_list("james", 10)
        assert "global-secret-token" not in repr(result)


class TestServiceRecreationPersistence:
    def test_ownership_and_defaults_survive_service_recreation(self, monkeypatch, fake_creds):
        """A restarted process (new service over the same config) preserves
        account ownership and per-user default selection."""
        raw = _two_user_raw_config()
        raw["users"][ALICE]["email_accounts"].append(
            {"account_id": "bob-personal", "backend": "imap", "credentials_key": BOB_CRED_REF}
        )
        del raw["users"][BOB]
        raw["users"][ALICE]["default_email_account_id"] = "bob-personal"

        first = _make_service(monkeypatch, raw, fake_creds)
        assert [m.subject for m in first.fetch_unread(user_id=ALICE)] == [f"mail-via-{BOB_HOST}"]

        second = _make_service(monkeypatch, raw, FakeCredentialManager())
        assert [m.subject for m in second.fetch_unread(user_id=ALICE)] == [f"mail-via-{BOB_HOST}"]
        with pytest.raises(PermissionError):
            second.fetch_unread(user_id="james", account_id="alice-work")

    def test_revoked_assignment_takes_effect_without_restart(self, monkeypatch, fake_creds):
        """A long-lived service must honour a config change that revokes a
        user's account assignment — no stale-authorization window."""
        import rex.config_manager as config_manager
        import rex.email_accounts as email_accounts

        svc = _make_service(monkeypatch, _two_user_raw_config(), fake_creds)
        assert [m.subject for m in svc.fetch_unread(user_id=BOB)] == [f"mail-via-{BOB_HOST}"]

        revoked = _two_user_raw_config()
        del revoked["users"][BOB]
        monkeypatch.setattr(config_manager, "load_config", lambda *a, **k: revoked)
        monkeypatch.setattr(email_accounts, "config_stamp", lambda: 12345)

        assert svc.fetch_unread(user_id=BOB) == []
        result = svc.send(to="x@e.com", subject="s", body="b", user_id=BOB)
        assert result.get("ok") is not True

    def test_per_user_default_survives_config_reload_from_disk(
        self, monkeypatch, fake_creds, tmp_path
    ):
        """Per-user defaults written to rex_config.json survive a reload."""
        import json as _json

        from rex.email_accounts import EmailAccountResolver

        raw = _two_user_raw_config()
        raw["users"][ALICE]["email_accounts"].append(
            {"account_id": "bob-personal", "backend": "imap", "credentials_key": BOB_CRED_REF}
        )
        del raw["users"][BOB]
        raw["users"][ALICE]["default_email_account_id"] = "bob-personal"

        config_file = tmp_path / "rex_config.json"
        config_file.write_text(_json.dumps(raw), encoding="utf-8")

        reloaded = _json.loads(config_file.read_text(encoding="utf-8"))
        resolver = EmailAccountResolver.from_raw_config(reloaded)
        assert resolver.default_account_id_for_user(ALICE) == "bob-personal"


class TestPerUserDefaults:
    def test_user_default_account_selected_over_config_order(self, monkeypatch, fake_creds):
        raw = _two_user_raw_config()
        # Alice owns both accounts; her personal default is the second one.
        raw["users"][ALICE]["email_accounts"].append(
            {"account_id": "bob-personal", "backend": "imap", "credentials_key": BOB_CRED_REF}
        )
        del raw["users"][BOB]
        raw["users"][ALICE]["default_email_account_id"] = "bob-personal"

        svc = _make_service(monkeypatch, raw, fake_creds)
        msgs = svc.fetch_unread(user_id=ALICE)

        assert [m.subject for m in msgs] == [f"mail-via-{BOB_HOST}"]

    def test_one_users_default_does_not_affect_another(self, monkeypatch, fake_creds):
        raw = _two_user_raw_config()
        raw["users"][ALICE]["default_email_account_id"] = "alice-work"

        svc = _make_service(monkeypatch, raw, fake_creds)
        msgs = svc.fetch_unread(user_id=BOB)

        assert [m.subject for m in msgs] == [f"mail-via-{BOB_HOST}"]

    def test_foreign_default_is_ignored_fail_closed(self, monkeypatch, fake_creds):
        """A configured default pointing at a foreign account must not grant access."""
        raw = _two_user_raw_config()
        raw["users"][BOB]["default_email_account_id"] = "alice-work"

        svc = _make_service(monkeypatch, raw, fake_creds)
        msgs = svc.fetch_unread(user_id=BOB)

        # Falls back to Bob's own account; never Alice's.
        assert [m.subject for m in msgs] == [f"mail-via-{BOB_HOST}"]
        assert ALICE_CRED_REF not in fake_creds.lookups
