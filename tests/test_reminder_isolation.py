"""US-303: per-user isolation for reminders and their scheduled execution.

Covers:
- Cross-user list/get/complete/cancel/delete/update (snooze) denial
- User-scoped manual firing cannot trigger another user's due reminders
- Background (system-context) firing executes as the stored owner
- Missing / blank / malformed / traversal identity fails closed
- Invalid persisted owner IDs are quarantined, preserved, and inaccessible
- Ownership survives service restart (reload from disk)
- Legacy records without a user_id belong to the distinct ``default``
  profile only
- Bridge and CLI paths resolve the caller identity, preserve it on create,
  and enforce ownership on mutation
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.reminder_service import ReminderService, set_reminder_service


def _future(hours: int = 2) -> datetime:
    return datetime.now(UTC) + timedelta(hours=hours)


def _past(hours: int = 1) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


@pytest.fixture()
def svc(tmp_path: Path) -> ReminderService:
    return ReminderService(storage_path=tmp_path / "reminders.json")


# =============================================================================
# Service-level cross-user denial
# =============================================================================


class TestServiceOwnership:
    def test_list_is_scoped_to_requesting_user(self, svc: ReminderService) -> None:
        svc.create_reminder("alice", "Alice appointment", _future())
        svc.create_reminder("bob", "Bob meeting", _future())

        alice_view = svc.list_reminders(user_id="alice")
        assert [r.title for r in alice_view] == ["Alice appointment"]
        assert all(r.user_id == "alice" for r in alice_view)

        # Unscoped listing must be impossible, not merely optional.
        with pytest.raises(TypeError):
            svc.list_reminders()  # type: ignore[call-arg]

    def test_get_denies_non_owner(self, svc: ReminderService) -> None:
        r = svc.create_reminder("bob", "Bob secret", _future())
        assert svc.get_reminder(r.reminder_id, "alice") is None
        assert svc.get_reminder(r.reminder_id, "bob") is not None

    def test_mark_done_denies_non_owner(self, svc: ReminderService) -> None:
        r = svc.create_reminder("bob", "Bob task", _future())
        assert svc.mark_done(r.reminder_id, "alice") is False
        assert svc.get_reminder(r.reminder_id, "bob").status == "pending"
        assert svc.mark_done(r.reminder_id, "bob") is True

    def test_cancel_denies_non_owner(self, svc: ReminderService) -> None:
        r = svc.create_reminder("bob", "Bob task", _future())
        assert svc.cancel_reminder(r.reminder_id, "alice") is False
        assert svc.get_reminder(r.reminder_id, "bob").status == "pending"

    def test_delete_denies_non_owner(self, svc: ReminderService) -> None:
        r = svc.create_reminder("bob", "Bob task", _future())
        assert svc.delete_reminder(r.reminder_id, "alice") is False
        assert svc.get_reminder(r.reminder_id, "bob") is not None

    def test_update_snooze_denies_non_owner(self, svc: ReminderService) -> None:
        original_time = _future()
        r = svc.create_reminder("bob", "Bob task", original_time)

        snoozed = svc.update_reminder(r.reminder_id, "alice", remind_at=_future(48))
        assert snoozed is None
        unchanged = svc.get_reminder(r.reminder_id, "bob")
        assert unchanged.remind_at == original_time.astimezone(UTC)

        assert svc.update_reminder(r.reminder_id, "bob", title="Renamed") is not None

    def test_create_cannot_overwrite_existing_id(self, svc: ReminderService) -> None:
        r = svc.create_reminder("alice", "Alice original", _future())
        with pytest.raises(ValueError):
            svc.create_reminder("mallory", "Takeover", _future(), reminder_id=r.reminder_id)
        assert svc.get_reminder(r.reminder_id, "alice").user_id == "alice"


# =============================================================================
# Scheduled / background execution
# =============================================================================


class TestScheduledExecution:
    def test_user_scoped_fire_cannot_trigger_other_users_work(
        self, svc: ReminderService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(svc, "_send_notification", lambda reminder: None)
        svc.create_reminder("alice", "Alice due", _past())
        bob_due = svc.create_reminder("bob", "Bob due", _past())

        fired = svc.fire_due_reminders(user_id="alice")

        assert [r.user_id for r in fired] == ["alice"]
        assert svc.get_reminder(bob_due.reminder_id, "bob").status == "pending"

    def test_background_fire_executes_as_stored_owner(
        self, svc: ReminderService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """System-context firing attributes work to the reminder's owner,
        not whichever user is currently active in the identity chain."""
        # Simulate a different currently-active user.
        monkeypatch.setattr("rex.identity.resolve_active_user", lambda *a, **k: "alice")

        notifications = []

        class RecordingNotifier:
            def send(self, notification) -> None:
                notifications.append(notification)

        monkeypatch.setattr("rex.notification.get_notifier", lambda: RecordingNotifier())

        cue_store = MagicMock()
        monkeypatch.setattr("rex.cue_store.get_cue_store", lambda: cue_store)

        svc.create_reminder("bob", "Bob background task", _past(), followup_enabled=True)

        fired = svc.fire_due_reminders()  # same call the scheduler job makes

        assert len(fired) == 1
        assert fired[0].user_id == "bob"
        assert notifications[0].metadata["user_id"] == "bob"
        assert cue_store.add_cue.call_args.kwargs["user_id"] == "bob"


# =============================================================================
# Fail-closed identity validation
# =============================================================================


class TestFailClosedIdentity:
    @pytest.mark.parametrize("bad_user", ["", "  ", "../evil", "a/b", "..", ".", "a" * 65])
    def test_invalid_identity_rejected_everywhere(
        self, svc: ReminderService, bad_user: str
    ) -> None:
        r = svc.create_reminder("alice", "Alice task", _future())

        with pytest.raises(ValueError):
            svc.create_reminder(bad_user, "X", _future())
        with pytest.raises(ValueError):
            svc.list_reminders(user_id=bad_user)
        with pytest.raises(ValueError):
            svc.get_reminder(r.reminder_id, bad_user)
        with pytest.raises(ValueError):
            svc.mark_done(r.reminder_id, bad_user)
        with pytest.raises(ValueError):
            svc.cancel_reminder(r.reminder_id, bad_user)
        with pytest.raises(ValueError):
            svc.delete_reminder(r.reminder_id, bad_user)
        with pytest.raises(ValueError):
            svc.fire_due_reminders(user_id=bad_user)

    def test_none_identity_rejected(self, svc: ReminderService) -> None:
        with pytest.raises(ValueError):
            svc.list_reminders(user_id=None)  # type: ignore[arg-type]


# =============================================================================
# Persistence, restart, legacy and quarantine behavior
# =============================================================================


class TestPersistenceAndMigration:
    def test_restart_preserves_ownership(self, tmp_path: Path) -> None:
        path = tmp_path / "reminders.json"
        svc1 = ReminderService(storage_path=path)
        a = svc1.create_reminder("alice", "Alice persists", _future())
        svc1.create_reminder("bob", "Bob persists", _future())

        svc2 = ReminderService(storage_path=path)
        assert [r.title for r in svc2.list_reminders(user_id="alice")] == ["Alice persists"]
        assert svc2.get_reminder(a.reminder_id, "bob") is None
        assert svc2.get_reminder(a.reminder_id, "alice").user_id == "alice"

    def test_legacy_record_without_user_id_belongs_to_default_only(self, tmp_path: Path) -> None:
        path = tmp_path / "reminders.json"
        legacy_record = {
            "reminder_id": "rem_legacy000001",
            "title": "Legacy unscoped reminder",
            "remind_at": "2030-01-15T14:00:00Z",
            "status": "pending",
        }
        path.write_text(json.dumps({"reminders": [legacy_record]}), encoding="utf-8")

        svc = ReminderService(storage_path=path)

        # Not visible to, or mutable by, any named user.
        assert svc.list_reminders(user_id="james") == []
        assert svc.get_reminder("rem_legacy000001", "james") is None
        assert svc.mark_done("rem_legacy000001", "james") is False
        assert svc.delete_reminder("rem_legacy000001", "james") is False

        # Accessible when explicitly operating as the default profile.
        default_view = svc.list_reminders(user_id="default")
        assert [r.reminder_id for r in default_view] == ["rem_legacy000001"]
        assert svc.mark_done("rem_legacy000001", "default") is True

    def test_invalid_persisted_identity_is_quarantined_and_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "reminders.json"
        bad_record = {
            "reminder_id": "rem_evil00000001",
            "user_id": "../../etc/passwd",
            "title": "Traversal owner",
            "remind_at": "2030-01-15T14:00:00Z",
            "status": "pending",
        }
        path.write_text(json.dumps({"reminders": [bad_record]}), encoding="utf-8")

        svc = ReminderService(storage_path=path)

        # Inaccessible to everyone, including the default profile.
        assert svc.get_reminder("rem_evil00000001", "default") is None
        assert svc.list_reminders(user_id="default") == []
        assert svc.mark_done("rem_evil00000001", "default") is False
        assert len(svc) == 0
        assert svc.stats()["quarantined"] == 1

        # Preserved (not deleted) across a save triggered by unrelated work.
        svc.create_reminder("alice", "New task", _future())
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved_ids = {r.get("reminder_id") for r in saved["reminders"]}
        assert "rem_evil00000001" in saved_ids
        preserved = next(
            r for r in saved["reminders"] if r.get("reminder_id") == "rem_evil00000001"
        )
        assert preserved["user_id"] == "../../etc/passwd"

        # Still quarantined after another restart.
        svc3 = ReminderService(storage_path=path)
        assert svc3.get_reminder("rem_evil00000001", "default") is None
        assert svc3.stats()["quarantined"] == 1


# =============================================================================
# Bridge (Electron GUI) path
# =============================================================================


def _run_bridge(stdin_data: str) -> dict:
    """Invoke the reminders bridge main() with the given stdin JSON."""
    mod = importlib.import_module("rex_reminders_bridge")

    try:
        payload = json.loads(stdin_data)
        payload.setdefault("data_scope", "private")
        stdin_data = json.dumps(payload)
    except json.JSONDecodeError:
        pass

    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)):
        with patch("sys.stdout", captured):
            try:
                mod.main()
            except SystemExit:
                pass
    return json.loads(captured.getvalue().strip())


class TestBridgeOwnership:
    @pytest.fixture(autouse=True)
    def _isolated_service(self, tmp_path: Path):
        service = ReminderService(storage_path=tmp_path / "reminders.json")
        set_reminder_service(service)
        yield service
        set_reminder_service(None)

    def test_save_records_the_requesting_user(self, _isolated_service) -> None:
        result = _run_bridge(
            json.dumps(
                {
                    "command": "save",
                    "user": "alice",
                    "reminder": {"title": "Alice via GUI", "dueAt": _future().isoformat()},
                }
            )
        )
        assert result["ok"] is True
        stored = _isolated_service.list_reminders(user_id="alice")
        assert [r.title for r in stored] == ["Alice via GUI"]
        assert stored[0].user_id == "alice"

    def test_save_does_not_hardcode_default(self, _isolated_service) -> None:
        _run_bridge(
            json.dumps(
                {
                    "command": "save",
                    "user": "alice",
                    "reminder": {"title": "Owned by alice", "dueAt": _future().isoformat()},
                }
            )
        )
        assert _isolated_service.list_reminders(user_id="default") == []

    def test_list_is_scoped_to_requesting_user(self, _isolated_service) -> None:
        _isolated_service.create_reminder("alice", "Alice item", _future())
        _isolated_service.create_reminder("bob", "Bob item", _future())

        result = _run_bridge(json.dumps({"command": "list", "user": "alice"}))
        assert result["ok"] is True
        assert [r["title"] for r in result["reminders"]] == ["Alice item"]

    def test_delete_denies_non_owner(self, _isolated_service) -> None:
        r = _isolated_service.create_reminder("bob", "Bob item", _future())
        result = _run_bridge(
            json.dumps({"command": "delete", "user": "alice", "id": r.reminder_id})
        )
        assert result["ok"] is False
        assert _isolated_service.get_reminder(r.reminder_id, "bob") is not None

    def test_complete_denies_non_owner(self, _isolated_service) -> None:
        r = _isolated_service.create_reminder("bob", "Bob item", _future())
        result = _run_bridge(
            json.dumps({"command": "complete", "user": "alice", "id": r.reminder_id})
        )
        assert result["ok"] is False
        assert _isolated_service.get_reminder(r.reminder_id, "bob").status == "pending"

    def test_save_update_denies_non_owner_and_does_not_overwrite(self, _isolated_service) -> None:
        r = _isolated_service.create_reminder("alice", "Alice original", _future())
        result = _run_bridge(
            json.dumps(
                {
                    "command": "save",
                    "user": "bob",
                    "reminder": {
                        "id": r.reminder_id,
                        "title": "Hijacked",
                        "dueAt": _future().isoformat(),
                    },
                }
            )
        )
        assert result["ok"] is False
        intact = _isolated_service.get_reminder(r.reminder_id, "alice")
        assert intact.title == "Alice original"
        assert intact.user_id == "alice"

    def test_missing_identity_fails_closed(
        self, _isolated_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rex.identity.resolve_active_user", lambda *a, **k: None)
        for command in ({"command": "list"}, {"command": "delete", "id": "rem_x"}):
            result = _run_bridge(json.dumps(command))
            assert result["ok"] is False
            assert "No active user" in result["error"]

    def test_malformed_explicit_identity_fails_closed(self, _isolated_service) -> None:
        result = _run_bridge(json.dumps({"command": "list", "user": "../evil"}))
        assert result["ok"] is False
        assert "No active user" in result["error"]


# =============================================================================
# CLI path
# =============================================================================


class TestCliOwnership:
    @pytest.fixture(autouse=True)
    def _isolated_service(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        service = ReminderService(storage_path=tmp_path / "reminders.json")
        set_reminder_service(service)
        monkeypatch.setattr("rex.cli.get_reminder_service", lambda: service, raising=False)
        yield service
        set_reminder_service(None)

    def _cmd(self):
        from rex.cli import cmd_reminders

        return cmd_reminders

    def test_add_records_the_requesting_user(self, _isolated_service) -> None:
        args = argparse.Namespace(
            reminders_command="add",
            title="Alice CLI reminder",
            at="2030-01-02 09:00",
            followup=False,
            user="alice",
        )
        assert self._cmd()(args) == 0
        stored = _isolated_service.list_reminders(user_id="alice")
        assert [r.title for r in stored] == ["Alice CLI reminder"]
        assert stored[0].user_id == "alice"

    def test_list_is_scoped_to_requesting_user(
        self, _isolated_service, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _isolated_service.create_reminder("alice", "Alice CLI item", _future())
        _isolated_service.create_reminder("bob", "Bob CLI secret", _future())

        args = argparse.Namespace(reminders_command="list", status=None, user="alice")
        assert self._cmd()(args) == 0
        out = capsys.readouterr().out
        assert "Alice CLI item" in out
        assert "Bob CLI secret" not in out

    def test_done_denies_non_owner(self, _isolated_service) -> None:
        r = _isolated_service.create_reminder("bob", "Bob CLI item", _future())
        args = argparse.Namespace(reminders_command="done", reminder_id=r.reminder_id, user="alice")
        assert self._cmd()(args) == 1
        assert _isolated_service.get_reminder(r.reminder_id, "bob").status == "pending"

    def test_cancel_denies_non_owner(self, _isolated_service) -> None:
        r = _isolated_service.create_reminder("bob", "Bob CLI item", _future())
        args = argparse.Namespace(
            reminders_command="cancel", reminder_id=r.reminder_id, user="alice"
        )
        assert self._cmd()(args) == 1
        assert _isolated_service.get_reminder(r.reminder_id, "bob").status == "pending"

    def test_missing_identity_fails_closed(
        self, _isolated_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rex.identity.resolve_active_user", lambda *a, **k: None)
        args = argparse.Namespace(reminders_command="list", status=None, user=None)
        assert self._cmd()(args) == 1

    def test_malformed_explicit_identity_fails_closed(self, _isolated_service) -> None:
        args = argparse.Namespace(reminders_command="list", status=None, user="../evil")
        assert self._cmd()(args) == 1
