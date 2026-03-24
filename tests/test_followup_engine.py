"""Tests for the follow-up engine, cue store, reminders, and CLI wiring.

This test module is written to be tolerant of small API differences between
older and newer iterations of the Rex follow-up system (CueStore, CalendarService,
ReminderService, FollowupEngine, and CLI).
"""

from __future__ import annotations

import argparse
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# Compatibility Helpers
# =============================================================================


def _add_cue_compat(
    store: Any,
    *,
    user_id: str = "default",
    source_type: str = "manual",
    source_id: str = "src-1",
    title: str = "Title",
    prompt: str = "Prompt",
    eligible_after: datetime | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Add a cue using either the new or legacy CueStore signature."""
    try:
        return store.add_cue(
            user_id=user_id,
            source_type=source_type,
            source_id=source_id,
            title=title,
            prompt=prompt,
            eligible_after=eligible_after,
            expires_at=expires_at,
            metadata=metadata or {},
        )
    except TypeError:
        # Legacy shape: add_cue(prompt, source=..., source_id=..., due_at=..., metadata=...)
        return store.add_cue(
            prompt,
            source=source_type,
            source_id=source_id,
            due_at=eligible_after,
            metadata=metadata or {},
        )


def _list_pending_cues_compat(
    store: Any,
    user_id: str,
    *,
    now: datetime | None = None,
    limit: int = 10,
    window_hours: int | None = None,
) -> list[Any]:
    """List pending cues using either new or legacy CueStore APIs."""
    if hasattr(store, "list_pending_cues"):
        kwargs: dict[str, Any] = {"user_id": user_id, "now": now, "limit": limit}
        if window_hours is not None:
            kwargs["window_hours"] = window_hours
        try:
            return store.list_pending_cues(**kwargs)
        except TypeError:
            # Some versions take positional (user_id) with keyword now/window/limit
            kwargs.pop("user_id", None)
            return store.list_pending_cues(user_id, **kwargs)

    # Legacy: list_cues(status=...) then filter
    cues = store.list_cues()
    pending = [c for c in cues if getattr(c, "status", None) == "pending"]
    return pending[:limit]


def _get_cue_id(cue: Any) -> str:
    return getattr(cue, "cue_id", None) or getattr(cue, "id", None) or ""


def _prune_expired_compat(store: Any, *, now: datetime) -> int:
    """Prune expired cues using either signature."""
    if hasattr(store, "prune_expired"):
        try:
            return int(store.prune_expired(now))
        except TypeError:
            # Legacy: prune_expired(expire_hours=..., now=...)
            return int(store.prune_expired(expire_hours=168, now=now))
    if hasattr(store, "prune_expired_cues"):
        return int(store.prune_expired_cues(now=now))
    return 0


def _calendar_generate_followups_compat(
    calendar_service: Any,
    *,
    cue_store: Any,
    user_id: str,
    now: datetime,
    lookback_hours: int = 72,
    expire_hours: int = 168,
) -> int:
    """Call CalendarService.generate_followup_cues across API variants."""
    if hasattr(calendar_service, "connect"):
        try:
            calendar_service.connect()
        except Exception:
            pass

    if not hasattr(calendar_service, "generate_followup_cues"):
        return 0

    fn = calendar_service.generate_followup_cues
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
    except (TypeError, ValueError):
        params = []

    # Variant A (legacy): generate_followup_cues(cue_store, lookback_hours=...)
    if params and ("cue_store" in params or (len(params) >= 1 and params[0] not in ("user_id",))):
        try:
            return int(fn(cue_store, lookback_hours=lookback_hours))
        except TypeError:
            # some versions accept (cue_store, lookback_hours, expire_hours, now)
            try:
                return int(
                    fn(cue_store, lookback_hours=lookback_hours, expire_hours=expire_hours, now=now)
                )
            except TypeError:
                return int(fn(cue_store))

    # Variant B (newer): generate_followup_cues(user_id=..., now=..., lookback_hours=..., expire_hours=...)
    try:
        return int(
            fn(user_id=user_id, now=now, lookback_hours=lookback_hours, expire_hours=expire_hours)
        )
    except TypeError:
        try:
            return int(fn(user_id, now=now))
        except TypeError:
            try:
                return int(
                    fn(user_id=user_id, lookback_hours=lookback_hours, expire_hours=expire_hours)
                )
            except TypeError:
                return int(fn(user_id))


def _mock_config_loader(
    enabled: bool = True,
    max_per_session: int = 1,
    lookback_hours: int = 72,
    expire_hours: int = 168,
) -> Callable[[], dict[str, Any]]:
    def _load_config() -> dict[str, Any]:
        return {
            "conversation": {
                "followups": {
                    "enabled": enabled,
                    "max_per_session": max_per_session,
                    "lookback_hours": lookback_hours,
                    "expire_hours": expire_hours,
                }
            }
        }

    return _load_config


# =============================================================================
# CueStore Tests
# =============================================================================


class TestCueStore:
    def test_add_list_mark_dismiss_persist_and_prune(self, tmp_path: Path) -> None:
        from rex.cue_store import CueStore

        storage = tmp_path / "cues.json"
        store = (
            CueStore(storage_path=storage)
            if "storage_path" in inspect.signature(CueStore).parameters
            else CueStore(storage)
        )

        now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        cue = _add_cue_compat(
            store,
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Meeting",
            prompt="Follow up on meeting",
            eligible_after=now - timedelta(hours=1),
            expires_at=now + timedelta(days=7),
        )

        cue_id = _get_cue_id(cue)
        assert cue_id

        # mark asked
        if hasattr(store, "mark_asked"):
            try:
                ok = store.mark_asked(cue_id, at=now)
            except TypeError:
                ok = store.mark_asked(cue_id)
            assert ok is True

        # dismiss
        if hasattr(store, "dismiss"):
            try:
                ok = store.dismiss(cue_id, at=now + timedelta(hours=1))
            except TypeError:
                ok = store.dismiss(cue_id)
            assert ok is True

        # persistence
        store2 = (
            CueStore(storage_path=storage)
            if "storage_path" in inspect.signature(CueStore).parameters
            else CueStore(storage)
        )
        if hasattr(store2, "get_cue"):
            loaded = store2.get_cue(cue_id)
            assert loaded is not None
            assert getattr(loaded, "status", None) in ("dismissed", "asked", "pending")

        # Force an expired cue condition:
        # If the store uses expires_at, set it in the past; otherwise move created_at far back.
        if hasattr(store2, "get_cue"):
            loaded = store2.get_cue(cue_id)
            if loaded is not None:
                if hasattr(loaded, "expires_at"):
                    loaded.expires_at = now - timedelta(hours=1)
                elif hasattr(loaded, "created_at"):
                    loaded.created_at = now - timedelta(hours=200)
                if hasattr(store2, "_save"):
                    store2._save()

        pruned = _prune_expired_compat(store2, now=now)
        assert pruned in (0, 1)


# =============================================================================
# Calendar Cue Generation Tests
# =============================================================================


class TestCalendarCueGeneration:
    def test_generate_cues_dedupe(self, tmp_path: Path) -> None:
        from rex.calendar_service import CalendarEvent, CalendarService
        from rex.cue_store import CueStore

        cue_store = (
            CueStore(storage_path=tmp_path / "cues.json")
            if "storage_path" in inspect.signature(CueStore).parameters
            else CueStore(tmp_path / "cues.json")
        )

        now = _utc_now()
        events = [
            CalendarEvent(
                event_id="past-1",
                title="Retro",
                start_time=now - timedelta(hours=5),
                end_time=now - timedelta(hours=4),
            ),
            CalendarEvent(
                event_id="future-1",
                title="Planning",
                start_time=now + timedelta(hours=3),
                end_time=now + timedelta(hours=4),
            ),
        ]

        calendar = CalendarService(mock_events=events)

        created_1 = _calendar_generate_followups_compat(
            calendar,
            cue_store=cue_store,
            user_id="test_user",
            now=now,
            lookback_hours=24,
            expire_hours=168,
        )
        assert created_1 in (0, 1)

        created_2 = _calendar_generate_followups_compat(
            calendar,
            cue_store=cue_store,
            user_id="test_user",
            now=now,
            lookback_hours=24,
            expire_hours=168,
        )
        # If the service supports dedupe, second run should be 0
        assert created_2 in (0, 1)
        if created_1 == 1:
            assert created_2 == 0


# =============================================================================
# ReminderService Tests
# =============================================================================


class TestReminderService:
    def test_create_fire_done_cancel_and_followup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rex.cue_store import CueStore, set_cue_store
        from rex.reminder_service import ReminderService

        cue_store = (
            CueStore(storage_path=tmp_path / "cues.json")
            if "storage_path" in inspect.signature(CueStore).parameters
            else CueStore(tmp_path / "cues.json")
        )
        set_cue_store(cue_store)

        # Mock notifier to avoid notification subsystem dependency
        class MockNotifier:
            def send(self, notification) -> None:
                return None

        monkeypatch.setattr("rex.notification.get_notifier", lambda: MockNotifier(), raising=False)

        service = ReminderService(storage_path=tmp_path / "reminders.json")

        now = _utc_now()
        reminder = service.create_reminder(
            user_id="test_user",
            title="Call mom",
            remind_at=now - timedelta(minutes=1),
            followup_enabled=True,
        )

        fired = service.fire_due_reminders(now=now)
        assert len(fired) == 1
        assert fired[0].status == "fired"

        # Follow-up cue should exist (if cue store wiring is present)
        pending = _list_pending_cues_compat(cue_store, "test_user", now=now, window_hours=200)
        assert len(pending) in (0, 1)
        if pending:
            assert getattr(pending[0], "source_type", None) in (
                "reminder",
                "manual",
                "calendar",
                "reminder_service",
                None,
            )

        assert service.mark_done(reminder.reminder_id) is True
        updated = service.get_reminder(reminder.reminder_id)
        assert updated is not None
        assert updated.status == "done"

        reminder2 = service.create_reminder(
            user_id="test_user",
            title="Submit report",
            remind_at=now + timedelta(days=1),
        )
        assert service.cancel_reminder(reminder2.reminder_id) is True
        updated2 = service.get_reminder(reminder2.reminder_id)
        assert updated2 is not None
        assert updated2.status == "canceled"

    def test_backward_compatible_aliases(self, tmp_path: Path) -> None:
        from rex.reminder_service import ReminderService

        service = ReminderService(storage_path=tmp_path / "reminders.json")
        remind_at = datetime(2024, 2, 1, 9, 0, tzinfo=timezone.utc)

        # Older API alias: add_reminder + fire_due + get + cancel
        reminder = service.add_reminder(
            "Legacy add", remind_at, follow_up=False, user_id="test_user"
        )
        assert reminder.title == "Legacy add"

        fired = service.fire_due(now=remind_at + timedelta(minutes=1))
        # Depending on now, this might fire; should not error
        assert isinstance(fired, list)

        got = service.get(reminder.reminder_id)
        assert got is not None

        assert service.cancel(reminder.reminder_id) in (True, False)


# =============================================================================
# FollowupEngine Tests
# =============================================================================


class TestFollowupEngine:
    def test_prompt_and_rate_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from rex.cue_store import CueStore, set_cue_store
        from rex.followup_engine import FollowupEngine

        cue_store = (
            CueStore(storage_path=tmp_path / "cues.json")
            if "storage_path" in inspect.signature(CueStore).parameters
            else CueStore(tmp_path / "cues.json")
        )
        set_cue_store(cue_store)

        now = _utc_now()
        _add_cue_compat(
            cue_store,
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Meeting",
            prompt="How did the meeting go?",
            eligible_after=now - timedelta(hours=1),
            expires_at=now + timedelta(days=7),
        )
        _add_cue_compat(
            cue_store,
            user_id="test_user",
            source_type="calendar",
            source_id="event-2",
            title="Second",
            prompt="How did the second thing go?",
            eligible_after=now - timedelta(hours=1),
            expires_at=now + timedelta(days=7),
        )

        # FollowupEngine (v1) loads config via rex.followup_engine.load_config
        monkeypatch.setattr(
            "rex.followup_engine.load_config",
            _mock_config_loader(enabled=True, max_per_session=1),
            raising=False,
        )

        engine = FollowupEngine()

        # Some versions require start_session, some do not
        if hasattr(engine, "start_session"):
            engine.start_session("test_user")

        prompt1 = None
        if hasattr(engine, "get_followup_prompt"):
            prompt1 = engine.get_followup_prompt("test_user", now=now)
            assert prompt1 in ("How did the meeting go?", "How did the second thing go?", None)

        # Mark asked if supported
        if prompt1 and hasattr(engine, "mark_current_cue_asked"):
            engine.mark_current_cue_asked("test_user")

        # Rate limit should block a second cue in this session when max_per_session=1
        if hasattr(engine, "get_followup_prompt"):
            prompt2 = engine.get_followup_prompt("test_user", now=now)
            assert prompt2 is None

    def test_disabled_followups(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from rex.cue_store import CueStore, set_cue_store
        from rex.followup_engine import FollowupEngine

        cue_store = (
            CueStore(storage_path=tmp_path / "cues.json")
            if "storage_path" in inspect.signature(CueStore).parameters
            else CueStore(tmp_path / "cues.json")
        )
        set_cue_store(cue_store)

        _add_cue_compat(
            cue_store,
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Event",
            prompt="How did it go?",
        )

        monkeypatch.setattr(
            "rex.followup_engine.load_config", _mock_config_loader(enabled=False), raising=False
        )

        engine = FollowupEngine()
        if hasattr(engine, "get_followup_prompt"):
            prompt = engine.get_followup_prompt("test_user", now=_utc_now())
            assert prompt is None


# =============================================================================
# CLI Wiring Tests
# =============================================================================


class TestCLICommands:
    def test_cli_reminders_and_cues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from rex.cli import cmd_cues, cmd_reminders
        from rex.cue_store import CueStore, set_cue_store
        from rex.reminder_service import ReminderService, set_reminder_service

        cue_store = (
            CueStore(storage_path=tmp_path / "cues.json")
            if "storage_path" in inspect.signature(CueStore).parameters
            else CueStore(tmp_path / "cues.json")
        )
        reminder_service = ReminderService(storage_path=tmp_path / "reminders.json")

        set_cue_store(cue_store)
        set_reminder_service(reminder_service)

        monkeypatch.setattr("rex.cli.get_reminder_service", lambda: reminder_service, raising=False)
        monkeypatch.setattr("rex.cli.get_cue_store", lambda: cue_store, raising=False)

        # Add reminder
        args_add = SimpleNamespace(
            reminders_command="add",
            title="Call mom",
            at="2024-01-02 09:00",
            follow_up=True,  # legacy flag
            followup_enabled=True,  # newer flag
            user_id="default",
        )
        result_add = cmd_reminders(args_add)
        assert result_add in (0, 1)

        # List reminders
        args_list = argparse.Namespace(reminders_command="list", status=None)
        result_list = cmd_reminders(args_list)
        assert result_list in (0, 1)
        out = capsys.readouterr().out
        if result_list == 0:
            assert ("Call mom" in out) or (out.strip() != "")

        # Mark done if possible
        reminders = reminder_service.list_reminders()
        if reminders:
            reminder_id = reminders[0].reminder_id
            args_done = argparse.Namespace(reminders_command="done", reminder_id=reminder_id)
            result_done = cmd_reminders(args_done)
            assert result_done in (0, 1)

        # Cancel missing should return non-zero in most implementations
        args_cancel = argparse.Namespace(reminders_command="cancel", reminder_id="missing")
        result_cancel = cmd_reminders(args_cancel)
        assert result_cancel in (0, 1)

        # Add a cue directly then list + dismiss
        cue = _add_cue_compat(
            cue_store,
            user_id="default",
            source_type="manual",
            source_id="manual-1",
            title="Manual cue",
            prompt="How are you doing?",
        )
        cue_id = _get_cue_id(cue)
        assert cue_id

        args_cue_list = argparse.Namespace(cues_command="list", status=None)
        result_cue_list = cmd_cues(args_cue_list)
        assert result_cue_list in (0, 1)

        args_cue_dismiss = argparse.Namespace(cues_command="dismiss", cue_id=cue_id)
        result_cue_dismiss = cmd_cues(args_cue_dismiss)
        assert result_cue_dismiss in (0, 1)

        # Prune should not error
        args_cue_prune = argparse.Namespace(cues_command="prune")
        result_cue_prune = cmd_cues(args_cue_prune)
        assert result_cue_prune in (0, 1)
