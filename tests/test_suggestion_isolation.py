"""Per-user isolation tests for the proactive suggestion pipeline (issue #303).

Proves that one user can never see, accept, or dismiss another user's
suggestion, that pattern entries feeding suggestion content are scoped per
user, and that missing or invalid identity fails closed.

Covers:
- SuggestionEngine: pending state, session flag, and dismissals per user
- Persistence: dismissals and accepted automations survive an engine restart
  with per-user attribution intact
- IntentRouter: the yes/no intercept only fires for the pending suggestion's
  owner
- ResponseBuilder: pending suggestion text is surfaced only to its owner
- ActionDispatcher: pattern entries are recorded under the requesting user
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from rex.actions.dispatcher import ActionDispatcher
from rex.intent.router import IntentRouter
from rex.response.builder import ResponseBuilder
from rex.suggestions.engine import SuggestionEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern(
    entity_id: str = "light.kitchen_ceiling",
    service: str = "turn_on",
    start_hour: int = 7,
    frequency: int = 5,
) -> dict[str, Any]:
    return {
        "pattern": f"{service} {entity_id} around {start_hour:02d}:00",
        "frequency": frequency,
        "suggested_automation": f"Automate: {service} {entity_id} daily at {start_hour:02d}:00",
        "entity_id": entity_id,
        "service": service,
        "start_hour": start_hour,
    }


def _make_engine(tmp_path: Path) -> SuggestionEngine:
    return SuggestionEngine(
        dismissed_path=tmp_path / "dismissed.json",
        automations_path=tmp_path / "automations.json",
    )


# ---------------------------------------------------------------------------
# SuggestionEngine — per-user pending state
# ---------------------------------------------------------------------------


class TestPendingIsolation:
    def test_pending_suggestion_isolated_per_user(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.get_suggestion([_make_pattern()], "alice") is not None
        assert engine.has_pending("alice")
        assert not engine.has_pending("bob")

    def test_user_b_yes_does_not_consume_user_a_pending(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")

        reply = engine.handle_yes("bob")

        assert "no pending" in reply.lower()
        # Alice's pending suggestion is untouched and still accepted by her
        assert engine.has_pending("alice")
        assert not (tmp_path / "automations.json").exists()
        assert "set that up" in engine.handle_yes("alice").lower()

    def test_user_b_dismiss_does_not_consume_user_a_pending(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")

        reply = engine.handle_dismiss("bob")

        assert "no pending" in reply.lower()
        assert engine.has_pending("alice")
        assert not (tmp_path / "dismissed.json").exists()

    def test_session_flag_is_per_user(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patterns = [_make_pattern()]
        assert engine.get_suggestion(patterns, "alice") is not None
        # Alice already got her one suggestion this session; Bob still gets his
        assert engine.get_suggestion(patterns, "alice") is None
        assert engine.get_suggestion(patterns, "bob") is not None

    def test_pending_spoken_text_scoped_to_owner(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        assert engine.pending_spoken_text("alice")
        assert engine.pending_spoken_text("bob") is None


# ---------------------------------------------------------------------------
# SuggestionEngine — per-user dismissals
# ---------------------------------------------------------------------------


class TestDismissalIsolation:
    def test_dismissal_scoped_to_dismissing_user(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patterns = [_make_pattern()]
        engine.get_suggestion(patterns, "alice")
        engine.handle_dismiss("alice")

        # Alice's dismissal must not suppress Bob's identical pattern
        assert engine.get_suggestion(patterns, "bob") is not None
        # ... while Alice stays suppressed in a fresh session
        engine.reset_session("alice")
        assert engine.get_suggestion(patterns, "alice") is None

    def test_is_dismissed_per_user(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        engine.handle_dismiss("alice")

        key = "light.kitchen_ceiling:turn_on"
        assert engine.is_dismissed(key, "alice")
        assert not engine.is_dismissed(key, "bob")


# ---------------------------------------------------------------------------
# Fail closed on missing / invalid identity
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_no_suggestion_without_user(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.get_suggestion([_make_pattern()], None) is None
        assert engine.get_suggestion([_make_pattern()], "") is None

    def test_no_suggestion_for_invalid_user(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.get_suggestion([_make_pattern()], "../evil") is None
        assert engine.get_suggestion([_make_pattern()], "..") is None

    def test_invalid_user_cannot_consume_pending(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")

        assert "no pending" in engine.handle_yes(None).lower()
        assert "no pending" in engine.handle_yes("../evil").lower()
        assert "no pending" in engine.handle_dismiss(None).lower()
        assert engine.has_pending("alice")

    def test_has_pending_false_for_invalid_user(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        assert not engine.has_pending(None)
        assert not engine.has_pending("../evil")


# ---------------------------------------------------------------------------
# Persistence across engine restart
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_dismissal_persists_per_user_across_restart(self, tmp_path: Path) -> None:
        patterns = [_make_pattern()]
        engine1 = _make_engine(tmp_path)
        engine1.get_suggestion(patterns, "alice")
        engine1.handle_dismiss("alice")

        # Fresh engine over the same files (process restart)
        engine2 = _make_engine(tmp_path)
        assert engine2.get_suggestion(patterns, "alice") is None
        assert engine2.get_suggestion(patterns, "bob") is not None

    def test_automations_persist_with_owner_across_restart(self, tmp_path: Path) -> None:
        engine1 = _make_engine(tmp_path)
        engine1.get_suggestion([_make_pattern()], "alice")
        engine1.handle_yes("alice")

        engine2 = _make_engine(tmp_path)
        engine2.get_suggestion([_make_pattern(entity_id="light.bedroom")], "bob")
        engine2.handle_yes("bob")

        saved = json.loads((tmp_path / "automations.json").read_text(encoding="utf-8"))
        assert len(saved) == 2
        owners = {entry["key"]: entry["user_id"] for entry in saved}
        assert owners["light.kitchen_ceiling:turn_on"] == "alice"
        assert owners["light.bedroom:turn_on"] == "bob"

    def test_legacy_flat_dismissed_file_not_shared(self, tmp_path: Path) -> None:
        """A pre-per-user flat dismissed file belongs to "default", not everyone."""
        pattern = _make_pattern()
        dismissed_path = tmp_path / "dismissed.json"
        dismissed_path.write_text(
            json.dumps({"light.kitchen_ceiling:turn_on": time.time()}),
            encoding="utf-8",
        )

        engine = _make_engine(tmp_path)
        assert engine.get_suggestion([pattern], "default") is None
        assert engine.get_suggestion([pattern], "alice") is not None

    def test_new_dismissal_upgrades_legacy_file_without_losing_entries(
        self, tmp_path: Path
    ) -> None:
        legacy_key = "light.hallway:turn_off"
        dismissed_path = tmp_path / "dismissed.json"
        dismissed_path.write_text(json.dumps({legacy_key: time.time()}), encoding="utf-8")

        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        engine.handle_dismiss("alice")

        data = json.loads(dismissed_path.read_text(encoding="utf-8"))
        assert legacy_key in data["users"]["default"]
        assert "light.kitchen_ceiling:turn_on" in data["users"]["alice"]


# ---------------------------------------------------------------------------
# IntentRouter — yes/no intercept scoped to the owner
# ---------------------------------------------------------------------------


class TestRouterIntercept:
    def test_owner_yes_accepts(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        router = IntentRouter()

        result = router.route("yes", suggestion_engine=engine, user_id="alice")

        assert result.handled
        assert result.intent_type == "suggestion_accept"
        assert not engine.has_pending("alice")

    def test_other_user_yes_not_intercepted(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        router = IntentRouter()

        result = router.route("yes", suggestion_engine=engine, user_id="bob")

        # Bob's "yes" must not be treated as an answer to Alice's suggestion
        assert result.intent_type != "suggestion_accept"
        assert engine.has_pending("alice")

    def test_other_user_no_not_intercepted(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        router = IntentRouter()

        result = router.route("no thanks", suggestion_engine=engine, user_id="bob")

        assert result.intent_type != "suggestion_dismiss"
        assert engine.has_pending("alice")
        assert not (tmp_path / "dismissed.json").exists()

    def test_missing_user_id_skips_intercept(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        router = IntentRouter()

        result = router.route("yes", suggestion_engine=engine)

        assert result.intent_type != "suggestion_accept"
        assert engine.has_pending("alice")


# ---------------------------------------------------------------------------
# ResponseBuilder — pending suggestion surfaced only to its owner
# ---------------------------------------------------------------------------


class TestResponseBuilderScoping:
    def _build(self, engine: SuggestionEngine, user_id: str | None):
        rb = ResponseBuilder(suggestion_engine=engine)
        action_result = MagicMock()
        action_result.response = "Done."
        return rb.build(action_result, None, transcript="hi", user_id=user_id)

    def test_owner_sees_pending_suggestion(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        result = self._build(engine, "alice")
        assert result.suggestions
        assert "Want me to automate that?" in result.suggestions[0]

    def test_other_user_sees_no_suggestion(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        assert self._build(engine, "bob").suggestions == []

    def test_no_user_sees_no_suggestion(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        assert self._build(engine, None).suggestions == []


# ---------------------------------------------------------------------------
# ActionDispatcher — pattern entries recorded per user
# ---------------------------------------------------------------------------


class _FakeCommandHistory:
    """Minimal stand-in for HABridge._command_history."""

    def __init__(self) -> None:
        self._entries: list[Any] = []

    def __len__(self) -> int:
        return len(self._entries)


def _make_ha_bridge(history: _FakeCommandHistory) -> MagicMock:
    ha = MagicMock()
    ha.enabled = True
    ha._command_history = history

    def _process(_transcript: str) -> str:
        entry = MagicMock()
        entry.entity_id = "light.kitchen_ceiling"
        entry.service = "turn_on"
        history._entries.append(entry)
        return "Done."

    ha.process_transcript.side_effect = _process
    return ha


class TestDispatcherPatternEntries:
    def test_pattern_entries_recorded_under_requesting_user(self) -> None:
        history = _FakeCommandHistory()
        engine = MagicMock()
        engine.get_suggestion.return_value = None
        pattern_entries: dict[str, list] = {}
        dispatcher = ActionDispatcher(
            context_builder=MagicMock(),
            llm=MagicMock(),
            result_handler=MagicMock(),
            ha_bridge=_make_ha_bridge(history),
            suggestion_engine=engine,
            pattern_entries=pattern_entries,
        )

        asyncio.run(
            dispatcher.dispatch(None, None, "turn on the kitchen light", active_user_id="alice")
        )
        asyncio.run(
            dispatcher.dispatch(None, None, "turn on the kitchen light", active_user_id="bob")
        )
        asyncio.run(
            dispatcher.dispatch(None, None, "turn on the kitchen light", active_user_id="alice")
        )

        assert len(pattern_entries["alice"]) == 2
        assert len(pattern_entries["bob"]) == 1

    def test_get_suggestion_called_with_requesting_user(self) -> None:
        history = _FakeCommandHistory()
        engine = MagicMock()
        engine.get_suggestion.return_value = None
        dispatcher = ActionDispatcher(
            context_builder=MagicMock(),
            llm=MagicMock(),
            result_handler=MagicMock(),
            ha_bridge=_make_ha_bridge(history),
            suggestion_engine=engine,
            pattern_entries={},
        )

        asyncio.run(
            dispatcher.dispatch(None, None, "turn on the kitchen light", active_user_id="alice")
        )

        assert engine.get_suggestion.call_count == 1
        _, kwargs = engine.get_suggestion.call_args
        assert kwargs["user_id"] == "alice"

    def test_suggestion_derived_only_from_own_entries(self, tmp_path: Path) -> None:
        """Three of Alice's commands never produce a suggestion for Bob."""
        history = _FakeCommandHistory()
        engine = _make_engine(tmp_path)
        dispatcher = ActionDispatcher(
            context_builder=MagicMock(),
            llm=MagicMock(),
            result_handler=MagicMock(),
            ha_bridge=_make_ha_bridge(history),
            suggestion_engine=engine,
            pattern_entries={},
        )

        # Alice repeats the same command three times (min_occurrences)
        for _ in range(3):
            asyncio.run(
                dispatcher.dispatch(None, None, "turn on the kitchen light", active_user_id="alice")
            )
        # Alice's own repetition triggered a pending suggestion for Alice
        assert engine.has_pending("alice")

        # Bob issues one command; his single entry can't reach the threshold,
        # so no suggestion may leak to him from Alice's history
        asyncio.run(
            dispatcher.dispatch(None, None, "turn on the kitchen light", active_user_id="bob")
        )
        assert not engine.has_pending("bob")
