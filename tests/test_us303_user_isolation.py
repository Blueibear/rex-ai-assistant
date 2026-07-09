"""Negative tests for per-user isolation of the response cache and the
in-memory conversation history (issue #303, gaps A and B).

Proves:

- A response cached for user A is never returned for user B (same text).
- The in-memory history window is partitioned per user: user B's LLM
  prompt never contains user A's turns.
- Turn recording (in-memory and persisted) is attributed to the identified
  speaker on every ``generate_reply`` path, including the cache-hit and
  post-``_end_request`` paths.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from rex.response_cache import ResponseCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant(user_id: str = "default", history_store=None):
    """Build a minimal Assistant without running the heavy __init__."""
    from rex.assistant import Assistant

    a = Assistant.__new__(Assistant)
    a._settings = MagicMock()
    a._settings.max_memory_items = 50
    a._settings.persist_history = False
    a._settings.followups_enabled = False
    a._settings.model_routing = None
    a._settings.transcripts_enabled = False
    a._user_id = user_id
    a._histories = {}
    a._history = []
    a._history_limit = 50
    a._plugins = []
    a._history_store = history_store
    a._followup_engine = None
    a._pending_followup = None
    a._followup_lock = asyncio.Lock()
    a._router = None
    a._response_cache = None
    a._ha_bridge = None
    a._suggestion_engine = None
    a._pattern_entries = {}
    return a


# ---------------------------------------------------------------------------
# ResponseCache — per-user partitioning (gap A)
# ---------------------------------------------------------------------------


class TestResponseCacheUserPartition:
    def test_user_b_cannot_read_user_a_entry(self):
        cache = ResponseCache(ttl=60.0)
        cache.put("what is my favorite color?", "Your favorite color is blue.", user_id="alice")
        assert cache.get("what is my favorite color?", user_id="bob") is None

    def test_same_user_still_hits(self):
        cache = ResponseCache(ttl=60.0)
        cache.put("what is my favorite color?", "Your favorite color is blue.", user_id="alice")
        assert cache.get("what is my favorite color?", user_id="alice") == (
            "Your favorite color is blue."
        )

    def test_anonymous_partition_isolated_from_named_users(self):
        cache = ResponseCache(ttl=60.0)
        cache.put("what is the capital of France?", "Paris.")
        assert cache.get("what is the capital of France?", user_id="alice") is None
        assert cache.get("what is the capital of France?") == "Paris."

    def test_named_partition_isolated_from_anonymous(self):
        cache = ResponseCache(ttl=60.0)
        cache.put("what is the capital of France?", "Paris.", user_id="alice")
        assert cache.get("what is the capital of France?") is None

    def test_two_users_can_hold_different_answers(self):
        cache = ResponseCache(ttl=60.0)
        cache.put("what is my dog's name?", "Rex.", user_id="alice")
        cache.put("what is my dog's name?", "Fido.", user_id="bob")
        assert cache.get("what is my dog's name?", user_id="alice") == "Rex."
        assert cache.get("what is my dog's name?", user_id="bob") == "Fido."


# ---------------------------------------------------------------------------
# ResponseBuilder — user id threads through to the cache
# ---------------------------------------------------------------------------


class TestResponseBuilderCacheUserThreading:
    def test_check_cache_passes_user_id(self):
        from rex.response.builder import ResponseBuilder

        cache = MagicMock()
        cache.get.return_value = None
        rb = ResponseBuilder(response_cache=cache)
        rb.check_cache("hello", user_id="alice")
        cache.get.assert_called_once_with("hello", user_id="alice")

    def test_check_cache_without_user_keeps_legacy_call(self):
        from rex.response.builder import ResponseBuilder

        cache = MagicMock()
        cache.get.return_value = None
        rb = ResponseBuilder(response_cache=cache)
        rb.check_cache("hello")
        cache.get.assert_called_once_with("hello")

    def test_build_puts_under_user_partition(self):
        from rex.actions.dispatcher import ActionResult
        from rex.context.builder import ContextPackage
        from rex.response.builder import ResponseBuilder

        cache = MagicMock()
        rb = ResponseBuilder(response_cache=cache)
        ctx = ContextPackage(
            messages=[], system_prompt="s", session_id="alice", user_facts={}, prompt=""
        )
        rb.build(
            ActionResult(success=True, response="answer"),
            ctx,
            transcript="question",
            user_id="alice",
        )
        cache.put.assert_called_once_with("question", "answer", user_id="alice")


# ---------------------------------------------------------------------------
# Assistant — per-user in-memory history (gap B)
# ---------------------------------------------------------------------------


class TestAssistantHistoryPartition:
    def test_record_completion_isolates_users(self):
        a = _make_assistant(user_id="alice")
        a._record_completion("my pin is 1234", "Noted.", user_id="alice")
        a._record_completion("what's the weather like", "Rainy.", user_id="bob")

        alice_history = a._history_for("alice")
        bob_history = a._history_for("bob")
        assert any("1234" in turn.text for turn in alice_history)
        assert not any("1234" in turn.text for turn in bob_history)

    def test_history_property_follows_active_user(self):
        a = _make_assistant(user_id="alice")
        a._record_completion("alice secret", "ok", user_id="alice")

        a._user_id = "bob"
        assert not any("alice secret" in turn.text for turn in a._history)

        a._user_id = "alice"
        assert any("alice secret" in turn.text for turn in a._history)

    def test_context_builder_prompt_excludes_other_users_turns(self):
        a = _make_assistant(user_id="alice")
        a._record_completion("my ssn is 000-00-0000", "Stored.", user_id="alice")

        cb = a._get_or_create_context_builder()

        # Simulate the _begin_request user swap for an identified speaker.
        a._user_id = "bob"
        messages = cb._build_messages("hello", system_prompt="sys")
        joined = " ".join(str(m.get("content", "")) for m in messages)
        assert "000-00-0000" not in joined

        # Alice's own context still contains her turns.
        a._user_id = "alice"
        messages = cb._build_messages("hello again", system_prompt="sys")
        joined = " ".join(str(m.get("content", "")) for m in messages)
        assert "000-00-0000" in joined

    def test_context_builder_sees_new_turns_after_trim_reassignment(self):
        """The cached builder must read live history, not an init-time snapshot."""
        a = _make_assistant(user_id="alice")
        cb = a._get_or_create_context_builder()

        # _record_completion reassigns self._history after trimming; the
        # builder previously kept a stale reference to the original list.
        a._record_completion("remember the cake", "Will do.", user_id="alice")
        messages = cb._build_messages("hello", system_prompt="sys")
        joined = " ".join(str(m.get("content", "")) for m in messages)
        assert "remember the cake" in joined


# ---------------------------------------------------------------------------
# Assistant.generate_reply — attribution and cache partition end to end
# ---------------------------------------------------------------------------


class TestGenerateReplyAttribution:
    def _wire_llm_path(self, a, reply_text: str):
        """Attach fake intent router / dispatcher so generate_reply reaches
        the response-builder path and returns *reply_text*."""
        from rex.actions.dispatcher import ActionResult
        from rex.intent.router import IntentResult

        ir = MagicMock()
        ir.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
        a._intent_router = ir

        async def _fake_dispatch(*args, **kwargs):
            return ActionResult(success=True, response=reply_text)

        ad = MagicMock()
        ad.dispatch = _fake_dispatch
        a._action_dispatcher = ad
        a._llm = MagicMock()
        a._llm.model_name = "m"

    def test_cached_answer_not_shared_across_active_users(self):
        a = _make_assistant(user_id="default")
        a._response_cache = ResponseCache(ttl=60.0)
        self._wire_llm_path(a, "Blue, your favorite.")

        first = asyncio.run(a.generate_reply("what is my favorite color", active_user_id="alice"))
        assert first == "Blue, your favorite."

        # Same question from bob must not hit alice's cached entry: the
        # dispatcher runs again and can produce a bob-specific answer.
        self._wire_llm_path(a, "Green, your favorite.")
        second = asyncio.run(a.generate_reply("what is my favorite color", active_user_id="bob"))
        assert second == "Green, your favorite."

        # And alice still gets her own cached answer without dispatch.
        a._action_dispatcher = MagicMock()  # would explode if dispatched
        third = asyncio.run(a.generate_reply("what is my favorite color", active_user_id="alice"))
        assert third == "Blue, your favorite."

    def test_persisted_turns_attributed_to_active_user(self):
        store = MagicMock()
        store.load_history.return_value = []
        a = _make_assistant(user_id="default", history_store=store)
        self._wire_llm_path(a, "Done.")

        asyncio.run(a.generate_reply("turn it up", active_user_id="bob"))

        saved_users = [call.args[0] for call in store.save_turn.call_args_list]
        assert saved_users, "expected save_turn to be called"
        assert set(saved_users) == {"bob"}

    def test_generate_reply_prompt_excludes_other_users_history(self):
        from rex.actions.dispatcher import ActionResult
        from rex.intent.router import IntentResult

        a = _make_assistant(user_id="default")
        a._record_completion("my password is hunter2", "Stored.", user_id="alice")

        ir = MagicMock()
        ir.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
        a._intent_router = ir
        a._llm = MagicMock()
        a._llm.model_name = "m"

        captured: dict = {}

        async def _capture_dispatch(intent, ctx, transcript, **kwargs):
            captured["messages"] = ctx.messages
            return ActionResult(success=True, response="ok")

        ad = MagicMock()
        ad.dispatch = _capture_dispatch
        a._action_dispatcher = ad

        asyncio.run(a.generate_reply("hello", active_user_id="bob"))

        joined = " ".join(str(m.get("content", "")) for m in captured["messages"])
        assert "hunter2" not in joined


# ---------------------------------------------------------------------------
# Assistant.generate_reply — pending suggestion isolation end to end
# ---------------------------------------------------------------------------


class TestGenerateReplySuggestionIsolation:
    """Suggestion accept/dismiss through the full generate_reply pipeline is
    scoped to the pending suggestion's owner (issue #303)."""

    @staticmethod
    def _pattern() -> dict:
        return {
            "pattern": "turn_on light.kitchen_ceiling around 07:00",
            "frequency": 5,
            "suggested_automation": "Automate: turn_on light.kitchen_ceiling daily at 07:00",
            "entity_id": "light.kitchen_ceiling",
            "service": "turn_on",
            "start_hour": 7,
        }

    def _make_assistant_with_pending(self, tmp_path, owner: str):
        """Assistant with a real IntentRouter and a real SuggestionEngine
        holding a pending suggestion for *owner*."""
        from rex.actions.dispatcher import ActionResult
        from rex.intent.router import IntentRouter
        from rex.suggestions.engine import SuggestionEngine

        a = _make_assistant(user_id="default")
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        assert engine.get_suggestion([self._pattern()], owner) is not None
        a._suggestion_engine = engine
        a._intent_router = IntentRouter()

        async def _fake_dispatch(*args, **kwargs):
            return ActionResult(success=True, response="LLM answer")

        ad = MagicMock()
        ad.dispatch = _fake_dispatch
        a._action_dispatcher = ad
        a._llm = MagicMock()
        a._llm.model_name = "m"
        return a, engine

    def test_other_users_yes_does_not_accept_pending(self, tmp_path):
        a, engine = self._make_assistant_with_pending(tmp_path, owner="alice")

        reply = asyncio.run(a.generate_reply("yes", active_user_id="bob"))

        # Bob's "yes" falls through to normal dispatch instead of accepting
        # Alice's suggestion, which remains pending and unsaved.
        assert "set that up" not in reply.lower()
        assert engine.has_pending("alice")
        assert not (tmp_path / "automations.json").exists()

    def test_other_users_no_does_not_dismiss_pending(self, tmp_path):
        a, engine = self._make_assistant_with_pending(tmp_path, owner="alice")

        asyncio.run(a.generate_reply("no thanks", active_user_id="bob"))

        assert engine.has_pending("alice")
        assert not (tmp_path / "dismissed.json").exists()

    def test_owner_yes_accepts_pending(self, tmp_path):
        import json

        a, engine = self._make_assistant_with_pending(tmp_path, owner="alice")

        reply = asyncio.run(a.generate_reply("yes", active_user_id="alice"))

        assert "set that up" in reply.lower()
        assert not engine.has_pending("alice")
        saved = json.loads((tmp_path / "automations.json").read_text(encoding="utf-8"))
        assert saved[0]["user_id"] == "alice"
