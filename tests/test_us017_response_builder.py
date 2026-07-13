"""Tests for ResponseBuilder (US-017).

Verifies that response post-processing logic extracted from assistant.py
produces correct output through the new ResponseBuilder class.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from rex.actions.dispatcher import ActionResult
from rex.context.builder import ContextPackage
from rex.response.builder import FinalResponse, ResponseBuilder, _clean_for_tts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ContextPackage:
    return ContextPackage(
        messages=[{"role": "user", "content": "test"}],
        system_prompt="system",
        session_id="default",
        user_facts={},
        prompt="test\nassistant:",
    )


def _make_action_result(response: str = "Hello, world!") -> ActionResult:
    return ActionResult(success=True, response=response, actions_taken=["llm"])


def _make_builder(**kwargs) -> ResponseBuilder:
    defaults: dict = {}
    defaults.update(kwargs)
    return ResponseBuilder(**defaults)


def _make_cache(cached: str | None = None) -> MagicMock:
    cache = MagicMock()
    cache.get.return_value = cached
    return cache


# ---------------------------------------------------------------------------
# FinalResponse dataclass
# ---------------------------------------------------------------------------


class TestFinalResponse:
    def test_required_fields(self):
        r = FinalResponse(text="hello", tts_text="hello")
        assert r.text == "hello"
        assert r.tts_text == "hello"
        assert r.suggestions == []
        assert r.followups == []
        assert r.cache_hit is False

    def test_all_fields(self):
        r = FinalResponse(
            text="hi",
            tts_text="hi",
            suggestions=["s1"],
            followups=["f1"],
            cache_hit=True,
        )
        assert r.suggestions == ["s1"]
        assert r.followups == ["f1"]
        assert r.cache_hit is True


# ---------------------------------------------------------------------------
# TTS cleaning
# ---------------------------------------------------------------------------


class TestCleanForTts:
    def test_strips_bold(self):
        assert _clean_for_tts("**bold** text") == "bold text"

    def test_strips_italic(self):
        assert _clean_for_tts("*italic* text") == "italic text"

    def test_strips_bold_italic(self):
        assert _clean_for_tts("***both*** text") == "both text"

    def test_strips_inline_code(self):
        assert _clean_for_tts("`code`") == "code"

    def test_strips_markdown_header(self):
        assert _clean_for_tts("## Header\nBody") == "Header\nBody"

    def test_strips_link(self):
        assert _clean_for_tts("[click here](https://example.com)") == "click here"

    def test_plain_text_unchanged(self):
        assert _clean_for_tts("Hello, world!") == "Hello, world!"

    def test_strips_leading_trailing_whitespace(self):
        assert _clean_for_tts("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# ResponseBuilder.check_cache
# ---------------------------------------------------------------------------


class TestCheckCache:
    def test_no_cache_returns_none(self):
        rb = _make_builder()
        assert rb.check_cache("anything") is None

    def test_cache_hit_returns_response(self):
        cache = _make_cache(cached="cached reply")
        rb = _make_builder(response_cache=cache)
        assert rb.check_cache("hello") == "cached reply"
        cache.get.assert_called_once_with("hello")

    def test_cache_miss_returns_none(self):
        cache = _make_cache(cached=None)
        rb = _make_builder(response_cache=cache)
        assert rb.check_cache("hello") is None

    def test_delegates_to_cache_get(self):
        cache = MagicMock()
        cache.get.return_value = "result"
        rb = _make_builder(response_cache=cache)
        result = rb.check_cache("my transcript")
        cache.get.assert_called_once_with("my transcript")
        assert result == "result"


# ---------------------------------------------------------------------------
# ResponseBuilder.build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_returns_final_response(self):
        rb = _make_builder()
        result = rb.build(_make_action_result("hello"), _make_context())
        assert isinstance(result, FinalResponse)

    def test_text_matches_action_result(self):
        rb = _make_builder()
        result = rb.build(_make_action_result("My response"), _make_context())
        assert result.text == "My response"

    def test_tts_text_strips_markdown(self):
        rb = _make_builder()
        result = rb.build(
            _make_action_result("**Bold** and *italic*"),
            _make_context(),
        )
        assert result.tts_text == "Bold and italic"

    def test_tts_text_plain_unchanged(self):
        rb = _make_builder()
        result = rb.build(_make_action_result("Plain text"), _make_context())
        assert result.tts_text == "Plain text"

    def test_cache_hit_is_false(self):
        rb = _make_builder()
        result = rb.build(_make_action_result("hi"), _make_context())
        assert result.cache_hit is False

    def test_writes_to_cache_when_transcript_provided(self):
        cache = _make_cache()
        rb = _make_builder(response_cache=cache)
        rb.build(_make_action_result("response"), _make_context(), transcript="hello")
        cache.put.assert_called_once_with("hello", "response")

    def test_no_cache_write_without_transcript(self):
        cache = _make_cache()
        rb = _make_builder(response_cache=cache)
        rb.build(_make_action_result("response"), _make_context())
        cache.put.assert_not_called()

    def test_no_cache_write_when_no_cache(self):
        rb = _make_builder()
        # Should not raise even without a cache
        result = rb.build(_make_action_result("response"), _make_context(), transcript="hi")
        assert result.text == "response"

    def test_suggestions_empty_no_engine(self):
        rb = _make_builder()
        result = rb.build(_make_action_result("hi"), _make_context())
        assert result.suggestions == []

    def test_suggestions_from_pending_engine(self):
        engine = MagicMock()
        engine.pending_spoken_text.return_value = "Want me to automate that?"
        rb = _make_builder(suggestion_engine=engine)
        result = rb.build(_make_action_result("ok"), _make_context(), user_id="alice")
        assert result.suggestions == ["Want me to automate that?"]
        engine.pending_spoken_text.assert_called_once_with("alice")

    def test_suggestions_empty_when_no_pending(self):
        engine = MagicMock()
        engine.pending_spoken_text.return_value = None
        rb = _make_builder(suggestion_engine=engine)
        result = rb.build(_make_action_result("ok"), _make_context(), user_id="alice")
        assert result.suggestions == []

    def test_suggestions_empty_without_user_id(self):
        # Fail closed: no user identity means no suggestion is surfaced (#303)
        engine = MagicMock()
        engine.pending_spoken_text.return_value = "Want me to automate that?"
        rb = _make_builder(suggestion_engine=engine)
        result = rb.build(_make_action_result("ok"), _make_context())
        assert result.suggestions == []
        engine.pending_spoken_text.assert_not_called()

    def test_followups_empty_no_engine(self):
        rb = _make_builder()
        result = rb.build(_make_action_result("hi"), _make_context())
        assert result.followups == []

    def test_followups_from_engine(self):
        engine = MagicMock()
        engine.format_followups.return_value = "Did you mean X?"
        rb = _make_builder(followup_engine=engine)
        result = rb.build(_make_action_result("ok"), _make_context(), user_id="alice")
        assert result.followups == ["Did you mean X?"]
        engine.format_followups.assert_called_once_with("alice")

    def test_followups_empty_without_user_id(self):
        # Fail closed: no user identity means no cue state is read (#303)
        engine = MagicMock()
        engine.format_followups.return_value = "Did you mean X?"
        rb = _make_builder(followup_engine=engine)
        result = rb.build(_make_action_result("ok"), _make_context())
        assert result.followups == []
        engine.format_followups.assert_not_called()

    def test_followups_empty_when_engine_returns_none(self):
        engine = MagicMock()
        engine.format_followups.return_value = None
        rb = _make_builder(followup_engine=engine)
        result = rb.build(_make_action_result("ok"), _make_context())
        assert result.followups == []

    def test_followups_empty_no_format_followups_attr(self):
        engine = object()  # no format_followups method
        rb = _make_builder(followup_engine=engine)
        result = rb.build(_make_action_result("ok"), _make_context())
        assert result.followups == []


# ---------------------------------------------------------------------------
# Integration: Assistant uses ResponseBuilder
# ---------------------------------------------------------------------------


class TestAssistantUsesResponseBuilder:
    def _make_assistant(self):
        from rex.assistant import Assistant

        a = Assistant.__new__(Assistant)
        a._settings = MagicMock()
        a._settings.max_memory_items = 50
        a._settings.persist_history = False
        a._settings.followups_enabled = False
        a._settings.model_routing = None
        a._user_id = "default"
        a._history = []
        a._history_limit = 50
        a._plugins = []
        a._history_store = None
        a._followup_engine = None
        a._pending_followup = None
        a._followup_lock = asyncio.Lock()
        a._router = None
        a._response_cache = None
        a._ha_bridge = None
        a._suggestion_engine = None
        a._pattern_entries = {}
        return a

    def test_get_or_create_response_builder_returns_instance(self):
        from rex.response.builder import ResponseBuilder

        a = self._make_assistant()
        rb = a._get_or_create_response_builder()
        assert isinstance(rb, ResponseBuilder)

    def test_get_or_create_response_builder_lazy(self):
        a = self._make_assistant()
        rb1 = a._get_or_create_response_builder()
        rb2 = a._get_or_create_response_builder()
        assert rb1 is rb2

    def test_generate_reply_uses_response_builder(self):
        """generate_reply() should return a string (via ResponseBuilder.build)."""
        a = self._make_assistant()

        llm = MagicMock()
        llm.generate.return_value = "LLM reply"
        a._llm = llm

        from rex.response.builder import ResponseBuilder

        rb = MagicMock(spec=ResponseBuilder)
        rb.check_cache.return_value = None
        from rex.response.builder import FinalResponse as FR

        rb.build.return_value = FR(text="LLM reply", tts_text="LLM reply")
        a._response_builder = rb

        # Patch action dispatcher to return a canned result
        from rex.actions.dispatcher import ActionResult

        async def _fake_dispatch(*args, **kwargs):
            return ActionResult(success=True, response="LLM reply")

        ad = MagicMock()
        ad.dispatch = _fake_dispatch
        a._action_dispatcher = ad

        # Patch other lazy-created objects
        from rex.context.builder import ContextPackage

        cb = MagicMock()
        cb.build.return_value = ContextPackage(
            messages=[], system_prompt="s", session_id="default", user_facts={}, prompt=""
        )
        a._context_builder = cb

        from rex.intent.router import IntentResult

        ir = MagicMock()
        ir.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
        a._intent_router = ir

        with (
            patch("rex.capabilities.responder.is_capability_query", return_value=False),
            patch("rex.capabilities.registry.get_capability_registry"),
            patch("rex.capabilities.responder.build_capability_response"),
        ):
            reply = asyncio.run(a.generate_reply("hello"))

        assert reply == "LLM reply"
        # Cache lookups are confined to the active user's partition (#303)
        rb.check_cache.assert_called_once_with("hello", user_id="default")
        rb.build.assert_called_once()

    def test_generate_reply_returns_cached_without_dispatch(self):
        """When cache hits, ActionDispatcher is never called."""
        a = self._make_assistant()
        a._llm = MagicMock()
        a._llm.model_name = "default-model"

        from rex.response.builder import ResponseBuilder

        rb = MagicMock(spec=ResponseBuilder)
        rb.check_cache.return_value = "cached reply"
        a._response_builder = rb

        ad = MagicMock()
        a._action_dispatcher = ad

        from rex.intent.router import IntentResult

        ir = MagicMock()
        ir.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
        a._intent_router = ir

        with (
            patch("rex.capabilities.responder.is_capability_query", return_value=False),
            patch("rex.capabilities.registry.get_capability_registry"),
            patch("rex.capabilities.responder.build_capability_response"),
        ):
            reply = asyncio.run(a.generate_reply("hello"))

        assert reply == "cached reply"
        ad.dispatch.assert_not_called()
