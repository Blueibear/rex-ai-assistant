"""Tests for ContextBuilder (US-014).

Verifies that context assembly logic extracted from assistant.py
produces correct ContextPackage output via ContextBuilder.build().
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(*, history=None, personality=None, user_id="default"):
    from rex.context.builder import ContextBuilder

    mock_settings = MagicMock()
    mock_settings.default_timezone = None
    mock_settings.default_location = None
    mock_settings.personality = personality
    return ContextBuilder(
        settings=mock_settings,
        history=history or [],
        user_id=user_id,
        followup_engine=None,
    )


# ---------------------------------------------------------------------------
# Import / structure tests
# ---------------------------------------------------------------------------


def test_import():
    from rex.context.builder import ContextBuilder, ContextPackage  # noqa: F401


def test_context_package_fields():
    from rex.context.builder import ContextPackage

    pkg = ContextPackage(
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="sys",
        session_id="alice",
        user_facts={"dog": "Max"},
        prompt="sys\nuser: hi\nassistant:",
    )
    assert pkg.session_id == "alice"
    assert pkg.user_facts == {"dog": "Max"}
    assert pkg.messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# build_system_context
# ---------------------------------------------------------------------------


def test_build_system_context_contains_date():
    import re

    builder = _make_builder()
    ctx = builder.build_system_context()
    assert "Current date and time:" in ctx
    assert re.search(r"\d{4}-\d{2}-\d{2}", ctx)


def test_build_system_context_contains_tool_instructions():
    builder = _make_builder()
    ctx = builder.build_system_context()
    assert "TOOL_REQUEST" in ctx
    assert "time_now" in ctx
    assert "weather_now" in ctx
    assert "web_search" in ctx


def test_build_system_context_includes_location_when_configured():
    from rex.context.builder import ContextBuilder

    mock_settings = MagicMock()
    mock_settings.default_timezone = "America/Chicago"
    mock_settings.default_location = "Dallas, TX"
    mock_settings.personality = None
    builder = ContextBuilder(settings=mock_settings, history=[], user_id="default")
    ctx = builder.build_system_context()
    assert "Dallas, TX" in ctx
    assert "America/Chicago" in ctx


# ---------------------------------------------------------------------------
# build() → ContextPackage
# ---------------------------------------------------------------------------


def test_build_returns_context_package():
    from rex.context.builder import ContextPackage

    builder = _make_builder()
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("hello")
    assert isinstance(pkg, ContextPackage)


def test_build_sets_session_id():
    builder = _make_builder(user_id="default")
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("hello")
    assert pkg.session_id == "default"


def test_build_active_user_id_sets_session_id():
    builder = _make_builder(user_id="default")
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("hello", active_user_id="alice")
    assert pkg.session_id == "alice"


def test_build_messages_includes_user_message():
    builder = _make_builder()
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("What time is it?")
    last_msg = pkg.messages[-1]
    assert last_msg["role"] == "user"
    assert last_msg["content"] == "What time is it?"


def test_build_prompt_contains_user_message():
    builder = _make_builder()
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("Hello world")
    assert "Hello world" in pkg.prompt


def test_build_prompt_ends_with_assistant():
    builder = _make_builder()
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("hello")
    assert pkg.prompt.endswith("assistant:")


# ---------------------------------------------------------------------------
# voice_mode
# ---------------------------------------------------------------------------


def test_voice_mode_adds_concise_instruction_to_prompt():
    builder = _make_builder()
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("What time?", voice_mode=True)
    assert (
        "sentence" in pkg.prompt.lower()
        or "short" in pkg.prompt.lower()
        or "concise" in pkg.prompt.lower()
    )


def test_non_voice_mode_omits_concise_instruction():
    builder = _make_builder()
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("What time?", voice_mode=False)
    assert "sentence" not in pkg.prompt.lower() or "short" not in pkg.prompt.lower()


def test_voice_mode_prompt_longer_than_non_voice():
    builder = _make_builder()
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg_default = builder.build("Hello")
        pkg_voice = builder.build("Hello", voice_mode=True)
    assert len(pkg_voice.prompt) > len(pkg_default.prompt)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_turns_appear_in_messages():
    from rex.assistant import ConversationTurn

    history = [
        ConversationTurn("user", "hi"),
        ConversationTurn("assistant", "hello"),
    ]
    builder = _make_builder(history=history)
    with patch.object(type(builder), "build_system_context", return_value="[sys]"):
        pkg = builder.build("next question")
    roles = [m["role"] for m in pkg.messages]
    assert "user" in roles
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# assistant.py backward-compat delegates
# ---------------------------------------------------------------------------


def test_assistant_build_system_context_delegates():
    """Assistant._build_system_context() delegates to ContextBuilder."""
    from unittest.mock import patch

    import rex.assistant as mod

    class DummyLLM:
        def __init__(self, *a, **kw):
            pass

        def generate(self, *a, **kw):
            return "ok"

    with patch.object(mod, "LanguageModel", DummyLLM):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            asst = mod.Assistant(transcripts_dir=tmp)

    result = asst._build_system_context()
    assert "Current date and time:" in result


def test_assistant_build_prompt_delegates():
    """Assistant._build_prompt() returns same prompt as ContextBuilder.build()."""
    from unittest.mock import patch

    import rex.assistant as mod

    class DummyLLM:
        def __init__(self, *a, **kw):
            pass

        def generate(self, *a, **kw):
            return "ok"

    with patch.object(mod, "LanguageModel", DummyLLM):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            asst = mod.Assistant(transcripts_dir=tmp, user_id="default")

    prompt = asst._build_prompt("hello")
    assert "hello" in prompt
    assert "assistant:" in prompt


# ---------------------------------------------------------------------------
# US-105 identity-safe deterministic artifact caching
# ---------------------------------------------------------------------------


def _context_cache_request(user_id: str = "alice"):
    from rex.context.revisions import ContextCacheRequest
    from rex.runtime.turn import AuthorizationSnapshotRef, TurnScope

    return ContextCacheRequest(
        user_id=user_id,
        scope=TurnScope.USER,
        authorization=AuthorizationSnapshotRef(
            policy_ref="test-policy",
            permission_ref=f"test-permission:{user_id}",
        ),
        model_provider="local",
        model_name="test-model",
    )


def _context_cache_versions():
    from rex.context.cache import ContextCacheVersions

    return ContextCacheVersions(
        identity="identity",
        policy="policy",
        permission="permission",
        model="model",
        capability="capability",
        config="config",
        memory="memory",
        prompt_template="prompt",
    )


def test_repeated_safe_build_reuses_one_private_artifact_load() -> None:
    builder = _make_builder(user_id="default")
    request = _context_cache_request("alice")

    with (
        patch(
            "rex.context.builder.build_context_cache_versions",
            return_value=_context_cache_versions(),
        ),
        patch.object(
            builder, "_get_active_personality_prompt", return_value="[persona]"
        ) as personality,
        patch.object(builder, "_load_user_profile_context", return_value="[profile]") as profile,
        patch.object(builder, "_get_user_facts", return_value={"city": "Dallas"}) as facts,
        patch.object(type(builder), "build_system_context", return_value="[sys]"),
    ):
        first = builder.build("first", active_user_id="alice", cache_request=request)
        second = builder.build("second", active_user_id="alice", cache_request=request)

    assert personality.call_count == 1
    assert profile.call_count == 1
    assert facts.call_count == 1
    assert first.user_facts == {"city": "Dallas"}
    assert second.user_facts == {"city": "Dallas"}
    assert first.user_facts is not second.user_facts
    assert "[Remembered facts about alice: city=Dallas]" in first.prompt
    assert first.messages[-1]["content"] == "first"
    assert second.messages[-1]["content"] == "second"


def test_cached_hit_matches_uncached_output_with_dynamic_context() -> None:
    from rex.assistant import ConversationTurn

    history = [ConversationTurn("user", "old question")]
    cached_builder = _make_builder(history=history, user_id="default")
    uncached_builder = _make_builder(history=history, user_id="default")
    request = _context_cache_request("alice")

    def configure(builder):
        return (
            patch.object(builder, "_get_active_personality_prompt", return_value="[persona]"),
            patch.object(builder, "_load_user_profile_context", return_value="[profile]"),
            patch.object(builder, "_get_user_facts", return_value={"city": "Dallas"}),
        )

    with patch(
        "rex.context.builder.build_context_cache_versions", return_value=_context_cache_versions()
    ):
        with (
            configure(cached_builder)[0],
            configure(cached_builder)[1],
            configure(cached_builder)[2],
            patch.object(type(cached_builder), "build_system_context", return_value="[sys]"),
        ):
            cached_builder.build("prime", active_user_id="alice", cache_request=request)

        history.append(ConversationTurn("assistant", "old answer"))
        with (
            configure(cached_builder)[0],
            configure(cached_builder)[1],
            configure(cached_builder)[2],
            patch.object(type(cached_builder), "build_system_context", return_value="[sys]"),
        ):
            cached = cached_builder.build(
                "fresh question",
                voice_mode=True,
                active_user_id="alice",
                tool_context="[tool]",
                cache_request=request,
            )
        with (
            configure(uncached_builder)[0],
            configure(uncached_builder)[1],
            configure(uncached_builder)[2],
            patch.object(type(uncached_builder), "build_system_context", return_value="[sys]"),
        ):
            uncached = uncached_builder.build(
                "fresh question",
                voice_mode=True,
                active_user_id="alice",
                tool_context="[tool]",
            )

    assert cached.messages == uncached.messages
    assert cached.system_prompt == uncached.system_prompt
    assert cached.session_id == uncached.session_id
    assert cached.user_facts == uncached.user_facts
    assert cached.prompt == uncached.prompt


def test_mismatched_cache_request_identity_bypasses_cache() -> None:
    builder = _make_builder(user_id="default")
    request = _context_cache_request("alice")

    with (
        patch(
            "rex.context.builder.build_context_cache_versions",
            return_value=_context_cache_versions(),
        ),
        patch.object(
            builder, "_get_active_personality_prompt", return_value="[persona]"
        ) as personality,
        patch.object(type(builder), "build_system_context", return_value="[sys]"),
    ):
        builder.build("one", active_user_id="cole", cache_request=request)
        builder.build("two", active_user_id="cole", cache_request=request)

    assert personality.call_count == 2
