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
            asst = mod.Assistant(transcripts_dir=tmp)

    prompt = asst._build_prompt("hello")
    assert "hello" in prompt
    assert "assistant:" in prompt
