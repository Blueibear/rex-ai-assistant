from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

import rex.assistant as assistant_module


class DummyStrategy:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, config=None):
        self.calls.append(prompt)
        return "hello"


class DummyPlugin:
    name = "dummy"

    def __init__(self):
        self.initialised = True

    def initialize(self):
        self.initialised = True

    def process(self, transcript):
        return "plugin info"

    def shutdown(self):
        pass


async def _run_assistant(assistant):
    return await assistant.generate_reply("hi")


def test_assistant_generates_reply(monkeypatch, tmp_path):
    dummy_strategy = DummyStrategy()

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            self.strategy = dummy_strategy

        def generate(self, prompt, config=None):
            return dummy_strategy.generate(prompt, config)

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    plugin_spec = assistant_module.PluginSpec(name="dummy", plugin=DummyPlugin())
    assistant = assistant_module.Assistant(plugins=[plugin_spec], transcripts_dir=tmp_path)

    reply = asyncio.run(_run_assistant(assistant))

    assert "hello" in reply
    assert "plugin info" in reply
    assert any("user:" in call for call in dummy_strategy.calls)


def test_build_prompt_contains_date_and_time(monkeypatch, tmp_path):
    """_build_prompt should prepend current date/time to the system context."""

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, config=None):
            return "ok"

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)
    prompt = asst._build_prompt("hello")

    # Prompt should start with "Current date and time: YYYY-MM-DD HH:MM <tz>"
    assert prompt.startswith(
        "Current date and time:"
    ), f"Expected prompt to start with date/time context, got: {prompt[:80]}"
    # Date portion must be today's UTC date (at minimum a valid YYYY-MM-DD pattern)
    assert re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", prompt
    ), "Expected YYYY-MM-DD HH:MM pattern in prompt"


def test_build_prompt_contains_location_when_configured(monkeypatch, tmp_path):
    """_build_prompt includes User location when default_location is set in settings."""

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, config=None):
            return "ok"

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    # Patch settings to include a default_location
    from rex.config import AppConfig

    settings_with_location = AppConfig(
        llm_provider="transformers",
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=settings_with_location)
    prompt = asst._build_prompt("hello")

    assert "User location: Dallas, TX" in prompt
    assert "America/Chicago" in prompt


def test_build_prompt_contains_tool_instructions(monkeypatch, tmp_path):
    """_build_prompt should include tool instructions so LLM can invoke tools."""

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, config=None):
            return "ok"

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)
    prompt = asst._build_prompt("What time is it in Dallas?")

    assert "TOOL_REQUEST" in prompt
    assert "time_now" in prompt
    assert "weather_now" in prompt
    assert "web_search" in prompt


def test_build_tool_context_with_settings(monkeypatch, tmp_path):
    """_build_tool_context returns location and timezone from settings."""

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, config=None):
            return "ok"

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    from rex.config import AppConfig

    settings_with_location = AppConfig(
        llm_provider="transformers",
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=settings_with_location)
    ctx = asst._build_tool_context()

    assert ctx["location"] == "Dallas, TX"
    assert ctx["timezone"] == "America/Chicago"


def test_followup_injected_at_most_once_with_concurrent_calls(monkeypatch, tmp_path):
    """Two concurrent generate_reply calls must inject the followup context at most once."""

    injected_prompts: list[str] = []

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt=None, *, messages=None, config=None, max_tool_rounds=3):
            if prompt and "[Note: You may want to ask" in prompt:
                injected_prompts.append(prompt)
            return "ok"

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)
    # Manually set a pending followup to simulate engine output
    asst._pending_followup = "How can I help you today?"

    async def run_two_concurrent():
        t1 = asyncio.create_task(asst.generate_reply("hello"))
        t2 = asyncio.create_task(asst.generate_reply("hi"))
        await asyncio.gather(t1, t2)

    asyncio.run(run_two_concurrent())

    assert len(injected_prompts) <= 1, (
        f"Followup context was injected {len(injected_prompts)} times; expected at most once"
    )


def _make_dummy_lm_class():
    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt=None, *, messages=None, config=None, max_tool_rounds=3):
            return "ok"

    return DummyLanguageModel


def test_history_store_saves_turns(monkeypatch, tmp_path):
    """generate_reply should persist user and assistant turns to HistoryStore."""
    monkeypatch.setattr(assistant_module, "LanguageModel", _make_dummy_lm_class())

    from rex.config import AppConfig
    from rex.history_store import HistoryStore

    db_path = tmp_path / "history.db"
    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=True,
        history_db_path=db_path,
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    asyncio.run(asst.generate_reply("hello"))

    store = HistoryStore(db_path=db_path)
    turns = store.load_history("default", limit=50)
    roles = [t["role"] for t in turns]
    assert "user" in roles
    assert "assistant" in roles
    contents = [t["content"] for t in turns]
    assert "hello" in contents
    assert "ok" in contents


def test_history_store_preloads_on_startup(monkeypatch, tmp_path):
    """Assistant should preload stored turns into in-memory history on startup."""
    monkeypatch.setattr(assistant_module, "LanguageModel", _make_dummy_lm_class())

    from rex.config import AppConfig
    from rex.history_store import HistoryStore

    db_path = tmp_path / "history.db"
    # Pre-seed the DB with a prior turn
    store = HistoryStore(db_path=db_path)
    store.save_turn("default", "user", "prior question", datetime.now(timezone.utc))
    store.save_turn("default", "assistant", "prior answer", datetime.now(timezone.utc))

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=True,
        history_db_path=db_path,
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    history = asst.history()
    speakers = [t.speaker for t in history]
    texts = [t.text for t in history]
    assert "user" in speakers
    assert "assistant" in speakers
    assert "prior question" in texts
    assert "prior answer" in texts


def test_history_not_persisted_when_disabled(monkeypatch, tmp_path):
    """When persist_history=False, no HistoryStore is created and no DB file is written."""
    monkeypatch.setattr(assistant_module, "LanguageModel", _make_dummy_lm_class())

    from rex.config import AppConfig

    db_path = tmp_path / "history.db"
    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=False,
        history_db_path=db_path,
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    asyncio.run(asst.generate_reply("hello"))

    assert asst._history_store is None
    assert not db_path.exists()


def test_chat_tool_request_routes_time_now(monkeypatch, tmp_path):
    """When LLM outputs a TOOL_REQUEST for time_now, it should be routed and re-called."""

    call_count = 0

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt=None, *, messages=None, config=None, max_tool_rounds=3):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: LLM decides to use the tool
                return (
                    'TOOL_REQUEST: {"tool": "time_now", ' '"args": {"location": "Dallas, Texas"}}'
                )
            # Second call (with tool result): LLM gives final answer
            return "The current local time in Dallas is 2026-03-20 01:37 CDT."

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)
    reply = asyncio.run(asst.generate_reply("What time is it in Dallas?"))

    assert call_count == 2, "LLM should be called twice: once for tool request, once with result"
    assert "Dallas" in reply
