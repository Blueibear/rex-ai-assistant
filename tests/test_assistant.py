from __future__ import annotations

import asyncio
import re

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
