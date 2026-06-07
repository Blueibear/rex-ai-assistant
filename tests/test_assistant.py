from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime

import pytest

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
    # tz_name is appended to the date/time line; should appear when ZoneInfo works
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

    injected_inputs: list[object] = []

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt=None, *, messages=None, config=None, max_tool_rounds=3):
            if prompt and "[Note: You may want to ask" in prompt:
                injected_inputs.append(prompt)
            if messages and any(
                "You may want to ask" in str(message.get("content", ""))
                for message in messages
                if isinstance(message, dict)
            ):
                injected_inputs.append(messages)
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

    assert (
        len(injected_inputs) <= 1
    ), f"Followup context was injected {len(injected_inputs)} times; expected at most once"


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
    assert "Hello. How can I help?" in contents


def test_history_store_preloads_on_startup(monkeypatch, tmp_path):
    """Assistant should preload stored turns into in-memory history on startup."""
    monkeypatch.setattr(assistant_module, "LanguageModel", _make_dummy_lm_class())

    from rex.config import AppConfig
    from rex.history_store import HistoryStore

    db_path = tmp_path / "history.db"
    # Pre-seed the DB with a prior turn
    store = HistoryStore(db_path=db_path)
    store.save_turn("default", "user", "prior question", datetime.now(UTC))
    store.save_turn("default", "assistant", "prior answer", datetime.now(UTC))

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


def test_history_pruned_on_startup(monkeypatch, tmp_path):
    """Old turns beyond retention_days should be pruned when the assistant starts."""
    monkeypatch.setattr(assistant_module, "LanguageModel", _make_dummy_lm_class())

    from datetime import timedelta

    from rex.config import AppConfig
    from rex.history_store import HistoryStore

    db_path = tmp_path / "history.db"
    store = HistoryStore(db_path=db_path)
    now = datetime.now(UTC)
    old_ts = now - timedelta(days=40)
    recent_ts = now - timedelta(days=1)
    store.save_turn("default", "user", "very old message", old_ts)
    store.save_turn("default", "user", "recent message", recent_ts)

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=True,
        history_db_path=db_path,
        history_retention_days=30,
    )
    # Cancel the daily timer immediately after startup so it doesn't linger
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)
    if asst._prune_timer is not None:
        asst._prune_timer.cancel()

    remaining = store.load_history("default", limit=50)
    contents = [r["content"] for r in remaining]
    assert "very old message" not in contents, "Old turn should have been pruned on startup"
    assert "recent message" in contents, "Recent turn should be preserved"


def test_prune_idempotent_via_assistant(monkeypatch, tmp_path):
    """Running prune twice gives the same result as running once (idempotency)."""
    monkeypatch.setattr(assistant_module, "LanguageModel", _make_dummy_lm_class())

    from datetime import timedelta

    from rex.config import AppConfig
    from rex.history_store import HistoryStore

    db_path = tmp_path / "history.db"
    store = HistoryStore(db_path=db_path)
    old_ts = datetime.now(UTC) - timedelta(days=40)
    store.save_turn("default", "user", "old", old_ts)

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=True,
        history_db_path=db_path,
        history_retention_days=30,
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)
    if asst._prune_timer is not None:
        asst._prune_timer.cancel()

    # Call prune a second time manually — should delete 0 rows (already gone)
    second_deleted = asst._history_store.prune("default", keep_days=30)
    assert second_deleted == 0


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
    reply = asyncio.run(asst.generate_reply("Please check the clock for Dallas."))

    assert call_count == 2, "LLM should be called twice: once for tool request, once with result"
    assert "Dallas" in reply


def test_generate_reply_freeform_uses_structured_messages(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt=None, *, messages=None, config=None, max_tool_rounds=3):
            captured["prompt"] = prompt
            captured["messages"] = messages
            return "normal reply"

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)
    reply = asyncio.run(asst.generate_reply("Tell me something simple."))

    assert reply == "normal reply"
    assert captured["prompt"] is None
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "Tell me something simple."}
    assert not any(
        message.get("role") == "user" and "assistant:" in message.get("content", "")
        for message in messages
    )


def test_stream_reply_freeform_uses_structured_messages(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("streaming provider should not fall back to generate")

        def stream(self, prompt=None, *, messages=None, config=None):
            captured["prompt"] = prompt
            captured["messages"] = messages
            return iter(["normal ", "reply"])

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    async def collect():
        return [chunk async for chunk in asst.stream_reply("Tell me something simple.")]

    chunks = asyncio.run(collect())

    assert chunks == ["normal ", "reply"]
    assert captured["prompt"] is None
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "Tell me something simple."}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("hello", "Hello. How can I help?"),
        ("How are you?", "I'm here and ready to help."),
    ],
)
def test_generate_reply_direct_conversation_bypasses_llm(monkeypatch, tmp_path, query, expected):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct conversation reply should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct conversation reply should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    assert asyncio.run(asst.generate_reply(query)) == expected


def test_stream_reply_direct_conversation_bypasses_llm(monkeypatch, tmp_path):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct conversation reply should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct conversation reply should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    async def collect():
        return [chunk async for chunk in asst.stream_reply("hello")]

    assert asyncio.run(collect()) == ["Hello. How can I help?"]


def test_generate_reply_direct_recipe_bypasses_shopping_and_llm(monkeypatch, tmp_path):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct recipe reply should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct recipe reply should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    reply = asyncio.run(asst.generate_reply("I need a chocolate cake recipe"))

    assert "chocolate cake recipe" in reply.lower()
    assert "shopping list" not in reply.lower()


def test_stream_reply_direct_recipe_bypasses_llm(monkeypatch, tmp_path):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct recipe reply should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct recipe reply should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    async def collect():
        return [
            chunk async for chunk in asst.stream_reply("Can you give me a chocolate cake recipe?")
        ]

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    assert "chocolate cake recipe" in chunks[0].lower()
    assert "shopping list" not in chunks[0].lower()


def test_generate_reply_suppresses_unverified_action_claim(monkeypatch, tmp_path):
    class ClaimingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return "I added that to your shopping list."

        def stream(self, *args, **kwargs):
            return iter(["I added that to your shopping list."])

    monkeypatch.setattr(assistant_module, "LanguageModel", ClaimingLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    reply = asyncio.run(asst.generate_reply("Tell me something useful"))

    assert "did not change anything" in reply
    assert "shopping list" not in reply


def test_generate_reply_creator_question_bypasses_action_guard(monkeypatch, tmp_path):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("creator question should use direct reply")

        def stream(self, *args, **kwargs):
            raise AssertionError("creator question should use direct reply")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    reply = asyncio.run(asst.generate_reply("Who created you?"))

    assert "AskRex" in reply
    assert "did not change anything" not in reply


def test_generate_reply_biographical_created_text_is_not_action_claim(monkeypatch, tmp_path):
    class OriginLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return "I was created to run locally from this project."

        def stream(self, *args, **kwargs):
            return iter(["I was created to run locally from this project."])

    monkeypatch.setattr(assistant_module, "LanguageModel", OriginLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    reply = asyncio.run(asst.generate_reply("Tell me about your origin."))

    assert reply == "I was created to run locally from this project."
    assert "did not change anything" not in reply


def test_stream_reply_buffers_tool_request_until_resolved(monkeypatch, tmp_path):
    class StreamingToolLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return "It's 9:00 AM in New York."

        def stream(self, *args, **kwargs):
            return iter(
                [
                    "TOOL",
                    '_REQUEST: {"tool": "time_now", ',
                    '"args": {"location": "New York, NY"}}',
                ]
            )

    monkeypatch.setattr(assistant_module, "LanguageModel", StreamingToolLanguageModel)

    def fake_execute_tool(*args, **kwargs):
        return {
            "local_time": "2026-04-22 09:00",
            "date": "2026-04-22",
            "timezone": "America/New_York",
        }

    monkeypatch.setattr("rex.openclaw.tool_executor.execute_tool", fake_execute_tool)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    async def collect():
        return [chunk async for chunk in asst.stream_reply("Use the time tool for New York.")]

    chunks = asyncio.run(collect())

    assert chunks == ["It's 9:00 AM in New York."]
    assert "TOOL_REQUEST" not in "".join(chunks)


def test_stream_reply_suppresses_unverified_action_claim(monkeypatch, tmp_path):
    class ClaimingStreamLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return "I added that to your shopping list."

        def stream(self, *args, **kwargs):
            return iter(["I added", " that to your shopping list."])

    monkeypatch.setattr(assistant_module, "LanguageModel", ClaimingStreamLanguageModel)

    asst = assistant_module.Assistant(transcripts_dir=tmp_path)

    async def collect():
        return [chunk async for chunk in asst.stream_reply("Tell me something useful")]

    chunks = asyncio.run(collect())

    assert chunks == [
        "I did not change anything. Please tell me exactly what you want me to add, "
        "send, save, or update."
    ]


def test_generate_reply_direct_time_query_bypasses_llm(monkeypatch, tmp_path):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct time query should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct time query should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    from rex.config import AppConfig

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=False,
        response_cache_ttl=0,
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    reply = asyncio.run(asst.generate_reply("What time is it?"))

    assert reply.startswith("It's ")
    assert "Dallas, TX" in reply


DIRECT_CITY_TIME_QUERIES = [
    ("What time is it in New York?", "New York"),
    ("What time is it in New York right now?", "New York"),
    ("What time is it in New York, NY?", "New York, NY"),
]


@pytest.mark.parametrize(("query", "location_label"), DIRECT_CITY_TIME_QUERIES)
def test_generate_reply_direct_time_query_uses_requested_city(
    monkeypatch,
    tmp_path,
    query,
    location_label,
):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct time query should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct time query should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    from rex.config import AppConfig

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=False,
        response_cache_ttl=0,
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    reply = asyncio.run(asst.generate_reply(query))

    assert reply.startswith("It's ")
    assert location_label in reply
    assert "Dallas" not in reply


DIRECT_DAY_DATE_QUERIES = [
    "What day is today?",
    "What day is it today?",
    "What's the day today?",
    "Whats the day today?",
    "What day is it?",
    "What's today's date?",
    "What is today's date?",
    "Whats todays date?",
]


@pytest.mark.parametrize("query", DIRECT_DAY_DATE_QUERIES)
def test_generate_reply_direct_day_date_query_bypasses_llm(monkeypatch, tmp_path, query):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct day/date query should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct day/date query should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    from rex.config import AppConfig

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=False,
        response_cache_ttl=0,
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    reply = asyncio.run(asst.generate_reply(query))

    assert reply.startswith("Today is ")
    assert "Dallas, TX" in reply


def test_stream_reply_direct_time_query_bypasses_llm(monkeypatch, tmp_path):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct time query should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct time query should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    from rex.config import AppConfig

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=False,
        response_cache_ttl=0,
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    async def collect():
        return [chunk async for chunk in asst.stream_reply("What time is it?")]

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    assert chunks[0].startswith("It's ")
    assert "Dallas, TX" in chunks[0]


@pytest.mark.parametrize(("query", "location_label"), DIRECT_CITY_TIME_QUERIES)
def test_stream_reply_direct_time_query_uses_requested_city(
    monkeypatch,
    tmp_path,
    query,
    location_label,
):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct time query should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct time query should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    from rex.config import AppConfig

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=False,
        response_cache_ttl=0,
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    async def collect():
        return [chunk async for chunk in asst.stream_reply(query)]

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    assert chunks[0].startswith("It's ")
    assert location_label in chunks[0]
    assert "Dallas" not in chunks[0]


@pytest.mark.parametrize("query", DIRECT_DAY_DATE_QUERIES)
def test_stream_reply_direct_day_date_query_bypasses_llm(monkeypatch, tmp_path, query):
    class BlockingLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            raise AssertionError("direct day/date query should not call the LLM")

        def stream(self, *args, **kwargs):
            raise AssertionError("direct day/date query should not stream from the LLM")

    monkeypatch.setattr(assistant_module, "LanguageModel", BlockingLanguageModel)

    from rex.config import AppConfig

    cfg = AppConfig(
        llm_provider="transformers",
        persist_history=False,
        response_cache_ttl=0,
        default_location="Dallas, TX",
        default_timezone="America/Chicago",
    )
    asst = assistant_module.Assistant(transcripts_dir=tmp_path, settings_obj=cfg)

    async def collect():
        return [chunk async for chunk in asst.stream_reply(query)]

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    assert chunks[0].startswith("Today is ")
    assert "Dallas, TX" in chunks[0]
