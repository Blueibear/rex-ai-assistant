"""Tests for the language model abstraction."""

from __future__ import annotations

import types

import pytest

from config import AppConfig
from llm_client import (
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    LanguageModel,
    register_strategy,
)


def test_language_model_generates_text():
    """Test standard generation using a Hugging Face model (tiny GPT-2)."""
    cfg = AppConfig(
        wakeword="rex",
        llm_model="sshleifer/tiny-gpt2",
        llm_provider="transformers",
        llm_max_tokens=12,
        llm_temperature=0.0,
    )
    model = LanguageModel(cfg)
    prompt = "User: Hello there!\nAssistant:"
    completion = model.generate(prompt)

    assert isinstance(completion, str)
    assert completion.strip() != ""

    if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
        assert completion.startswith("(offline)")


def test_language_model_accepts_messages():
    """Ensure the model can handle structured chat messages."""
    cfg = AppConfig(llm_model="sshleifer/tiny-gpt2", llm_provider="transformers")
    model = LanguageModel(cfg)

    messages = [
        {"role": "system", "content": "You are cheerful."},
        {"role": "user", "content": "Say hi"},
    ]
    completion = model.generate(messages=messages)

    assert isinstance(completion, str)
    assert completion.strip() != ""


def test_language_model_rejects_empty_prompt():
    """Empty or whitespace prompts should raise a ValueError."""
    cfg = AppConfig(llm_model="sshleifer/tiny-gpt2", llm_provider="transformers")
    model = LanguageModel(cfg)

    with pytest.raises(ValueError):
        model.generate("   ")


def test_invalid_model_name_rejected():
    """Reject path traversal attempts in model names."""
    with pytest.raises(ValueError):
        AppConfig(llm_model="../../bad_model")


def test_openai_provider(monkeypatch):
    """Simulate OpenAI provider and validate correct parsing of response."""
    fake_client = types.SimpleNamespace()

    class _Response:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="hello"))]

    fake_client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **_: _Response())
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(LanguageModel, "_ensure_openai_client", lambda self: fake_client)

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    model = LanguageModel(cfg)

    completion = model.generate(messages=[{"role": "user", "content": "hello"}])
    assert completion == "hello"


def test_openai_tool_calls_serialized_to_json(monkeypatch):
    """OpenAIStrategy.generate() must return a JSON string when tool_calls are present."""
    import json

    fake_client = types.SimpleNamespace()

    tool_call = types.SimpleNamespace(
        id="call_abc123",
        function=types.SimpleNamespace(
            name="web_search",
            arguments='{"query": "current weather"}',
        ),
    )
    message_with_tools = types.SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
    )

    class _ToolResponse:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=message_with_tools)]

    fake_client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **_: _ToolResponse())
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(LanguageModel, "_ensure_openai_client", lambda self: fake_client)

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    from rex.llm_client import OpenAIStrategy, GenerationConfig

    strategy = OpenAIStrategy("gpt-test", lambda: fake_client)
    gen_cfg = GenerationConfig(
        max_new_tokens=100, temperature=0.7, top_p=1.0, top_k=50, seed=42
    )
    result = strategy.generate("find weather", gen_cfg, messages=[{"role": "user", "content": "find weather"}])

    assert isinstance(result, str), "generate() must return str even with tool_calls"
    parsed = json.loads(result)
    assert "__tool_calls__" in parsed, "Result must contain '__tool_calls__' key"
    tc = parsed["__tool_calls__"][0]
    assert tc["function"]["name"] == "web_search"
    assert "current weather" in tc["function"]["arguments"]


def test_openai_strategy_stream_yields_token_deltas():
    from rex.llm_client import GenerationConfig, OpenAIStrategy

    fake_stream = [
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="Hello "))]
        ),
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="world"))]
        ),
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=None))]
        ),
    ]
    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **_: iter(fake_stream))
        )
    )

    strategy = OpenAIStrategy("gpt-test", lambda: fake_client)
    tokens = list(
        strategy.stream(
            "hello",
            GenerationConfig(max_new_tokens=32, temperature=0.0, top_p=1.0, top_k=0, seed=1),
        )
    )

    assert tokens == ["Hello ", "world"]
    assert "".join(tokens) == "Hello world"


def test_language_model_streams_echo_prompt_word_by_word():
    cfg = AppConfig(llm_provider="echo", llm_model="echo-test")
    model = LanguageModel(cfg)

    tokens = list(model.stream("hello streaming world"))

    assert tokens == ["hello ", "streaming ", "world"]
    assert "".join(tokens) == "hello streaming world"


def test_ollama_connection_refused_returns_friendly_message(monkeypatch):
    """ConnectionRefusedError → connection-failed sentinel string."""
    from rex.llm_client import OllamaStrategy, GenerationConfig

    strategy = OllamaStrategy.__new__(OllamaStrategy)
    strategy.model_name = "llama3"
    strategy.base_url = "http://localhost:11434"
    strategy.use_cloud = False
    strategy._ollama = None
    strategy._client = None
    strategy._retry_config = __import__("rex.retry", fromlist=["RetryConfig"]).RetryConfig(
        max_attempts=1
    )

    monkeypatch.setattr(
        "rex.llm_client.with_retry",
        lambda fn, **_kw: (_ for _ in ()).throw(ConnectionRefusedError("refused")),
    )

    result = strategy.generate("hello", GenerationConfig(100, 0.7, 1.0, 50, 42))
    assert result == "[Ollama: connection failed — is Ollama running?]"


def test_ollama_httpx_connect_error_returns_friendly_message(monkeypatch):
    """httpx.ConnectError → connection-failed sentinel string."""
    from rex.llm_client import OllamaStrategy, GenerationConfig

    strategy = OllamaStrategy.__new__(OllamaStrategy)
    strategy.model_name = "llama3"
    strategy.base_url = "http://localhost:11434"
    strategy.use_cloud = False
    strategy._ollama = None
    strategy._client = None
    strategy._retry_config = __import__("rex.retry", fromlist=["RetryConfig"]).RetryConfig(
        max_attempts=1
    )

    # Simulate an httpx.ConnectError by raising a class that looks like one
    class _FakeConnectError(Exception):
        pass

    _FakeConnectError.__name__ = "ConnectError"
    _FakeConnectError.__qualname__ = "ConnectError"
    _FakeConnectError.__module__ = "httpx"

    monkeypatch.setattr(
        "rex.llm_client.with_retry",
        lambda fn, **_kw: (_ for _ in ()).throw(_FakeConnectError("connect error")),
    )

    result = strategy.generate("hello", GenerationConfig(100, 0.7, 1.0, 50, 42))
    assert result == "[Ollama: connection failed — is Ollama running?]"


def test_ollama_model_not_found_returns_pull_hint(monkeypatch):
    """Exception with status_code=404 → model-not-found sentinel string."""
    from rex.llm_client import OllamaStrategy, GenerationConfig

    strategy = OllamaStrategy.__new__(OllamaStrategy)
    strategy.model_name = "no-such-model"
    strategy.base_url = "http://localhost:11434"
    strategy.use_cloud = False
    strategy._ollama = None
    strategy._client = None
    strategy._retry_config = __import__("rex.retry", fromlist=["RetryConfig"]).RetryConfig(
        max_attempts=1
    )

    class _ResponseError(Exception):
        status_code = 404

    monkeypatch.setattr(
        "rex.llm_client.with_retry",
        lambda fn, **_kw: (_ for _ in ()).throw(_ResponseError("model not found")),
    )

    result = strategy.generate("hello", GenerationConfig(100, 0.7, 1.0, 50, 42))
    assert result == (
        "[Ollama: model 'no-such-model' not found — run: ollama pull no-such-model]"
    )


def test_ollama_unexpected_error_returns_generic_message(monkeypatch):
    """Any other exception → generic unexpected-error sentinel string."""
    from rex.llm_client import OllamaStrategy, GenerationConfig

    strategy = OllamaStrategy.__new__(OllamaStrategy)
    strategy.model_name = "llama3"
    strategy.base_url = "http://localhost:11434"
    strategy.use_cloud = False
    strategy._ollama = None
    strategy._client = None
    strategy._retry_config = __import__("rex.retry", fromlist=["RetryConfig"]).RetryConfig(
        max_attempts=1
    )

    monkeypatch.setattr(
        "rex.llm_client.with_retry",
        lambda fn, **_kw: (_ for _ in ()).throw(RuntimeError("some weird error")),
    )

    result = strategy.generate("hello", GenerationConfig(100, 0.7, 1.0, 50, 42))
    assert result.startswith("[Ollama: unexpected error:")
    assert "some weird error" in result


def test_language_model_custom_strategy():
    """Test injecting a custom backend via register_strategy."""

    class DummyStrategy:
        name = "dummy"

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate(self, prompt: str, config):
            return f"dummy::{prompt}::{config.max_new_tokens}"

    register_strategy("dummy", DummyStrategy)

    cfg = AppConfig(llm_provider="dummy", llm_model="xyz", llm_max_tokens=5)
    model = LanguageModel(cfg)
    result = model.generate("test")

    assert result == "dummy::test::5"
