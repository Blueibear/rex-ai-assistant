"""Tests for US-003: OpenAI strategy routed through OpenClaw chat completions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rex.config import AppConfig
from rex.llm_client import GenerationConfig, LanguageModel, OpenAIStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion_response(content: str, user_field: str | None = None) -> MagicMock:
    """Build a fake openai ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_config(
    *,
    base_url: str = "http://127.0.0.1:18789/v1",
    api_key: str = "test-token",
    user_id: str = "alice",
    active_profile: str = "default",
    llm_model: str = "openclaw:main",
) -> AppConfig:
    # AppConfig requires openai_model when llm_provider == "openai".
    return AppConfig(
        llm_provider="openai",
        llm_model=llm_model,
        openai_model=llm_model,
        openai_base_url=base_url,
        openai_api_key=api_key,
        user_id=user_id,
        active_profile=active_profile,
    )


# ---------------------------------------------------------------------------
# OpenAIStrategy unit tests
# ---------------------------------------------------------------------------


class TestOpenAIStrategyUserField:
    """OpenAIStrategy passes the user field through to the API call."""

    def _make_strategy(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion_response("hello")
        strategy = OpenAIStrategy("openclaw:main", lambda: mock_client)
        return strategy, mock_client

    def test_user_field_included_when_provided(self):
        strategy, mock_client = self._make_strategy()
        gen_cfg = GenerationConfig(
            max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50, seed=42
        )
        strategy.generate("hi", gen_cfg, user="alice")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["user"] == "alice"

    def test_user_field_omitted_when_none(self):
        strategy, mock_client = self._make_strategy()
        gen_cfg = GenerationConfig(
            max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50, seed=42
        )
        strategy.generate("hi", gen_cfg, user=None)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "user" not in call_kwargs

    def test_model_field_matches_strategy_model(self):
        strategy, mock_client = self._make_strategy()
        gen_cfg = GenerationConfig(
            max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50, seed=42
        )
        strategy.generate("hi", gen_cfg)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "openclaw:main"


# ---------------------------------------------------------------------------
# LanguageModel integration tests (mock openai SDK)
# ---------------------------------------------------------------------------


class TestLanguageModelOpenClawIntegration:
    """LanguageModel routes to OpenClaw gateway and sends correct fields."""

    def _make_lm_with_mock_client(self, config: AppConfig):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion_response("I am Rex.")
        lm = LanguageModel(config=config)
        # Inject mock client directly
        lm._openai_client = mock_client
        lm.strategy._cached_client = mock_client
        return lm, mock_client

    def test_authorization_bearer_header_via_api_key(self):
        """The openai client is initialised with the gateway token as api_key."""
        config = _make_config(api_key="secret-gateway-token")
        with (
            patch("rex.llm_client.OPENAI_AVAILABLE", True),
            patch("rex.llm_client.import_module") as mock_import,
        ):
            mock_openai_mod = MagicMock()
            mock_openai_instance = MagicMock()
            mock_openai_mod.OpenAI.return_value = mock_openai_instance
            mock_openai_mod.OpenAI.return_value.chat.completions.create.return_value = (
                _make_completion_response("hi")
            )
            mock_import.return_value = mock_openai_mod

            lm = LanguageModel(config=config)
            lm.generate("hello")

            # OpenAI was constructed with api_key = gateway token
            mock_openai_mod.OpenAI.assert_called_once_with(
                api_key="secret-gateway-token",
                base_url="http://127.0.0.1:18789/v1",
            )

    def test_model_field_matches_config(self):
        config = _make_config(llm_model="openclaw:main")
        lm = LanguageModel(config=config)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion_response("pong")
        lm._openai_client = mock_client
        lm.strategy._cached_client = mock_client

        lm.generate("ping")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "openclaw:main"

    def test_user_field_is_present_non_default_user_id(self):
        """user_id != 'default' → user field set to user_id."""
        config = _make_config(user_id="alice", active_profile="default")
        lm = LanguageModel(config=config)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion_response("hi")
        lm._openai_client = mock_client
        lm.strategy._cached_client = mock_client

        lm.generate("hello")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("user") == "alice"

    def test_user_field_falls_back_to_active_profile(self):
        """When user_id is 'default', user field falls back to active_profile."""
        config = _make_config(user_id="default", active_profile="work")
        lm = LanguageModel(config=config)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion_response("hi")
        lm._openai_client = mock_client
        lm.strategy._cached_client = mock_client

        lm.generate("hello")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("user") == "work"

    def test_messages_array_sent_in_openai_format(self):
        """Conversation messages are sent in standard OpenAI role/content format."""
        config = _make_config()
        lm = LanguageModel(config=config)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion_response("reply")
        lm._openai_client = mock_client
        lm.strategy._cached_client = mock_client

        messages = [
            {"role": "system", "content": "You are Rex."},
            {"role": "user", "content": "What time is it?"},
        ]
        lm.generate(messages=messages)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        sent_messages = call_kwargs["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["role"] == "user"

    def test_user_field_not_sent_for_non_openai_provider(self):
        """Non-OpenAI providers do not receive a user field."""

        config = AppConfig(llm_provider="echo", llm_model="echo")
        lm = LanguageModel(config=config)
        # Patch the strategy's generate to capture kwargs
        captured: list[dict] = []
        original = lm.strategy.generate

        def _capture(prompt, gen_cfg, **kwargs):
            captured.append(kwargs)
            return original(prompt, gen_cfg, **kwargs)

        lm.strategy.generate = _capture  # type: ignore[method-assign]
        lm.generate("hello")
        # EchoStrategy should not receive a user kwarg
        assert all("user" not in kw for kw in captured)
