"""Tests for US-105: Retry with exponential backoff for LLM provider calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from rex.retry import (
    NonRetryableError,
    RetryConfig,
    _get_status_code,
    _is_non_retryable,
    _is_transient,
    with_retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStatusError(Exception):
    """Simulates an HTTP-status-bearing exception (like openai.APIStatusError)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeResponseError(Exception):
    """Simulates an exception that carries a .response object."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code} via .response")
        self.response = MagicMock()
        self.response.status_code = status_code


class _FakeTimeoutError(Exception):
    """Simulates a timeout exception (no status code)."""


class _FakeConnectionError(ConnectionError):
    """Simulates a network connection failure."""


# ---------------------------------------------------------------------------
# RetryConfig dataclass
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_defaults(self) -> None:
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.base_delay == 1.0

    def test_custom_values(self) -> None:
        cfg = RetryConfig(max_attempts=5, base_delay=2.0)
        assert cfg.max_attempts == 5
        assert cfg.base_delay == 2.0

    def test_zero_delay_is_valid(self) -> None:
        cfg = RetryConfig(base_delay=0.0)
        assert cfg.base_delay == 0.0


# ---------------------------------------------------------------------------
# _get_status_code helper
# ---------------------------------------------------------------------------


class TestGetStatusCode:
    def test_direct_status_code_attribute(self) -> None:
        exc = _FakeStatusError(429)
        assert _get_status_code(exc) == 429

    def test_via_response_object(self) -> None:
        exc = _FakeResponseError(503)
        assert _get_status_code(exc) == 503

    def test_plain_exception_returns_none(self) -> None:
        assert _get_status_code(ValueError("oops")) is None

    def test_non_int_status_code_returns_none(self) -> None:
        exc = Exception("bad")
        exc.status_code = "not-an-int"  # type: ignore[attr-defined]
        assert _get_status_code(exc) is None


# ---------------------------------------------------------------------------
# _is_transient and _is_non_retryable helpers
# ---------------------------------------------------------------------------


class TestTransientCheck:
    def test_429_is_transient(self) -> None:
        assert _is_transient(_FakeStatusError(429))

    def test_503_is_transient(self) -> None:
        assert _is_transient(_FakeStatusError(503))

    def test_timeout_exception_is_transient(self) -> None:
        assert _is_transient(_FakeTimeoutError("timed out"))

    def test_connection_error_is_transient(self) -> None:
        assert _is_transient(_FakeConnectionError("connection refused"))

    def test_400_is_not_transient(self) -> None:
        assert not _is_transient(_FakeStatusError(400))

    def test_401_is_not_transient(self) -> None:
        assert not _is_transient(_FakeStatusError(401))

    def test_403_is_not_transient(self) -> None:
        assert not _is_transient(_FakeStatusError(403))

    def test_500_is_not_transient_by_code(self) -> None:
        # 500 is not in the known-transient set; may still be retried as "unexpected"
        assert not _is_transient(_FakeStatusError(500))


class TestNonRetryableCheck:
    def test_400_is_non_retryable(self) -> None:
        assert _is_non_retryable(_FakeStatusError(400))

    def test_401_is_non_retryable(self) -> None:
        assert _is_non_retryable(_FakeStatusError(401))

    def test_403_is_non_retryable(self) -> None:
        assert _is_non_retryable(_FakeStatusError(403))

    def test_429_is_not_non_retryable(self) -> None:
        assert not _is_non_retryable(_FakeStatusError(429))

    def test_503_is_not_non_retryable(self) -> None:
        assert not _is_non_retryable(_FakeStatusError(503))

    def test_plain_exception_is_not_non_retryable(self) -> None:
        assert not _is_non_retryable(ValueError("generic"))


# ---------------------------------------------------------------------------
# with_retry: success after N failures
# ---------------------------------------------------------------------------


class TestWithRetrySuccess:
    @patch("rex.retry.time.sleep")
    def test_succeeds_on_first_attempt(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(return_value="ok")
        result = with_retry(fn, config=RetryConfig(max_attempts=3, base_delay=1.0))
        assert result == "ok"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("rex.retry.time.sleep")
    def test_succeeds_after_one_failure(self, mock_sleep: MagicMock) -> None:
        """Mock fails once with transient 429, then succeeds."""
        side_effects: list[Any] = [_FakeStatusError(429), "hello"]
        fn = MagicMock(side_effect=side_effects)
        result = with_retry(fn, config=RetryConfig(max_attempts=3, base_delay=1.0))
        assert result == "hello"
        assert fn.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @patch("rex.retry.time.sleep")
    def test_succeeds_after_two_failures(self, mock_sleep: MagicMock) -> None:
        """Mock fails twice with transient errors, then succeeds on attempt 3."""
        side_effects: list[Any] = [
            _FakeStatusError(429),
            _FakeStatusError(503),
            "final",
        ]
        fn = MagicMock(side_effect=side_effects)
        result = with_retry(fn, config=RetryConfig(max_attempts=3, base_delay=1.0))
        assert result == "final"
        assert fn.call_count == 3
        # delay doubles: 1.0, then 2.0
        assert mock_sleep.call_args_list == [call(1.0), call(2.0)]

    @patch("rex.retry.time.sleep")
    def test_succeeds_after_n_failures_custom_config(self, mock_sleep: MagicMock) -> None:
        """Custom config: max_attempts=5, base_delay=0.5 — fails 4 times then succeeds."""
        side_effects: list[Any] = [
            _FakeStatusError(503),
            _FakeStatusError(503),
            _FakeStatusError(503),
            _FakeStatusError(503),
            "result",
        ]
        fn = MagicMock(side_effect=side_effects)
        result = with_retry(fn, config=RetryConfig(max_attempts=5, base_delay=0.5))
        assert result == "result"
        assert fn.call_count == 5
        # Delays: 0.5, 1.0, 2.0, 4.0
        assert mock_sleep.call_args_list == [
            call(0.5),
            call(1.0),
            call(2.0),
            call(4.0),
        ]


# ---------------------------------------------------------------------------
# with_retry: non-retryable errors are not retried
# ---------------------------------------------------------------------------


class TestWithRetryNonRetryable:
    @patch("rex.retry.time.sleep")
    def test_400_raises_immediately(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=_FakeStatusError(400))
        with pytest.raises(NonRetryableError) as exc_info:
            with_retry(fn, config=RetryConfig(max_attempts=3, base_delay=1.0))
        assert exc_info.value.status_code == 400
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("rex.retry.time.sleep")
    def test_401_raises_immediately(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=_FakeStatusError(401))
        with pytest.raises(NonRetryableError) as exc_info:
            with_retry(fn, config=RetryConfig(max_attempts=3, base_delay=1.0))
        assert exc_info.value.status_code == 401
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("rex.retry.time.sleep")
    def test_403_raises_immediately(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=_FakeStatusError(403))
        with pytest.raises(NonRetryableError) as exc_info:
            with_retry(fn, config=RetryConfig(max_attempts=3, base_delay=1.0))
        assert exc_info.value.status_code == 403
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("rex.retry.time.sleep")
    def test_non_retryable_wraps_original_exception(self, mock_sleep: MagicMock) -> None:
        original = _FakeStatusError(401)
        fn = MagicMock(side_effect=original)
        with pytest.raises(NonRetryableError) as exc_info:
            with_retry(fn, config=RetryConfig(max_attempts=3))
        assert exc_info.value.__cause__ is original

    @patch("rex.retry.time.sleep")
    def test_non_retryable_via_response_object(self, mock_sleep: MagicMock) -> None:
        """NonRetryableError also triggered when status code comes from .response."""
        fn = MagicMock(side_effect=_FakeResponseError(403))
        with pytest.raises(NonRetryableError) as exc_info:
            with_retry(fn, config=RetryConfig(max_attempts=3))
        assert exc_info.value.status_code == 403
        fn.assert_called_once()


# ---------------------------------------------------------------------------
# with_retry: max attempts exhausted
# ---------------------------------------------------------------------------


class TestWithRetryExhausted:
    @patch("rex.retry.time.sleep")
    def test_raises_after_max_attempts_transient(self, mock_sleep: MagicMock) -> None:
        """After max_attempts all fail with 429, the original exception is re-raised."""
        err = _FakeStatusError(429)
        fn = MagicMock(side_effect=err)
        with pytest.raises(_FakeStatusError):
            with_retry(fn, config=RetryConfig(max_attempts=3, base_delay=1.0))
        assert fn.call_count == 3
        # Two sleeps before the final attempt
        assert mock_sleep.call_count == 2

    @patch("rex.retry.time.sleep")
    def test_raises_after_max_attempts_generic(self, mock_sleep: MagicMock) -> None:
        """A generic exception with no status code is still retried and eventually re-raised."""
        err = RuntimeError("unknown failure")
        fn = MagicMock(side_effect=err)
        with pytest.raises(RuntimeError, match="unknown failure"):
            with_retry(fn, config=RetryConfig(max_attempts=2, base_delay=0.5))
        assert fn.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("rex.retry.time.sleep")
    def test_max_attempts_one_means_no_retry(self, mock_sleep: MagicMock) -> None:
        """max_attempts=1 means a single call — no retries at all."""
        err = _FakeStatusError(503)
        fn = MagicMock(side_effect=err)
        with pytest.raises(_FakeStatusError):
            with_retry(fn, config=RetryConfig(max_attempts=1, base_delay=1.0))
        fn.assert_called_once()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# with_retry: exponential backoff delay values
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    @patch("rex.retry.time.sleep")
    def test_delay_doubles_each_attempt(self, mock_sleep: MagicMock) -> None:
        """Verify sleep durations follow base_delay * 2^n progression."""
        side_effects: list[Any] = [
            _FakeStatusError(429),
            _FakeStatusError(429),
            _FakeStatusError(429),
            "success",
        ]
        fn = MagicMock(side_effect=side_effects)
        with_retry(fn, config=RetryConfig(max_attempts=4, base_delay=0.25))
        # Delays: 0.25, 0.5, 1.0
        assert mock_sleep.call_args_list == [call(0.25), call(0.5), call(1.0)]

    @patch("rex.retry.time.sleep")
    def test_zero_base_delay(self, mock_sleep: MagicMock) -> None:
        """base_delay=0.0 means no sleep between retries."""
        side_effects: list[Any] = [_FakeStatusError(503), "ok"]
        fn = MagicMock(side_effect=side_effects)
        with_retry(fn, config=RetryConfig(max_attempts=2, base_delay=0.0))
        mock_sleep.assert_called_once_with(0.0)


# ---------------------------------------------------------------------------
# with_retry: default config
# ---------------------------------------------------------------------------


class TestWithRetryDefaultConfig:
    @patch("rex.retry.time.sleep")
    def test_default_config_max_attempts_three(self, mock_sleep: MagicMock) -> None:
        """When config=None the default RetryConfig(max_attempts=3, base_delay=1.0) is used."""
        err = _FakeStatusError(503)
        fn = MagicMock(side_effect=err)
        with pytest.raises(_FakeStatusError):
            with_retry(fn)
        assert fn.call_count == 3

    @patch("rex.retry.time.sleep")
    def test_default_config_base_delay_one_second(self, mock_sleep: MagicMock) -> None:
        side_effects: list[Any] = [_FakeStatusError(503), "done"]
        fn = MagicMock(side_effect=side_effects)
        with_retry(fn)
        mock_sleep.assert_called_once_with(1.0)


# ---------------------------------------------------------------------------
# Integration: LLM strategy classes use with_retry
# ---------------------------------------------------------------------------


class TestLLMStrategyRetryIntegration:
    """Verify OpenAIStrategy and AnthropicStrategy wire up with_retry correctly."""

    @patch("rex.retry.time.sleep")
    def test_openai_strategy_retries_on_rate_limit(self, mock_sleep: MagicMock) -> None:
        from rex.llm_client import GenerationConfig, OpenAIStrategy

        call_count = 0

        def _fake_call(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _FakeStatusError(429)
            resp = MagicMock()
            resp.choices[0].message.tool_calls = None
            resp.choices[0].message.content = "hello from openai"
            return resp

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _fake_call

        strategy = OpenAIStrategy(
            "gpt-4",
            client_factory=lambda: fake_client,
            retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
        )
        gen_cfg = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=1.0, top_k=50, seed=0)
        result = strategy.generate("hi", gen_cfg)
        assert result == "hello from openai"
        assert call_count == 3
        # Two sleeps (0.0 seconds each, but still called)
        assert mock_sleep.call_count == 2

    @patch("rex.retry.time.sleep")
    def test_openai_strategy_does_not_retry_on_401(self, mock_sleep: MagicMock) -> None:
        from rex.llm_client import GenerationConfig, OpenAIStrategy

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _FakeStatusError(401)

        strategy = OpenAIStrategy(
            "gpt-4",
            client_factory=lambda: fake_client,
            retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
        )
        gen_cfg = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=1.0, top_k=50, seed=0)
        with pytest.raises(NonRetryableError) as exc_info:
            strategy.generate("hi", gen_cfg)
        assert exc_info.value.status_code == 401
        fake_client.chat.completions.create.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("rex.retry.time.sleep")
    def test_anthropic_strategy_retries_on_503(self, mock_sleep: MagicMock) -> None:
        from rex.llm_client import AnthropicStrategy, GenerationConfig

        call_count = 0

        def _fake_create(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _FakeStatusError(503)
            resp = MagicMock()
            resp.content = [MagicMock(text="anthropic reply")]
            return resp

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = _fake_create

        strategy = AnthropicStrategy(
            "claude-3",
            client_factory=lambda: fake_client,
            retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
        )
        gen_cfg = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=1.0, top_k=50, seed=0)
        result = strategy.generate("hi", gen_cfg)
        assert result == "anthropic reply"
        assert call_count == 2
        mock_sleep.assert_called_once()

    @patch("rex.retry.time.sleep")
    def test_anthropic_strategy_does_not_retry_on_403(self, mock_sleep: MagicMock) -> None:
        from rex.llm_client import AnthropicStrategy, GenerationConfig

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = _FakeStatusError(403)

        strategy = AnthropicStrategy(
            "claude-3",
            client_factory=lambda: fake_client,
            retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
        )
        gen_cfg = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=1.0, top_k=50, seed=0)
        with pytest.raises(NonRetryableError) as exc_info:
            strategy.generate("hi", gen_cfg)
        assert exc_info.value.status_code == 403
        fake_client.messages.create.assert_called_once()
        mock_sleep.assert_not_called()
