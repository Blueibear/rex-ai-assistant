"""Unit tests for rex.autonomy.retry — retry_step and is_transient."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from rex.autonomy.retry import is_transient, retry_step

# ---------------------------------------------------------------------------
# is_transient classification tests
# ---------------------------------------------------------------------------


class TestIsTransient:
    def test_timeout_error_is_transient(self) -> None:
        assert is_transient(TimeoutError("timed out")) is True

    def test_connection_error_is_transient(self) -> None:
        assert is_transient(ConnectionError("refused")) is True

    def test_http_429_string_is_transient(self) -> None:
        assert is_transient(RuntimeError("HTTP 429 Too Many Requests")) is True

    def test_http_503_string_is_transient(self) -> None:
        assert is_transient(RuntimeError("503 Service Unavailable")) is True

    def test_value_error_is_not_transient(self) -> None:
        assert is_transient(ValueError("bad input")) is False

    def test_runtime_error_is_not_transient(self) -> None:
        assert is_transient(RuntimeError("unexpected crash")) is False

    def test_key_error_is_not_transient(self) -> None:
        assert is_transient(KeyError("missing")) is False

    def test_http_400_is_not_transient(self) -> None:
        assert is_transient(RuntimeError("HTTP 400 Bad Request")) is False

    def test_http_404_is_not_transient(self) -> None:
        assert is_transient(RuntimeError("404 Not Found")) is False

    def test_subclass_of_connection_error_is_transient(self) -> None:
        """BrokenPipeError is a subclass of ConnectionError."""
        assert is_transient(BrokenPipeError("pipe broken")) is True


# ---------------------------------------------------------------------------
# retry_step — success on first try
# ---------------------------------------------------------------------------


class TestRetryStepSuccess:
    def test_returns_result_on_first_success(self) -> None:
        fn = MagicMock(return_value="ok")
        result = retry_step(fn, "s1")
        assert result == "ok"

    def test_calls_fn_exactly_once_on_success(self) -> None:
        fn = MagicMock(return_value="ok")
        retry_step(fn, "s1")
        fn.assert_called_once()


# ---------------------------------------------------------------------------
# retry_step — transient errors are retried
# ---------------------------------------------------------------------------


class TestRetryStepTransient:
    @patch("rex.autonomy.retry.time.sleep")
    def test_transient_error_retries_and_succeeds(self, mock_sleep: MagicMock) -> None:
        """Transient error on attempt 1, success on attempt 2."""
        call_count = [0]

        def fn() -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("timed out")
            return "ok"

        result = retry_step(fn, "s1", max_attempts=3)
        assert result == "ok"
        assert call_count[0] == 2

    @patch("rex.autonomy.retry.time.sleep")
    def test_retries_up_to_max_attempts(self, mock_sleep: MagicMock) -> None:
        """Transient error on every attempt raises after max_attempts."""
        fn = MagicMock(side_effect=ConnectionError("refused"))

        with pytest.raises(ConnectionError):
            retry_step(fn, "s1", max_attempts=3)

        assert fn.call_count == 3

    @patch("rex.autonomy.retry.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep: MagicMock) -> None:
        """Sleep called with exponentially increasing delays."""
        fn = MagicMock(side_effect=TimeoutError("t/o"))

        with pytest.raises(TimeoutError):
            retry_step(fn, "s1", max_attempts=3, base_delay=1.0)

        # Attempt 1 fails → sleep(1.0); attempt 2 fails → sleep(2.0); attempt 3 fails → raise
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == pytest.approx(1.0)
        assert mock_sleep.call_args_list[1][0][0] == pytest.approx(2.0)

    @patch("rex.autonomy.retry.time.sleep")
    def test_no_sleep_after_final_attempt(self, mock_sleep: MagicMock) -> None:
        """sleep() is NOT called after the last failed attempt."""
        fn = MagicMock(side_effect=TimeoutError("t/o"))

        with pytest.raises(TimeoutError):
            retry_step(fn, "s1", max_attempts=1)

        mock_sleep.assert_not_called()

    @patch("rex.autonomy.retry.time.sleep")
    def test_http_429_is_retried(self, mock_sleep: MagicMock) -> None:
        call_count = [0]

        def fn() -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("HTTP 429 Too Many Requests")
            return "finally ok"

        result = retry_step(fn, "s1", max_attempts=3)
        assert result == "finally ok"

    @patch("rex.autonomy.retry.time.sleep")
    def test_http_503_is_retried(self, mock_sleep: MagicMock) -> None:
        call_count = [0]

        def fn() -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("503 Service Unavailable")
            return "recovered"

        result = retry_step(fn, "s1", max_attempts=2)
        assert result == "recovered"


# ---------------------------------------------------------------------------
# retry_step — non-transient errors fail immediately
# ---------------------------------------------------------------------------


class TestRetryStepNonTransient:
    @patch("rex.autonomy.retry.time.sleep")
    def test_value_error_not_retried(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            retry_step(fn, "s1", max_attempts=3)

        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("rex.autonomy.retry.time.sleep")
    def test_runtime_error_not_retried(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            retry_step(fn, "s1", max_attempts=3)

        fn.assert_called_once()

    @patch("rex.autonomy.retry.time.sleep")
    def test_http_400_not_retried(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=RuntimeError("HTTP 400 Bad Request"))

        with pytest.raises(RuntimeError):
            retry_step(fn, "s1", max_attempts=3)

        fn.assert_called_once()


# ---------------------------------------------------------------------------
# retry_step — logging
# ---------------------------------------------------------------------------


class TestRetryStepLogging:
    @patch("rex.autonomy.retry.time.sleep")
    def test_retry_logged_at_debug(
        self, mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        call_count = [0]

        def fn() -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("t/o")
            return "ok"

        with caplog.at_level(logging.DEBUG, logger="rex.autonomy.retry"):
            retry_step(fn, "my-step", max_attempts=3)

        assert any("Retrying step my-step" in r.message for r in caplog.records)

    @patch("rex.autonomy.retry.time.sleep")
    def test_log_message_contains_attempt_info(
        self, mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        fn = MagicMock(side_effect=ConnectionError("refused"))

        with caplog.at_level(logging.DEBUG, logger="rex.autonomy.retry"):
            with pytest.raises(ConnectionError):
                retry_step(fn, "step-42", max_attempts=3)

        retry_logs = [r.message for r in caplog.records if "Retrying step step-42" in r.message]
        assert len(retry_logs) == 2  # attempts 1 and 2 (not 3, no sleep after last)
        assert "1/3" in retry_logs[0]
        assert "2/3" in retry_logs[1]
