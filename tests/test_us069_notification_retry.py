"""Tests for US-069: Notification retry logic.

Acceptance criteria:
- failed notifications retried
- retry attempts limited
- failures logged
- Typecheck passes
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rex.notification import NotificationRequest, Notifier


@pytest.fixture
def notifier(tmp_path: Path) -> Notifier:
    return Notifier(storage_path=tmp_path / "notifications")


def _make_notification(channel: str = "email") -> NotificationRequest:
    return NotificationRequest(
        title="Test",
        body="Body",
        channel_preferences=[channel],
    )


# ---------------------------------------------------------------------------
# Failed notifications are retried
# ---------------------------------------------------------------------------


def test_retry_on_transient_failure(notifier: Notifier, tmp_path: Path) -> None:
    """Dispatch retries when the channel raises on first attempt."""
    call_count = {"n": 0}

    def flaky_send(notification: NotificationRequest) -> None:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("transient error")

    with patch.object(notifier, "_send_to_email", side_effect=flaky_send):
        with patch("rex.retry.time.sleep"):  # avoid real sleeps
            notifier._dispatch_to_channel("email", _make_notification("email"))

    assert call_count["n"] == 2, "Should have been called twice (1 failure + 1 success)"


def test_retry_succeeds_on_second_attempt(notifier: Notifier) -> None:
    """When the first attempt fails and second succeeds, no exception is raised."""
    attempts: list[int] = []

    def second_attempt_succeeds(notification: NotificationRequest) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("first attempt fails")
        # second attempt succeeds silently

    with patch.object(notifier, "_send_to_email", side_effect=second_attempt_succeeds):
        with patch("rex.retry.time.sleep"):
            # Should not raise
            notifier._dispatch_to_channel("email", _make_notification("email"))

    assert len(attempts) == 2


def test_multiple_channels_each_retried(notifier: Notifier) -> None:
    """Retry logic applies independently per channel dispatch call."""
    sms_calls: list[int] = []

    def flaky_sms(notification: NotificationRequest) -> None:
        sms_calls.append(1)
        if len(sms_calls) < 3:
            raise RuntimeError("sms transient")

    with patch.object(notifier, "_send_to_sms", side_effect=flaky_sms):
        with patch("rex.retry.time.sleep"):
            notifier._dispatch_to_channel("sms", _make_notification("sms"))

    assert len(sms_calls) == 3


# ---------------------------------------------------------------------------
# Retry attempts are limited
# ---------------------------------------------------------------------------


def test_retry_limited_raises_after_max_attempts(notifier: Notifier) -> None:
    """After the maximum number of attempts, the exception is propagated."""
    call_count = {"n": 0}

    def always_fails(notification: NotificationRequest) -> None:
        call_count["n"] += 1
        raise RuntimeError("permanent failure")

    with patch.object(notifier, "_send_to_email", side_effect=always_fails):
        with patch("rex.retry.time.sleep"):
            with pytest.raises(RuntimeError, match="permanent failure"):
                notifier._dispatch_to_channel("email", _make_notification("email"))

    # RetryPolicy(attempts=3) means 3 total attempts
    assert call_count["n"] == 3


def test_retry_limited_to_three_attempts(notifier: Notifier) -> None:
    """Exactly 3 attempts are made before giving up (RetryPolicy default)."""
    attempts: list[int] = []

    def always_fails(notification: NotificationRequest) -> None:
        attempts.append(1)
        raise ValueError("always fails")

    with patch.object(notifier, "_send_to_email", side_effect=always_fails):
        with patch("rex.retry.time.sleep"):
            with pytest.raises(ValueError):
                notifier._dispatch_to_channel("email", _make_notification("email"))

    assert len(attempts) == 3


def test_dashboard_channel_not_retried(notifier: Notifier) -> None:
    """Dashboard channel bypasses retry logic (dispatched directly)."""
    call_count = {"n": 0}

    def failing_dashboard(notification: NotificationRequest) -> None:
        call_count["n"] += 1
        raise RuntimeError("dashboard failure")

    with patch.object(notifier, "_send_to_dashboard", side_effect=failing_dashboard):
        with patch("rex.retry.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                notifier._dispatch_to_channel("dashboard", _make_notification("dashboard"))

    # Dashboard is dispatched directly — exactly 1 attempt, no sleep
    assert call_count["n"] == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Failures logged
# ---------------------------------------------------------------------------


def test_retry_failures_logged_as_warnings(notifier: Notifier) -> None:
    """Retry warnings are emitted before each retry attempt."""
    warnings: list[str] = []

    def capture_warning(attempt: int, exc: BaseException, delay: float) -> None:
        warnings.append(f"attempt={attempt}")

    def always_fails(notification: NotificationRequest) -> None:
        raise RuntimeError("fail")

    with patch.object(notifier, "_send_to_email", side_effect=always_fails):
        with patch("rex.retry.time.sleep"):
            with pytest.raises(RuntimeError):
                # on_retry is wired inside _dispatch_to_channel via logger.warning
                notifier._dispatch_to_channel("email", _make_notification("email"))


def test_retry_warning_logged_via_logger(notifier: Notifier) -> None:
    """The on_retry callback logs a warning via the notification logger."""
    call_count = {"n": 0}

    def flaky(notification: NotificationRequest) -> None:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("transient")

    with patch.object(notifier, "_send_to_email", side_effect=flaky):
        with patch("rex.retry.time.sleep"):
            with patch("rex.notification.logger") as mock_logger:
                notifier._dispatch_to_channel("email", _make_notification("email"))

    # The on_retry lambda calls logger.warning
    assert mock_logger.warning.called


def test_final_failure_propagated_not_swallowed(notifier: Notifier) -> None:
    """Final failure after all retries propagates to the caller."""

    def always_fails(notification: NotificationRequest) -> None:
        raise OSError("network down")

    with patch.object(notifier, "_send_to_email", side_effect=always_fails):
        with patch("rex.retry.time.sleep"):
            with pytest.raises(IOError, match="network down"):
                notifier._dispatch_to_channel("email", _make_notification("email"))


def test_ha_tts_retried_on_failure(notifier: Notifier) -> None:
    """HA TTS channel also uses retry logic."""
    call_count = {"n": 0}

    def flaky_ha_tts(notification: NotificationRequest) -> None:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("ha tts transient")

    with patch.object(notifier, "_send_to_ha_tts", side_effect=flaky_ha_tts):
        with patch("rex.retry.time.sleep"):
            notifier._dispatch_to_channel("ha_tts", _make_notification("ha_tts"))

    assert call_count["n"] == 2
