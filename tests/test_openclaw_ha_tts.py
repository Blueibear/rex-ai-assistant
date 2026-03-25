"""Tests for US-P5-006: HA TTS through OpenClaw notification path.

Acceptance criteria:
  - HaTtsClient.speak() can be called from the OpenClaw-mediated notification channel
  - The ha_tts channel in notification.py correctly delegates to build_ha_tts_client
  - EventBridge + notification routing does not break HA TTS delivery
  - TtsResult.ok=True on success; TtsResult.ok=False on error

These tests exercise the full in-process path:
  notification channel routing → _send_to_ha_tts → build_ha_tts_client → speak()

Network calls are replaced by unittest.mock.patch. No real HA instance required.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from rex.ha_tts.client import HaTtsClient, TtsResult

# ---------------------------------------------------------------------------
# DNS mock for SSRF validator (tests that construct HaTtsClient directly)
# ---------------------------------------------------------------------------


def _public_addrinfo(*_args, **_kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


# ---------------------------------------------------------------------------
# US-P5-006: HaTtsClient.speak() — basic functionality
# ---------------------------------------------------------------------------


class TestHaTtsClientSpeak:
    """HaTtsClient.speak() called from OpenClaw notification context."""

    def _make_client(self) -> HaTtsClient:
        with patch("socket.getaddrinfo", side_effect=_public_addrinfo):
            return HaTtsClient(
                base_url="https://homeassistant.example.com:8123",
                token="test-token",
                default_entity_id="media_player.living_room",
                allow_http=False,
            )

    def test_speak_returns_ok_on_success(self):
        """speak() returns TtsResult(ok=True) when HA returns 200."""
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = client.speak("Good morning, James.")

        assert result.ok is True
        assert result.error is None
        mock_post.assert_called_once()

    def test_speak_sends_correct_payload(self):
        """speak() sends entity_id and message in the POST body."""
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.speak("Lights off in 5 minutes.", entity_id="media_player.bedroom")

        _args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["entity_id"] == "media_player.bedroom"
        assert payload["message"] == "Lights off in 5 minutes."

    def test_speak_uses_default_entity_id(self):
        """speak() falls back to default_entity_id when no override supplied."""
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.speak("Test announcement.")

        _args, kwargs = mock_post.call_args
        assert kwargs["json"]["entity_id"] == "media_player.living_room"

    def test_speak_returns_error_on_http_failure(self):
        """speak() returns TtsResult(ok=False) when the HA request fails."""
        import requests as _requests

        client = self._make_client()

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        http_error = _requests.HTTPError(response=mock_resp)

        with patch("requests.post", side_effect=http_error):
            result = client.speak("Alert message.")

        assert result.ok is False
        assert result.error is not None
        assert "token" not in (result.error or "").lower()

    def test_speak_empty_message_rejected(self):
        """speak() returns error without network call for empty message."""
        client = self._make_client()

        with patch("requests.post") as mock_post:
            result = client.speak("   ")

        assert result.ok is False
        assert result.error is not None
        mock_post.assert_not_called()

    def test_speak_no_entity_id_rejected(self):
        """speak() returns error when no entity_id and no default."""
        with patch("socket.getaddrinfo", side_effect=_public_addrinfo):
            client = HaTtsClient(
                base_url="https://homeassistant.example.com:8123",
                token="test-token",
                # no default_entity_id
            )

        with patch("requests.post") as mock_post:
            result = client.speak("Hello.")

        assert result.ok is False
        assert "entity_id" in (result.error or "").lower()
        mock_post.assert_not_called()

    def test_speak_with_extra_data_merged(self):
        """speak() merges extra_data into the HA service payload."""
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.speak("Bonjour.", extra_data={"language": "fr"})

        _args, kwargs = mock_post.call_args
        assert kwargs["json"]["language"] == "fr"

    def test_speak_uses_correct_tts_endpoint(self):
        """speak() POST goes to /api/services/{tts_domain}/{tts_service}."""
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client.speak("Test.")

        url = mock_post.call_args.args[0]
        assert "/api/services/tts/speak" in url


# ---------------------------------------------------------------------------
# US-P5-006: Notification channel routing — ha_tts channel
# ---------------------------------------------------------------------------


class TestHaTtsNotificationChannel:
    """Notification.py routes ha_tts channel to HaTtsClient.speak()."""

    def test_send_to_ha_tts_calls_speak(self, tmp_path):
        """_send_to_ha_tts calls speak() on the configured TTS client."""
        from rex.notification import NotificationRequest, Notifier

        mock_client = MagicMock()
        mock_client.speak.return_value = TtsResult(ok=True)

        notifier = Notifier(storage_path=tmp_path)

        req = NotificationRequest(
            title="Laundry",
            body="Your laundry is done.",
            channel_preferences=["ha_tts"],
        )

        with patch("rex.ha_tts.client.build_ha_tts_client", return_value=mock_client):
            notifier._send_to_ha_tts(req)

        mock_client.speak.assert_called_once()
        call_kwargs = mock_client.speak.call_args
        # message body is passed positionally or as kwarg
        args, kwargs = call_kwargs
        msg = args[0] if args else kwargs.get("message", "")
        assert "laundry" in msg

    def test_send_to_ha_tts_no_call_when_client_none(self, tmp_path):
        """_send_to_ha_tts is a no-op when build_ha_tts_client returns None."""
        from rex.notification import NotificationRequest, Notifier

        notifier = Notifier(storage_path=tmp_path)
        req = NotificationRequest(
            title="Alert",
            body="Alert.",
            channel_preferences=["ha_tts"],
        )

        with patch("rex.ha_tts.client.build_ha_tts_client", return_value=None):
            # Must not raise
            notifier._send_to_ha_tts(req)

    def test_send_to_ha_tts_entity_id_override_from_metadata(self, tmp_path):
        """entity_id from metadata is forwarded to speak()."""
        from rex.notification import NotificationRequest, Notifier

        mock_client = MagicMock()
        mock_client.speak.return_value = TtsResult(ok=True)

        notifier = Notifier(storage_path=tmp_path)
        req = NotificationRequest(
            title="Meeting Reminder",
            body="Reminder: meeting in 5 minutes.",
            channel_preferences=["ha_tts"],
            metadata={"ha_entity_id": "media_player.office"},
        )

        with patch("rex.ha_tts.client.build_ha_tts_client", return_value=mock_client):
            notifier._send_to_ha_tts(req)

        mock_client.speak.assert_called_once()
        _args, kwargs = mock_client.speak.call_args
        assert kwargs.get("entity_id") == "media_player.office"


# ---------------------------------------------------------------------------
# US-P5-006: build_ha_tts_client factory — integration checks
# ---------------------------------------------------------------------------


class TestBuildHaTtsClient:
    """build_ha_tts_client factory returns None when disabled or misconfigured."""

    def test_returns_none_when_disabled(self):
        """Returns None when config has enabled=false."""
        from rex.ha_tts.client import build_ha_tts_client
        from rex.ha_tts.config import HaTtsConfig

        with patch(
            "rex.ha_tts.config.load_ha_tts_config",
            return_value=HaTtsConfig(enabled=False),
        ):
            result = build_ha_tts_client()

        assert result is None

    def test_returns_none_when_base_url_missing(self):
        """Returns None when base_url is not set."""
        from rex.ha_tts.client import build_ha_tts_client
        from rex.ha_tts.config import HaTtsConfig

        with patch(
            "rex.ha_tts.config.load_ha_tts_config",
            return_value=HaTtsConfig(enabled=True, token_ref="my-token"),
        ):
            result = build_ha_tts_client()

        assert result is None

    def test_returns_none_when_token_ref_missing(self):
        """Returns None when token_ref is not set."""
        from rex.ha_tts.client import build_ha_tts_client
        from rex.ha_tts.config import HaTtsConfig

        with patch(
            "rex.ha_tts.config.load_ha_tts_config",
            return_value=HaTtsConfig(enabled=True, base_url="https://homeassistant.example.com"),
        ):
            result = build_ha_tts_client()

        assert result is None

    def test_returns_none_when_token_resolution_fails(self):
        """Returns None when CredentialManager cannot resolve the token."""
        from rex.ha_tts.client import build_ha_tts_client
        from rex.ha_tts.config import HaTtsConfig

        with patch(
            "rex.ha_tts.config.load_ha_tts_config",
            return_value=HaTtsConfig(
                enabled=True,
                base_url="https://homeassistant.example.com",
                token_ref="ha-token",
            ),
        ):
            with patch("rex.credentials.CredentialManager.get_token", return_value=None):
                result = build_ha_tts_client()

        assert result is None
