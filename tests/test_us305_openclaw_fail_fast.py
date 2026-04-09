"""Tests for US-305: OpenClaw voice backend fail-fast on startup.

Acceptance criteria:
  - When use_openclaw_voice_backend=False, VoiceBridge is never imported/instantiated
  - When use_openclaw_voice_backend=True but gateway URL not configured, raises RuntimeError
    with "no gateway URL" message
  - When use_openclaw_voice_backend=True but gateway unreachable, raises RuntimeError
    with "unreachable" message
  - When use_openclaw_voice_backend=True and gateway reachable, VoiceBridge is instantiated
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_voice_loop(mock_settings, mock_client=None, mock_bridge=None):
    """Construct a VoiceLoop inside appropriate patches."""
    from rex.voice_loop import VoiceLoop

    patches = [patch("rex.voice_loop.settings", mock_settings)]
    if mock_client is not None:
        patches.append(
            patch("rex.openclaw.http_client.get_openclaw_client", return_value=mock_client)
        )
    if mock_bridge is not None:
        patches.append(
            patch("rex.openclaw.voice_bridge.VoiceBridge", return_value=mock_bridge)
        )

    ctx = [p.__enter__() for p in patches]
    try:
        vl = VoiceLoop(
            MagicMock(),
            wake_listener=MagicMock(),
            detection_source=AsyncMock(),
            record_phrase=AsyncMock(return_value=MagicMock()),
            transcribe=AsyncMock(return_value=""),
            speak=AsyncMock(),
        )
    finally:
        for p in reversed(patches):
            p.__exit__(None, None, None)
    return vl


class TestUS305OpenClawFailFast:
    def test_flag_false_does_not_instantiate_voice_bridge(self):
        """When flag is False, VoiceBridge is never touched and base assistant is used."""
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = False
        base_assistant = MagicMock()

        with patch("rex.voice_loop.settings", mock_settings):
            from rex.voice_loop import VoiceLoop

            vl = VoiceLoop(
                base_assistant,
                wake_listener=MagicMock(),
                detection_source=AsyncMock(),
                record_phrase=AsyncMock(return_value=MagicMock()),
                transcribe=AsyncMock(return_value=""),
                speak=AsyncMock(),
            )

        assert vl._assistant is base_assistant

    def test_flag_true_no_gateway_url_raises_runtime_error(self):
        """Missing gateway URL raises RuntimeError with descriptive message."""
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True

        # get_openclaw_client returns None when no URL is configured
        with (
            patch("rex.voice_loop.settings", mock_settings),
            patch("rex.openclaw.http_client.get_openclaw_client", return_value=None),
        ):
            from rex.voice_loop import VoiceLoop

            with pytest.raises(RuntimeError, match="no gateway URL"):
                VoiceLoop(
                    MagicMock(),
                    wake_listener=MagicMock(),
                    detection_source=AsyncMock(),
                    record_phrase=AsyncMock(return_value=MagicMock()),
                    transcribe=AsyncMock(return_value=""),
                    speak=AsyncMock(),
                )

    def test_flag_true_gateway_unreachable_raises_runtime_error(self):
        """Unreachable gateway raises RuntimeError with 'unreachable' in message."""
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = "http://localhost:9999"

        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("Connection refused")

        with (
            patch("rex.voice_loop.settings", mock_settings),
            patch("rex.openclaw.http_client.get_openclaw_client", return_value=mock_client),
        ):
            from rex.voice_loop import VoiceLoop

            with pytest.raises(RuntimeError, match="unreachable"):
                VoiceLoop(
                    MagicMock(),
                    wake_listener=MagicMock(),
                    detection_source=AsyncMock(),
                    record_phrase=AsyncMock(return_value=MagicMock()),
                    transcribe=AsyncMock(return_value=""),
                    speak=AsyncMock(),
                )

    def test_flag_true_gateway_reachable_uses_voice_bridge(self):
        """Reachable gateway causes VoiceLoop to swap assistant for VoiceBridge."""
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = "http://localhost:8765"

        mock_client = MagicMock()
        mock_client.get.return_value = {"status": "ok"}
        mock_bridge_instance = MagicMock()

        with (
            patch("rex.voice_loop.settings", mock_settings),
            patch("rex.openclaw.http_client.get_openclaw_client", return_value=mock_client),
            patch("rex.openclaw.voice_bridge.VoiceBridge", return_value=mock_bridge_instance),
        ):
            from rex.voice_loop import VoiceLoop

            vl = VoiceLoop(
                MagicMock(),
                wake_listener=MagicMock(),
                detection_source=AsyncMock(),
                record_phrase=AsyncMock(return_value=MagicMock()),
                transcribe=AsyncMock(return_value=""),
                speak=AsyncMock(),
            )

        assert vl._assistant is mock_bridge_instance

    def test_runtime_error_message_includes_gateway_url(self):
        """RuntimeError for unreachable gateway includes the configured URL."""
        gateway_url = "http://my-openclaw-host:8765"
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = gateway_url

        mock_client = MagicMock()
        mock_client.get.side_effect = TimeoutError("timed out")

        with (
            patch("rex.voice_loop.settings", mock_settings),
            patch("rex.openclaw.http_client.get_openclaw_client", return_value=mock_client),
        ):
            from rex.voice_loop import VoiceLoop

            with pytest.raises(RuntimeError) as exc_info:
                VoiceLoop(
                    MagicMock(),
                    wake_listener=MagicMock(),
                    detection_source=AsyncMock(),
                    record_phrase=AsyncMock(return_value=MagicMock()),
                    transcribe=AsyncMock(return_value=""),
                    speak=AsyncMock(),
                )

        assert gateway_url in str(exc_info.value)
