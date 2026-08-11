"""Regression tests for the superseding US-097 canonical voice contract.

US-305 originally made ``use_openclaw_voice_backend=True`` a startup dependency
that could replace Rex's Assistant with VoiceBridge. US-097 intentionally removes
that supported brain-swap path. The legacy flag is now compatibility-only: it logs
a warning, never probes the gateway, never instantiates VoiceBridge, and never
replaces the canonical TurnEngine-backed Assistant.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from rex.voice_loop import VoiceLoop


def _make_voice_loop(mock_settings: MagicMock, base_assistant: MagicMock) -> VoiceLoop:
    """Construct a canonical VoiceLoop with heavy voice dependencies stubbed."""
    with patch("rex.voice_loop.settings", mock_settings):
        return VoiceLoop(
            base_assistant,
            wake_listener=MagicMock(),
            detection_source=AsyncMock(),
            record_phrase=AsyncMock(return_value=MagicMock()),
            transcribe=AsyncMock(return_value=""),
            speak=AsyncMock(),
        )


class TestUS305SupersededVoiceBackendContract:
    def test_flag_false_keeps_assistant_and_never_touches_voice_bridge(self):
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = False
        base_assistant = MagicMock()

        with patch("rex.openclaw.voice_bridge.VoiceBridge") as mock_bridge_cls:
            vl = _make_voice_loop(mock_settings, base_assistant)

        assert vl._assistant is base_assistant
        mock_bridge_cls.assert_not_called()

    def test_flag_true_without_gateway_keeps_assistant_and_does_not_probe_gateway(self):
        """Legacy flag no longer makes OpenClaw availability a voice startup dependency."""
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = ""
        base_assistant = MagicMock()

        with (
            patch("rex.openclaw.http_client.get_openclaw_client") as mock_get_client,
            patch("rex.openclaw.voice_bridge.VoiceBridge") as mock_bridge_cls,
        ):
            vl = _make_voice_loop(mock_settings, base_assistant)

        assert vl._assistant is base_assistant
        mock_get_client.assert_not_called()
        mock_bridge_cls.assert_not_called()

    def test_flag_true_unreachable_gateway_is_not_consulted_by_voice_loop(self):
        """An unavailable external capability provider cannot block canonical voice startup."""
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = "http://localhost:9999"
        base_assistant = MagicMock()

        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("Connection refused")
        with (
            patch(
                "rex.openclaw.http_client.get_openclaw_client",
                return_value=mock_client,
            ) as mock_get_client,
            patch("rex.openclaw.voice_bridge.VoiceBridge") as mock_bridge_cls,
        ):
            vl = _make_voice_loop(mock_settings, base_assistant)

        assert vl._assistant is base_assistant
        mock_get_client.assert_not_called()
        mock_client.get.assert_not_called()
        mock_bridge_cls.assert_not_called()

    def test_flag_true_reachable_gateway_still_does_not_replace_assistant(self):
        """Reachability does not change the one-brain invariant."""
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = "http://localhost:8765"
        base_assistant = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = {"status": "ok"}
        with (
            patch(
                "rex.openclaw.http_client.get_openclaw_client",
                return_value=mock_client,
            ) as mock_get_client,
            patch("rex.openclaw.voice_bridge.VoiceBridge") as mock_bridge_cls,
        ):
            vl = _make_voice_loop(mock_settings, base_assistant)

        assert vl._assistant is base_assistant
        mock_get_client.assert_not_called()
        mock_client.get.assert_not_called()
        mock_bridge_cls.assert_not_called()

    def test_flag_true_emits_truthful_legacy_warning(self, caplog):
        mock_settings = MagicMock()
        mock_settings.use_openclaw_voice_backend = True
        mock_settings.openclaw_gateway_url = "http://my-openclaw-host:8765"
        base_assistant = MagicMock()

        with caplog.at_level(logging.WARNING, logger="rex.voice_loop"):
            vl = _make_voice_loop(mock_settings, base_assistant)

        assert vl._assistant is base_assistant
        assert any(
            "ignored by the canonical voice loop" in record.getMessage()
            and "TurnEngine brain" in record.getMessage()
            for record in caplog.records
        )
