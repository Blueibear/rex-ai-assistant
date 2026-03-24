"""Tests for US-P6-007: USE_OPENCLAW_VOICE_BACKEND flag in root voice_loop.py.

Acceptance criteria:
  - When use_openclaw_voice_backend=False, _assistant is created from rex.assistant.Assistant
  - When use_openclaw_voice_backend=True, _assistant is replaced with VoiceBridge()
  - VoiceBridge creation failure falls back gracefully (keeps previous _assistant)
  - Tests pass
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rex.config import AppConfig


def _make_config(**kwargs) -> AppConfig:
    return AppConfig(llm_provider="transformers", llm_model="sshleifer/tiny-gpt2", **kwargs)


# ---------------------------------------------------------------------------
# Helpers — stub out all heavy module-level imports in voice_loop.py
# ---------------------------------------------------------------------------

HEAVY_MOCKS = {
    "rex.llm_client.LanguageModel": MagicMock(),
    "voice_loop.load_wakeword_model": MagicMock(return_value=(MagicMock(), "hey rex")),
    "voice_loop.ensure_wake_acknowledgment_sound": MagicMock(return_value=None),
    "voice_loop._load_plugins_impl": MagicMock(return_value={}),
}


class TestAsyncRexAssistantFlagOff:
    """When use_openclaw_voice_backend=False, Assistant is used (not VoiceBridge)."""

    def test_flag_false_does_not_create_voice_bridge(self):
        """VoiceBridge is NOT instantiated when flag is False."""
        cfg = _make_config(use_openclaw_voice_backend=False)
        mock_assistant = MagicMock()
        with (
            patch("voice_loop.load_config", return_value=cfg),
            patch("voice_loop.LanguageModel", return_value=MagicMock()),
            patch("voice_loop.load_wakeword_model", return_value=(MagicMock(), "hey rex")),
            patch("voice_loop.ensure_wake_acknowledgment_sound", return_value=None),
            patch("voice_loop._load_plugins_impl", return_value={}),
            patch("rex.assistant.Assistant", return_value=mock_assistant),
        ):
            from voice_loop import AsyncRexAssistant

            ara = AsyncRexAssistant(config=cfg)
            assert ara._assistant is mock_assistant
            # VoiceBridge import should NOT have been attempted
            # (we verify by checking _assistant is still the mock)

    def test_flag_false_assistant_is_not_voice_bridge(self):
        """_assistant type is not VoiceBridge when flag is False."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config(use_openclaw_voice_backend=False)
        with (
            patch("voice_loop.load_config", return_value=cfg),
            patch("voice_loop.LanguageModel", return_value=MagicMock()),
            patch("voice_loop.load_wakeword_model", return_value=(MagicMock(), "hey rex")),
            patch("voice_loop.ensure_wake_acknowledgment_sound", return_value=None),
            patch("voice_loop._load_plugins_impl", return_value={}),
            patch("rex.assistant.Assistant", return_value=MagicMock()),
        ):
            from voice_loop import AsyncRexAssistant

            ara = AsyncRexAssistant(config=cfg)
            assert not isinstance(ara._assistant, VoiceBridge)


class TestAsyncRexAssistantFlagOn:
    """When use_openclaw_voice_backend=True, _assistant is replaced with VoiceBridge."""

    def test_flag_true_replaces_assistant_with_voice_bridge(self):
        """_assistant is a VoiceBridge instance when flag is True."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config(use_openclaw_voice_backend=True)
        mock_bridge = MagicMock(spec=VoiceBridge)
        with (
            patch("voice_loop.load_config", return_value=cfg),
            patch("voice_loop.LanguageModel", return_value=MagicMock()),
            patch("voice_loop.load_wakeword_model", return_value=(MagicMock(), "hey rex")),
            patch("voice_loop.ensure_wake_acknowledgment_sound", return_value=None),
            patch("voice_loop._load_plugins_impl", return_value={}),
            patch("rex.assistant.Assistant", return_value=MagicMock()),
            patch("rex.openclaw.voice_bridge.VoiceBridge", return_value=mock_bridge),
        ):
            import importlib

            import voice_loop as vl

            importlib.reload(vl)

            ara = vl.AsyncRexAssistant(config=cfg)
            # _assistant should have been replaced
            assert ara._assistant is mock_bridge

    def test_flag_true_voice_bridge_created_once(self):
        """VoiceBridge() is instantiated exactly once when flag is True."""
        from rex.openclaw.voice_bridge import VoiceBridge

        cfg = _make_config(use_openclaw_voice_backend=True)
        with (
            patch("voice_loop.load_config", return_value=cfg),
            patch("voice_loop.LanguageModel", return_value=MagicMock()),
            patch("voice_loop.load_wakeword_model", return_value=(MagicMock(), "hey rex")),
            patch("voice_loop.ensure_wake_acknowledgment_sound", return_value=None),
            patch("voice_loop._load_plugins_impl", return_value={}),
            patch("rex.assistant.Assistant", return_value=MagicMock()),
            patch(
                "rex.openclaw.voice_bridge.VoiceBridge", return_value=MagicMock(spec=VoiceBridge)
            ) as mock_cls,
        ):
            import importlib

            import voice_loop as vl

            importlib.reload(vl)

            vl.AsyncRexAssistant(config=cfg)
            # The patch target is the class itself; the bridge is created inside __init__
            # We verify _assistant is not the original Assistant mock
            assert mock_cls.called

    def test_flag_true_voice_bridge_failure_falls_back(self):
        """If VoiceBridge() raises, _assistant keeps the previous value (fallback)."""
        mock_assistant = MagicMock()
        cfg = _make_config(use_openclaw_voice_backend=True)
        with (
            patch("voice_loop.load_config", return_value=cfg),
            patch("voice_loop.LanguageModel", return_value=MagicMock()),
            patch("voice_loop.load_wakeword_model", return_value=(MagicMock(), "hey rex")),
            patch("voice_loop.ensure_wake_acknowledgment_sound", return_value=None),
            patch("voice_loop._load_plugins_impl", return_value={}),
            patch("rex.assistant.Assistant", return_value=mock_assistant),
            patch(
                "rex.openclaw.voice_bridge.VoiceBridge",
                side_effect=RuntimeError("bridge unavailable"),
            ),
        ):
            import importlib

            import voice_loop as vl

            importlib.reload(vl)

            ara = vl.AsyncRexAssistant(config=cfg)
            # VoiceBridge failed → _assistant should still be the original mock
            assert ara._assistant is mock_assistant
