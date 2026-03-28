"""Tests for VoiceBridge — US-P6-005.

Verifies that VoiceBridge:
- Exposes generate_reply(transcript, voice_mode=False) -> str
- Delegates to RexAgent.respond()
- Accepts voice_mode=True without error (for root/rex voice_loop.py compatibility)
- Accepts generate_reply(transcript) without voice_mode (for voice_loop_optimized.py)
- Raises ValueError on empty or whitespace-only transcript
- Accepts extra **kwargs without error (forward compatibility)
- Supports optional user_key for history persistence
- Lazily creates RexAgent when none is injected
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rex.openclaw.voice_bridge import VoiceBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(reply: str = "Hello from Rex.") -> tuple[VoiceBridge, MagicMock]:
    """Return (VoiceBridge, mock_agent) with the agent pre-injected."""
    mock_agent = MagicMock()
    mock_agent.respond.return_value = reply
    bridge = VoiceBridge(agent=mock_agent)
    return bridge, mock_agent


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestVoiceBridgeBasic:
    def test_generate_reply_returns_string(self):
        """generate_reply returns a non-empty string."""
        bridge, _ = _make_bridge("It is 3 PM.")
        result = bridge.generate_reply("What time is it?")
        assert isinstance(result, str)
        assert result == "It is 3 PM."

    def test_generate_reply_delegates_to_agent_respond(self):
        """generate_reply delegates to RexAgent.respond() with the transcript."""
        bridge, mock_agent = _make_bridge()
        bridge.generate_reply("Turn off the lights.")
        mock_agent.respond.assert_called_once_with("Turn off the lights.", user_key=None)

    def test_generate_reply_called_once_per_transcript(self):
        """Each generate_reply call invokes RexAgent.respond exactly once."""
        bridge, mock_agent = _make_bridge()
        bridge.generate_reply("Hello.")
        assert mock_agent.respond.call_count == 1


# ---------------------------------------------------------------------------
# voice_mode compatibility (all three voice loop call signatures)
# ---------------------------------------------------------------------------


class TestVoiceModeSigCompatibility:
    def test_no_voice_mode_arg_works(self):
        """voice_loop_optimized.py: generate_reply(transcript) with no voice_mode."""
        bridge, _ = _make_bridge("OK")
        result = bridge.generate_reply("Set a timer.")
        assert result == "OK"

    def test_voice_mode_true_works(self):
        """root/rex voice_loop.py: generate_reply(transcript, voice_mode=True)."""
        bridge, _ = _make_bridge("OK")
        result = bridge.generate_reply("Set a timer.", voice_mode=True)
        assert result == "OK"

    def test_voice_mode_false_works(self):
        """generate_reply(transcript, voice_mode=False) is the default signature."""
        bridge, _ = _make_bridge("OK")
        result = bridge.generate_reply("Set a timer.", voice_mode=False)
        assert result == "OK"

    def test_extra_kwargs_accepted(self):
        """Extra keyword arguments are silently ignored for forward compatibility."""
        bridge, _ = _make_bridge("OK")
        result = bridge.generate_reply("Hello.", voice_mode=True, future_param="ignored")
        assert result == "OK"

    def test_voice_mode_not_forwarded_to_agent(self):
        """voice_mode is NOT forwarded to RexAgent.respond (it has no such param)."""
        bridge, mock_agent = _make_bridge()
        bridge.generate_reply("Hello.", voice_mode=True)
        call_kwargs = mock_agent.respond.call_args[1]
        assert "voice_mode" not in call_kwargs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestVoiceBridgeValidation:
    def test_empty_transcript_returns_empty_string(self):
        """generate_reply returns empty string for an empty transcript."""
        bridge, _ = _make_bridge()
        assert bridge.generate_reply("") == ""

    def test_whitespace_only_transcript_returns_empty_string(self):
        """generate_reply returns empty string for whitespace-only transcript."""
        bridge, _ = _make_bridge()
        assert bridge.generate_reply("   ") == ""


# ---------------------------------------------------------------------------
# user_key forwarding
# ---------------------------------------------------------------------------


class TestVoiceBridgeUserKey:
    def test_user_key_forwarded_to_agent(self):
        """user_key set at construction is forwarded to RexAgent.respond()."""
        mock_agent = MagicMock()
        mock_agent.respond.return_value = "Hi Alice"
        bridge = VoiceBridge(agent=mock_agent, user_key="alice")
        bridge.generate_reply("Hello.")
        mock_agent.respond.assert_called_once_with("Hello.", user_key="alice")

    def test_no_user_key_by_default(self):
        """Without a user_key, respond() is called with user_key=None."""
        bridge, mock_agent = _make_bridge()
        bridge.generate_reply("Hello.")
        mock_agent.respond.assert_called_once_with("Hello.", user_key=None)


# ---------------------------------------------------------------------------
# Lazy agent initialisation
# ---------------------------------------------------------------------------


class TestVoiceBridgeLazyAgent:
    def test_agent_created_lazily(self):
        """When no agent is injected, RexAgent is created on first generate_reply call."""
        with patch("rex.openclaw.voice_bridge.RexAgent") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.respond.return_value = "Lazy reply"
            mock_cls.return_value = mock_instance

            bridge = VoiceBridge()
            result = bridge.generate_reply("Hello.")

            mock_cls.assert_called_once()
            assert result == "Lazy reply"

    def test_agent_created_only_once(self):
        """RexAgent is instantiated only once across multiple generate_reply calls."""
        with patch("rex.openclaw.voice_bridge.RexAgent") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.respond.return_value = "Reply"
            mock_cls.return_value = mock_instance

            bridge = VoiceBridge()
            bridge.generate_reply("First.")
            bridge.generate_reply("Second.")

            assert mock_cls.call_count == 1
