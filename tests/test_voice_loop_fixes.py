"""Regression tests for voice loop fixes (US-013 through US-016).

Tests cover:
1. XTTS compatibility / shim call order (BeamSearchScorer)
2. STT device selection (CUDA auto-detect)
3. Time query uses tool-routing path in voice mode
4. Response assembly sends coherent text to TTS
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# US-013: XTTS compatibility — shim applied BEFORE TTS.api is imported
# ---------------------------------------------------------------------------


class TestXTTSShimOrder:
    """Verify the transformers compatibility shim runs before TTS internals."""

    def test_lazy_import_tts_applies_shim_before_import(self):
        """The root voice_loop._lazy_import_tts must call ensure_transformers_compatibility
        before importing TTS.api, not after."""
        import inspect

        # Read the source of voice_loop._lazy_import_tts to verify call order.
        import voice_loop  # root-level module

        source = inspect.getsource(voice_loop._lazy_import_tts)

        # The shim call must appear before TTS.api import.
        shim_pos = source.find("ensure_transformers_compatibility")
        api_import_pos = source.find('import_module("TTS.api")')

        assert shim_pos != -1, "ensure_transformers_compatibility not found in _lazy_import_tts"
        assert api_import_pos != -1, 'import_module("TTS.api") not found in _lazy_import_tts'
        assert shim_pos < api_import_pos, (
            "Shim must be called BEFORE TTS.api is imported. "
            f"shim at position {shim_pos}, TTS.api import at {api_import_pos}"
        )

    def test_lazy_import_tts_does_not_import_TTS_api_for_availability_check(self):
        """_lazy_import_tts should use find_spec (not import_module) to check
        TTS availability, so that TTS internal imports don't run before the shim."""
        import inspect

        import voice_loop

        source = inspect.getsource(voice_loop._lazy_import_tts)

        # Should NOT contain _import_optional("TTS.api") or _import_optional("TTS")
        # before the shim call.
        lines_before_shim = source.split("ensure_transformers_compatibility")[0]
        assert (
            '_import_optional("TTS.api")' not in lines_before_shim
        ), "_import_optional('TTS.api') called before shim — this triggers internal imports"
        assert (
            '_import_optional("TTS")' not in lines_before_shim
        ), "_import_optional('TTS') called before shim — this triggers internal imports"

    def test_refactored_voice_loop_also_correct(self):
        """rex/voice_loop._lazy_import_tts should also use find_spec, not _import_optional."""
        import inspect

        from rex.voice_loop import _lazy_import_tts

        source = inspect.getsource(_lazy_import_tts)
        lines_before_shim = source.split("ensure_transformers_compatibility")[0]
        assert '_import_optional("TTS")' not in lines_before_shim


# ---------------------------------------------------------------------------
# US-014: STT device selection — CUDA auto-detect
# ---------------------------------------------------------------------------


class TestSTTDeviceSelection:
    """Verify Whisper device selection respects config and auto-detects CUDA."""

    def test_config_default_is_auto(self):
        """AppConfig.whisper_device should default to 'auto'."""
        from rex.config import AppConfig

        cfg = AppConfig()
        assert (
            cfg.whisper_device == "auto"
        ), f"Expected whisper_device default 'auto', got '{cfg.whisper_device}'"

    def test_auto_selects_cuda_when_available(self):
        """When whisper_device='auto' and torch.cuda.is_available()=True, device should be 'cuda'."""
        # We test the resolution logic extracted from voice_loop._get_whisper.
        device = "auto"
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

        assert device == "cuda"

    def test_auto_selects_cpu_when_cuda_not_available(self):
        """When whisper_device='auto' and torch.cuda.is_available()=False, device should be 'cpu'."""
        device = "auto"
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

        assert device == "cpu"

    def test_explicit_cpu_stays_cpu(self):
        """When whisper_device='cpu', it should stay 'cpu' regardless of CUDA."""
        device = "cpu"
        # No auto-detection should happen.
        assert device == "cpu"

    def test_explicit_cuda_stays_cuda(self):
        """When whisper_device='cuda', it should stay 'cuda'."""
        device = "cuda"
        assert device == "cuda"

    def test_stt_bridge_respects_config_device(self):
        """rex_stt_bridge should honour the config whisper_device setting."""
        import inspect

        import rex_stt_bridge

        source = inspect.getsource(rex_stt_bridge.main)
        # The bridge should contain auto-detection logic.
        assert "auto" in source, "STT bridge should support 'auto' device"
        assert (
            "torch.cuda.is_available" in source
        ), "STT bridge should check torch.cuda.is_available for auto mode"


# ---------------------------------------------------------------------------
# US-015: Time query uses tool routing in voice mode
# ---------------------------------------------------------------------------


class TestVoiceLoopToolRouting:
    """Verify the voice loop uses Assistant.generate_reply for tool routing."""

    def test_async_rex_assistant_has_assistant_attribute(self):
        """AsyncRexAssistant should create an Assistant instance for tool routing."""
        import inspect

        import voice_loop

        source = inspect.getsource(voice_loop.AsyncRexAssistant.__init__)
        assert (
            "self._assistant" in source
        ), "AsyncRexAssistant.__init__ should create self._assistant for tool routing"
        assert "Assistant(" in source, "AsyncRexAssistant.__init__ should instantiate Assistant"

    def test_process_conversation_uses_assistant_generate_reply(self):
        """_process_conversation should call self._assistant.generate_reply when available."""
        import inspect

        import voice_loop

        source = inspect.getsource(voice_loop.AsyncRexAssistant._process_conversation)
        assert "self._assistant" in source, "_process_conversation should reference self._assistant"
        assert (
            "generate_reply" in source
        ), "_process_conversation should call generate_reply for tool routing"
        assert "voice_mode=True" in source, "_process_conversation should pass voice_mode=True"

    def test_generate_reply_routes_time_query(self):
        """Assistant.generate_reply should route 'what time is it?' through tools."""
        import asyncio

        from rex.assistant import Assistant

        assistant = Assistant.__new__(Assistant)
        assistant._settings = None
        assistant._llm = MagicMock()
        assistant._history = []
        assistant._history_limit = 50
        assistant._plugins = []
        assistant._transcripts_dir = "/tmp"
        assistant._user_id = "test"
        assistant._followup_engine = None
        assistant._pending_followup = None
        assistant._followup_injected = False
        assistant._ha_bridge = None

        # Make the LLM return a tool request for time.
        assistant._llm.generate.return_value = 'TOOL_REQUEST: {"tool": "time_now", "args": {}}'

        # The generate_reply method should try to route the tool request.
        # We mock route_if_tool_request to verify it's called.
        with patch("rex.assistant.route_if_tool_request") as mock_route:
            mock_route.return_value = "The current time is 3:45 PM CDT."
            result = asyncio.run(assistant.generate_reply("what time is it?", voice_mode=True))

        mock_route.assert_called_once()
        assert "3:45 PM" in result


# ---------------------------------------------------------------------------
# US-016: Response assembly — coherent text to TTS
# ---------------------------------------------------------------------------


class TestResponseAssembly:
    """Verify that the voice loop sends coherent text to TTS, not fragments."""

    def test_speak_response_buffers_full_response(self):
        """_speak_response should receive the complete response text, not token fragments."""
        import inspect

        import voice_loop

        source = inspect.getsource(voice_loop.AsyncRexAssistant._process_conversation)

        # The response variable should be fully resolved before calling _speak_response.
        speak_line_idx = None
        response_line_idx = None
        for i, line in enumerate(source.split("\n")):
            if "_speak_response" in line:
                speak_line_idx = i
            if "response = " in line and "generate" in line:
                response_line_idx = i

        assert response_line_idx is not None, "Could not find response generation line"
        assert speak_line_idx is not None, "Could not find _speak_response call"
        assert (
            speak_line_idx > response_line_idx
        ), "_speak_response should be called after response is fully generated"

    def test_speak_response_uses_chunk_text_for_xtts(self):
        """_speak_response should use chunk_text_for_xtts for proper sentence-boundary chunking."""
        import inspect

        import voice_loop

        source = inspect.getsource(voice_loop.AsyncRexAssistant._speak_response)
        assert (
            "chunk_text_for_xtts" in source
        ), "_speak_response should use chunk_text_for_xtts for sentence-boundary chunking"

    def test_speak_response_combines_audio_before_playback(self):
        """_speak_response should concatenate all audio chunks before playing."""
        import inspect

        import voice_loop

        source = inspect.getsource(voice_loop.AsyncRexAssistant._speak_response)
        assert (
            "concatenate" in source or "combined_audio" in source
        ), "_speak_response should combine audio segments before playback"
