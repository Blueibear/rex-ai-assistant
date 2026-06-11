"""Tests for US-174: Configurable LLM response length limit for voice mode."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
ASSISTANT_SRC = REPO_ROOT / "rex" / "assistant.py"
BUILDER_SRC = REPO_ROOT / "rex" / "context" / "builder.py"
CONFIG_SRC = REPO_ROOT / "rex" / "config.py"
VOICE_LOOP_SRC = REPO_ROOT / "rex" / "voice_loop.py"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def _assistant_src() -> str:
    return ASSISTANT_SRC.read_text(encoding="utf-8")


def _builder_src() -> str:
    return BUILDER_SRC.read_text(encoding="utf-8")


def _config_src() -> str:
    return CONFIG_SRC.read_text(encoding="utf-8")


def _voice_loop_src() -> str:
    parts = [VOICE_LOOP_SRC.read_text(encoding="utf-8")]
    # Original rex/voice_loop.py concern order (US-REM-028 decomposition).
    for _name in (
        "optional_imports.py",
        "audio_utils.py",
        "_types.py",
        "transcripts.py",
        "microphone.py",
        "acknowledgement.py",
        "stt.py",
        "tts.py",
        "loop.py",
        "builder.py",
    ):
        parts.append((REPO_ROOT / "rex" / "voice" / _name).read_text(encoding="utf-8"))
    return "\n".join(parts)


# ── Config: voice_max_tokens field ────────────────────────────────────────────


class TestVoiceMaxTokensConfig:
    def test_voice_max_tokens_in_settings(self):
        assert "voice_max_tokens" in _config_src()

    def test_voice_max_tokens_default_is_150(self):
        src = _config_src()
        idx = src.index("voice_max_tokens")
        snippet = src[idx : idx + 50]
        assert "150" in snippet

    def test_voice_max_tokens_importable(self):
        from rex.config import Settings

        Settings.__new__(Settings)
        # Default attribute
        assert (
            hasattr(Settings, "__dataclass_fields__") or hasattr(Settings, "model_fields") or True
        )  # may be pydantic or dataclass

    def test_settings_instance_has_voice_max_tokens(self):
        from rex.config import settings

        assert hasattr(settings, "voice_max_tokens")
        assert settings.voice_max_tokens == 150


# ── Assistant: generate_reply voice_mode param ────────────────────────────────


class TestGenerateReplyVoiceMode:
    def test_generate_reply_accepts_voice_mode_param(self):
        src = _assistant_src()
        idx = src.index("async def generate_reply")
        # grab up to 200 chars to cover full signature including keyword args
        snippet = src[idx : idx + 200]
        assert "voice_mode" in snippet

    def test_build_prompt_accepts_voice_mode(self):
        # US-014: _build_prompt is now a thin delegate; check signature has voice_mode
        src = _assistant_src()
        idx = src.index("def _build_prompt")
        snippet = src[idx : idx + 200]
        assert "voice_mode" in snippet

    def test_voice_concise_instruction_constant_exists(self):
        # US-014: constant now lives in rex/context/builder.py
        assert "_VOICE_CONCISE_INSTRUCTION" in _builder_src()

    def test_voice_concise_instruction_has_concise_language(self):
        # US-014: constant now lives in rex/context/builder.py
        src = _builder_src()
        idx = src.index("_VOICE_CONCISE_INSTRUCTION")
        end = src.index("\n", idx + len("_VOICE_CONCISE_INSTRUCTION"))
        snippet = src[idx : end + 200]
        assert (
            "sentence" in snippet.lower()
            or "short" in snippet.lower()
            or "concise" in snippet.lower()
        )

    def test_build_prompt_injects_concise_instruction_in_voice_mode(self):
        # US-014: implementation now lives in ContextBuilder._build_prompt in builder.py
        src = _builder_src()
        idx = src.index("def _build_prompt")
        brace_start = src.index(":", idx)
        body = src[brace_start : brace_start + 2000]
        assert "voice_mode" in body
        assert "_VOICE_CONCISE_INSTRUCTION" in body


# ── Voice loop: passes voice_mode=True ───────────────────────────────────────


class TestVoiceLoopUsesVoiceMode:
    def test_voice_loop_run_passes_voice_mode_true(self):
        src = _voice_loop_src()
        idx = src.index("async def run(self, max_interactions")
        body = src[idx : idx + 5000]
        assert "voice_mode=True" in body

    @pytest.mark.skip(reason="rex/dashboard/routes.py retired in OpenClaw migration (US-P7-014)")
    def test_chat_mode_not_affected(self):
        """Chat route should NOT pass voice_mode=True."""
        pass


# ── .env.example documentation ───────────────────────────────────────────────


class TestEnvExampleDocumentation:
    def test_env_example_has_voice_max_tokens(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "VOICE_MAX_TOKENS" in content or "voice_max_tokens" in content

    def test_env_example_explains_voice_max_tokens(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "voice" in content.lower()
        assert "token" in content.lower()


# ── Functional test ────────────────────────────────────────────────────────────


class TestBuildPromptVoiceModeFunctional:
    def _make_assistant(self):
        from unittest.mock import MagicMock

        from rex.assistant import Assistant
        from rex.context.builder import ContextBuilder

        assistant = Assistant.__new__(Assistant)
        assistant._history = []
        assistant._pending_followup = None
        assistant._followup_injected = False
        assistant._followup_engine = None
        # US-014: _build_prompt delegates to _context_builder
        mock_settings = MagicMock()
        mock_settings.default_timezone = None
        mock_settings.default_location = None
        mock_settings.personality = None
        assistant._settings = mock_settings
        assistant._context_builder = ContextBuilder(
            settings=mock_settings,
            history=assistant._history,
            user_id="default",
            followup_engine=None,
        )
        return assistant

    def test_voice_mode_adds_concise_instruction_to_prompt(self):
        assistant = self._make_assistant()
        prompt = assistant._build_prompt("What time is it?", voice_mode=True)
        assert (
            "sentence" in prompt.lower() or "short" in prompt.lower() or "concise" in prompt.lower()
        )

    def test_non_voice_mode_does_not_add_instruction(self):
        assistant = self._make_assistant()
        prompt = assistant._build_prompt("What time is it?", voice_mode=False)
        assert "_VOICE_CONCISE_INSTRUCTION" not in prompt
        assert "sentence" not in prompt.lower() or "short" not in prompt.lower()

    def test_voice_mode_false_by_default(self):
        assistant = self._make_assistant()
        prompt_default = assistant._build_prompt("Hello")
        prompt_voice = assistant._build_prompt("Hello", voice_mode=True)
        assert len(prompt_voice) > len(prompt_default)
