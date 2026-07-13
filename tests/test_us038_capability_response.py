"""Tests for US-038: 'What can you do?' dynamic capability response."""

from __future__ import annotations

import pytest

from rex.capabilities.registry import (
    Capability,
    CapabilityRegistry,
    reset_capability_registry,
)
from rex.capabilities.responder import build_capability_response, is_capability_query

# ---------------------------------------------------------------------------
# Intent detection tests
# ---------------------------------------------------------------------------


class TestIsCapabilityQuery:
    """Test that intent phrases are recognised correctly."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "What can you do?",
            "what can you do",
            "WHAT CAN YOU DO?",
            "What are your capabilities?",
            "what are your capabilities",
            "What do you support?",
            "list capabilities",
            "list your capabilities",
            "list features",
            "list your features",
            "What features do you have?",
            "what can rex do",
            "show me your capabilities",
            "show your features",
        ],
    )
    def test_recognised_intents(self, phrase: str) -> None:
        assert is_capability_query(phrase), f"Expected True for: {phrase!r}"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Turn on the lights",
            "What is the weather?",
            "Play some jazz",
            "remind me at 8am",
            "hello",
            "what time is it",
        ],
    )
    def test_non_capability_phrases(self, phrase: str) -> None:
        assert not is_capability_query(phrase), f"Expected False for: {phrase!r}"


# ---------------------------------------------------------------------------
# Response builder tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure each test starts with a clean global registry."""
    reset_capability_registry()
    yield
    reset_capability_registry()


class TestBuildCapabilityResponse:
    """Test the response formatter."""

    def _make_registry(self, capabilities: list[Capability]) -> CapabilityRegistry:
        reg = CapabilityRegistry()
        for cap in capabilities:
            reg.register(cap)
        return reg

    def test_no_integrations_fallback(self) -> None:
        """When only 'chat' is enabled, return the fallback message."""
        reg = self._make_registry(
            [
                Capability(name="chat", description="Converse", enabled=True, category="General"),
                Capability(
                    name="home_assistant",
                    description="Control HA",
                    enabled=False,
                    category="Home",
                ),
            ]
        )
        response = build_capability_response(reg)
        assert response == "I can chat with you, but no integrations are set up yet."

    def test_empty_registry_fallback(self) -> None:
        """Completely empty registry returns the fallback message."""
        reg = CapabilityRegistry()
        response = build_capability_response(reg)
        assert response == "I can chat with you, but no integrations are set up yet."

    def test_integrations_listed(self) -> None:
        """Enabled integrations appear in the response."""
        reg = self._make_registry(
            [
                Capability(name="chat", description="Converse", enabled=True, category="General"),
                Capability(
                    name="home_assistant",
                    description="Control Home Assistant devices",
                    enabled=True,
                    category="Home",
                ),
                Capability(
                    name="send_email",
                    description="Compose and send email",
                    enabled=True,
                    category="Communication",
                ),
            ]
        )
        response = build_capability_response(reg)
        assert "Control Home Assistant devices" in response
        assert "Compose and send email" in response

    def test_grouped_by_category(self) -> None:
        """Response groups capabilities under category headings."""
        reg = self._make_registry(
            [
                Capability(name="chat", description="Converse", enabled=True, category="General"),
                Capability(
                    name="home_assistant",
                    description="Control HA",
                    enabled=True,
                    category="Home",
                ),
                Capability(
                    name="send_email",
                    description="Send email",
                    enabled=True,
                    category="Communication",
                ),
            ]
        )
        response = build_capability_response(reg)
        assert "Home:" in response
        assert "Communication:" in response
        assert "General:" in response

    def test_disabled_capabilities_excluded(self) -> None:
        """Disabled capabilities must not appear in the response."""
        reg = self._make_registry(
            [
                Capability(name="chat", description="Converse", enabled=True, category="General"),
                Capability(
                    name="home_assistant",
                    description="Control HA",
                    enabled=True,
                    category="Home",
                ),
                Capability(
                    name="send_email",
                    description="Send email",
                    enabled=False,
                    category="Communication",
                ),
            ]
        )
        response = build_capability_response(reg)
        assert "Send email" not in response

    def test_response_reflects_enabled_capabilities(self) -> None:
        """Response dynamically reflects which capabilities are enabled."""
        reg = self._make_registry(
            [
                Capability(name="chat", description="Chat", enabled=True, category="General"),
                Capability(
                    name="web_search",
                    description="Search the web",
                    enabled=False,
                    category="Search",
                ),
            ]
        )
        # Initially only chat — fallback
        assert build_capability_response(reg) == (
            "I can chat with you, but no integrations are set up yet."
        )

        # Enable web_search and regenerate
        cap = reg.get("web_search")
        assert cap is not None
        cap.enabled = True
        response2 = build_capability_response(reg)
        assert "Search the web" in response2

    def test_general_category_listed_first(self) -> None:
        """General category appears before other categories."""
        reg = self._make_registry(
            [
                Capability(name="chat", description="Converse", enabled=True, category="General"),
                Capability(
                    name="home_assistant",
                    description="Control HA",
                    enabled=True,
                    category="Home",
                ),
            ]
        )
        response = build_capability_response(reg)
        general_pos = response.index("General:")
        home_pos = response.index("Home:")
        assert general_pos < home_pos


# ---------------------------------------------------------------------------
# Integration: assistant.generate_reply routes capability queries
# ---------------------------------------------------------------------------


class TestAssistantCapabilityIntegration:
    """Smoke-test that generate_reply intercepts capability queries."""

    def test_generate_reply_intercepts_capability_query(self, monkeypatch) -> None:
        """generate_reply should return capability response without calling LLM."""
        import asyncio

        from rex.assistant import Assistant

        # Patch LLM so it raises if called — it must NOT be called
        class _FakeLLM:
            model_name = "fake"

            def generate(self, *a, **kw):
                raise AssertionError("LLM should not be called for capability queries")

            def stream(self, *a, **kw):
                raise NotImplementedError

        assistant = Assistant.__new__(Assistant)
        assistant._settings = type(
            "S",
            (),
            {
                "ha_base_url": None,
                "ha_token": None,
                "max_memory_items": 20,
                "model_routing": None,
                "ollama_base_url": "http://localhost:11434",
                "persist_history": False,
                "response_cache_ttl": 0,
                "skills_path": None,
                "shopping_list_path": None,
                "music_assistant_url": None,
                "music_assistant_token": None,
                "transcripts_dir": "/tmp/rex_test_transcripts",
                "followups_enabled": False,
            },
        )()
        assistant._llm = _FakeLLM()
        assistant._user_id = "default"
        assistant._history = []
        assistant._history_limit = 20
        assistant._history_store = None
        assistant._prune_timer = None
        assistant._followup_engine = None
        assistant._pending_followup = None
        assistant._ha_bridge = None
        assistant._response_cache = None
        assistant._suggestion_engine = None
        assistant._pattern_entries = {}
        assistant._user_id = "default"
        assistant._plugins = []
        assistant._transcripts_dir = __import__("pathlib").Path("/tmp/rex_test_transcripts")

        # Dummy handlers that return None (not handled)
        assistant._shopping_list_handler = None
        assistant._music_handler = None
        assistant._device_state_handler = None
        assistant._tool_dispatcher = None
        assistant._skill_trainer = None
        assistant._skill_registry = None
        assistant._skill_router = None

        from rex.openclaw.tool_bridge import ToolBridge

        assistant._tool_router_fn = ToolBridge().route_if_tool_request

        from rex.capabilities.registry import CapabilityRegistry

        # Use a fresh registry with only 'chat' enabled
        fresh_reg = CapabilityRegistry()
        from rex.capabilities.registry import Capability

        fresh_reg.register(
            Capability(name="chat", description="Chat", enabled=True, category="General")
        )

        monkeypatch.setattr(
            "rex.capabilities.registry.get_capability_registry",
            lambda config=None: fresh_reg,
        )

        async def _run():
            return await assistant.generate_reply("What can you do?")

        reply = asyncio.run(_run())
        assert "I can chat with you" in reply
        assert "no integrations" in reply
