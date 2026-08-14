"""Golden tests for US-110 ModelRouter 2.0 fast/deep routing."""

from types import SimpleNamespace

from rex.llm_client import LanguageModel
from rex.model_router import ModelRouter


def _routing(**overrides):
    values = {
        "default": "fast-local",
        "fast": "fast-local",
        "coding": "",
        "reasoning": "deep-local",
        "search": "",
        "vision": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _router(config=None, available=None):
    router = ModelRouter()
    router._available_ollama_models = set(available or ())
    return router


def test_simple_command_uses_fast_route_with_explicit_evidence():
    config = _routing()
    router = _router(config, {"fast-local", "deep-local"})

    decision = router.decide("Hello", routing_config=config, current_model="fast-local")

    assert decision.tier == "fast"
    assert decision.complexity == "simple"
    assert decision.confidence >= 0.9
    assert decision.model == "fast-local"
    assert decision.escalation_count == 0
    assert decision.evidence


def test_ambiguous_tool_choice_escalates_once_to_deep():
    config = _routing()
    router = _router(config, {"fast-local", "deep-local"})

    decision = router.decide(
        "Use whichever tool makes sense to handle that.",
        routing_config=config,
        current_model="fast-local",
    )

    assert decision.tier == "deep"
    assert decision.model == "deep-local"
    assert decision.confidence < 0.6
    assert decision.escalation_count == 1
    assert "tool_choice_low_confidence" in decision.evidence


def test_complex_reasoning_routes_directly_to_deep_model():
    config = _routing()
    router = _router(config, {"fast-local", "deep-local"})

    decision = router.decide(
        "Analyze the tradeoffs and plan a complex multi-step migration strategy.",
        routing_config=config,
        current_model="fast-local",
    )

    assert decision.tier == "deep"
    assert decision.complexity == "complex"
    assert decision.model == "deep-local"
    assert decision.escalation_count == 0
    assert "category_reasoning" in decision.evidence


def test_deep_provider_outage_falls_back_to_fast_without_retry_loop():
    config = _routing()
    router = _router(config, {"fast-local", "deep-local"})

    decision = router.decide(
        "Analyze the tradeoffs in this architecture.",
        routing_config=config,
        current_model="fast-local",
        deep_provider_available=False,
    )

    assert decision.tier == "fast"
    assert decision.model == "fast-local"
    assert decision.fallback_reason == "deep_provider_unavailable"
    assert decision.escalation_count <= 1


def test_unavailable_local_deep_model_falls_back_to_available_fast_model():
    config = _routing()
    router = _router(config, {"fast-local"})

    decision = router.decide(
        "Analyze a complex migration plan.",
        routing_config=config,
        current_model="fast-local",
    )

    assert decision.tier == "fast"
    assert decision.model == "fast-local"
    assert decision.fallback_reason == "deep_model_unavailable"
    assert "deep_model_unavailable" in decision.evidence


def test_route_metadata_is_privacy_safe_and_bounded():
    config = _routing()
    router = _router(config, {"fast-local", "deep-local"})
    private_marker = "PRIVATE-USER-CONTENT-MARKER"

    decision = router.decide(
        f"Analyze {private_marker} and compare the tradeoffs.",
        routing_config=config,
        current_model="fast-local",
    )
    metadata = decision.to_metadata()

    assert set(metadata) == {
        "category",
        "complexity",
        "confidence",
        "evidence",
        "route",
        "model",
        "escalation_count",
        "fallback_reason",
    }
    assert private_marker not in repr(metadata)
    assert all(len(code) <= 64 for code in metadata["evidence"])


def test_request_model_selection_is_context_local_across_workers():
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from rex.config import AppConfig

    model = LanguageModel(AppConfig(llm_provider="echo", llm_model="base-local"))

    token = model.set_request_model("deep-a")
    context_a = contextvars.copy_context()
    model.reset_request_model(token)
    token = model.set_request_model("deep-b")
    context_b = contextvars.copy_context()
    model.reset_request_model(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(context_a.run, model.generate, "hello")
        second = pool.submit(context_b.run, model.generate, "hello")
        results = {first.result(), second.result()}

    assert results == {"[deep-a] hello", "[deep-b] hello"}
    assert model.model_name == "base-local"
    assert model.strategy.model_name == "base-local"


def test_local_only_mode_never_silently_falls_back_to_cloud():
    router = ModelRouter()
    router._available_ollama_models = set()

    result = router.route(
        "Hello",
        local_model="missing-local",
        cloud_model="gpt-5",
        routing_mode="local_only",
    )

    assert result == "missing-local"


def test_turn_event_exposes_privacy_safe_model_route_metadata():
    import asyncio
    from unittest.mock import MagicMock

    from rex.actions.dispatcher import ActionResult
    from rex.assistant import Assistant
    from rex.intent.router import IntentResult
    from rex.response.builder import FinalResponse
    from rex.runtime.events import EventKind

    private_marker = "PRIVATE-TURN-CONTENT-MARKER"
    config = _routing()
    assistant = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(
        max_memory_items=50,
        persist_history=False,
        followups_enabled=False,
        model_routing=config,
        transcripts_enabled=False,
        llm_provider="ollama",
        llm_model="fast-local",
        llm_routing_mode="local_preferred",
        llm=None,
    )
    assistant._user_id = "james"
    assistant._histories = {}
    assistant._history_limit = 50
    assistant._plugins = []
    assistant._history_store = None
    assistant._followup_engine = None
    assistant._followup_sessions = set()
    assistant._followup_bootstrap_pending = False
    assistant._pending_followups = {}
    assistant._response_cache = None
    assistant._ha_bridge = None
    assistant._suggestion_engine = None
    assistant._pattern_entries = {}
    assistant._llm = MagicMock()
    assistant._llm.model_name = "fast-local"
    assistant._router = _router(config, {"fast-local", "deep-local"})

    intent_router = MagicMock()
    intent_router.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
    assistant._intent_router = intent_router
    assistant._context_builder = MagicMock()
    assistant._context_builder.build.return_value = SimpleNamespace(
        messages=[], prompt="prompt", system_prompt="system"
    )
    assistant._response_builder = MagicMock()
    assistant._response_builder.check_cache.return_value = None
    assistant._response_builder.build.return_value = FinalResponse(
        text="deep reply", tts_text="deep reply"
    )
    assistant._turn_events = []
    assistant._turn_event_observer = assistant._turn_events.append
    assistant._log_turn = MagicMock()

    async def dispatch(*_args, **_kwargs):
        return ActionResult(success=True, response="deep reply")

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch

    asyncio.run(
        assistant.generate_reply(
            f"Analyze {private_marker} and compare the tradeoffs.", active_user_id="james"
        )
    )

    route_event = next(
        event
        for event in assistant._turn_events
        if event.kind is EventKind.ROUTE_PROGRESS and event.details.get("stage") == "model_router"
    )
    assert route_event.details["route"] == "deep"
    assert route_event.details["complexity"] == "complex"
    assert route_event.details["confidence"] >= 0.9
    assert route_event.details["model"] == "deep-local"
    assert route_event.details["evidence"]
    assert private_marker not in repr(dict(route_event.details))
    assert assistant._llm.model_name == "fast-local"


def test_unmatched_ordinary_request_stays_fast_without_low_confidence_signal():
    config = _routing()
    router = _router(config, {"fast-local", "deep-local"})

    decision = router.decide(
        "Tell me something interesting about octopuses.",
        routing_config=config,
        current_model="fast-local",
    )

    assert decision.tier == "fast"
    assert decision.model == "fast-local"
    assert decision.confidence >= 0.6


def test_local_only_policy_refuses_override_when_active_provider_is_cloud():
    config = _routing(reasoning="gpt-deep")
    router = ModelRouter()

    decision = router.decide(
        "Analyze the tradeoffs.",
        routing_config=config,
        current_model="gpt-fast",
        routing_mode="local_only",
        provider="openai",
    )

    assert decision.model == "gpt-fast"
    assert decision.tier == "fast"
    assert decision.fallback_reason == "local_only_provider_conflict"


def test_context_cache_uses_request_scoped_routed_model() -> None:
    import asyncio
    from unittest.mock import MagicMock

    from rex.actions.dispatcher import ActionResult
    from rex.assistant import Assistant
    from rex.config import AppConfig
    from rex.intent.router import IntentResult
    from rex.response.builder import FinalResponse

    config = _routing()
    assistant = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(
        max_memory_items=50,
        persist_history=False,
        followups_enabled=False,
        model_routing=config,
        transcripts_enabled=False,
        llm_provider="echo",
        llm_model="fast-local",
        llm_routing_mode="local_preferred",
        llm=None,
    )
    assistant._user_id = "james"
    assistant._histories = {}
    assistant._history_limit = 50
    assistant._plugins = []
    assistant._history_store = None
    assistant._followup_engine = None
    assistant._followup_sessions = set()
    assistant._followup_bootstrap_pending = False
    assistant._pending_followups = {}
    assistant._response_cache = None
    assistant._ha_bridge = None
    assistant._suggestion_engine = None
    assistant._pattern_entries = {}
    assistant._llm = LanguageModel(AppConfig(llm_provider="echo", llm_model="fast-local"))
    assistant._router = _router(config, {"fast-local", "deep-local"})

    intent_router = MagicMock()
    intent_router.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
    assistant._intent_router = intent_router
    assistant._context_builder = MagicMock()
    assistant._context_builder.build.return_value = SimpleNamespace(
        messages=[], prompt="prompt", system_prompt="system"
    )
    assistant._response_builder = MagicMock()
    assistant._response_builder.check_cache.return_value = None
    assistant._response_builder.build.return_value = FinalResponse(
        text="deep reply", tts_text="deep reply"
    )
    assistant._turn_events = []
    assistant._turn_event_observer = assistant._turn_events.append
    assistant._log_turn = MagicMock()

    async def dispatch(*_args, **_kwargs):
        return ActionResult(success=True, response="deep reply")

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch

    asyncio.run(
        assistant.generate_reply(
            "Analyze the tradeoffs in this complex migration.", active_user_id="james"
        )
    )

    cache_request = assistant._context_builder.build.call_args.kwargs["cache_request"]
    assert cache_request.model_provider == "echo"
    assert cache_request.model_name == "deep-local"
    assert assistant._llm.model_name == "fast-local"
