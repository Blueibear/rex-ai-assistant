"""Action dispatch orchestration (US-016).

Extracted from ``rex.assistant.Assistant.generate_reply``.  Handles
skill invocation, HA command routing, auto tool dispatch, the LLM call,
and OpenClaw tool-bridge post-processing in one component so that
``generate_reply`` reads as a thin orchestration spec.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_UNDO_PATTERN = re.compile(r"^\s*undo\s*(?:that)?\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """Result returned by :class:`ActionDispatcher`."""

    success: bool
    response: str
    actions_taken: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# ActionDispatcher
# ---------------------------------------------------------------------------


class ActionDispatcher:
    """Route a user transcript through all action layers before returning a response.

    Handles (in order):

    1. Skill training detection (natural-language skill creation)
    2. Skill invocation (registered trigger matching)
    3. Shopping list voice commands
    4. Music Assistant voice commands
    5. Device state queries
    6. Auto tool dispatch (pre-LLM tool context building via ToolDispatcher)
    7. Home Assistant command routing (undo + process_transcript)
    8. LLM call (via ContextBuilder + LanguageModel)
    9. Post-LLM tool-request resolution (ToolResultHandler / OpenClaw bridge)

    Args:
        context_builder:      :class:`~rex.context.builder.ContextBuilder` instance.
        llm:                  :class:`~rex.llm_client.LanguageModel` instance.
        result_handler:       :class:`~rex.tools.result_handler.ToolResultHandler` instance.
        ha_bridge:            Optional :class:`~rex.ha_bridge.HABridge` instance.
        tool_dispatcher:      Optional :class:`~rex.tools.dispatcher.ToolDispatcher` instance.
        skill_trainer:        Optional :class:`~rex.skills.trainer.SkillTrainer` instance.
        skill_registry:       Optional :class:`~rex.skills.registry.SkillRegistry` instance.
        skill_router:         Optional :class:`~rex.skills.router.SkillRouter` instance.
        shopping_list_handler: Optional :class:`~rex.shopping_list_handler.ShoppingListHandler`.
        music_handler:        Optional :class:`~rex.music_handler.MusicHandler`.
        device_state_handler: Optional :class:`~rex.ha.state_handler.DeviceStateHandler`.
        suggestion_engine:    Optional proactive suggestion engine.
        pattern_entries:      Shared mutable list of PatternEntry objects for suggestion detection.
        build_tool_context_fn: Callable returning ``{"location": ..., "timezone": ...}`` dict.
        model_call_fn_builder: Callable ``(transcript) -> model_call_fn`` used for tool re-prompts.
        run_plugins_fn:       Async callable ``(transcript) -> list[str]`` for plugin enrichments.
    """

    def __init__(
        self,
        *,
        context_builder: Any,
        llm: Any,
        result_handler: Any,
        ha_bridge: Any = None,
        tool_dispatcher: Any = None,
        skill_trainer: Any = None,
        skill_registry: Any = None,
        skill_router: Any = None,
        shopping_list_handler: Any = None,
        music_handler: Any = None,
        device_state_handler: Any = None,
        suggestion_engine: Any = None,
        pattern_entries: list | None = None,
        build_tool_context_fn: Callable[[], dict] | None = None,
        model_call_fn_builder: Callable[[str], Any] | None = None,
        run_plugins_fn: Callable[..., Awaitable[list[str]]] | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._llm = llm
        self._result_handler = result_handler
        self._ha_bridge = ha_bridge
        self._tool_dispatcher = tool_dispatcher
        self._skill_trainer = skill_trainer
        self._skill_registry = skill_registry
        self._skill_router = skill_router
        self._shopping_list_handler = shopping_list_handler
        self._music_handler = music_handler
        self._device_state_handler = device_state_handler
        self._suggestion_engine = suggestion_engine
        self._pattern_entries: list = pattern_entries if pattern_entries is not None else []
        self._build_tool_context_fn = build_tool_context_fn
        self._model_call_fn_builder = model_call_fn_builder
        self._run_plugins_fn = run_plugins_fn

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        intent: Any,
        context: Any,
        transcript: str,
        *,
        voice_mode: bool = False,
        active_user_id: str | None = None,
        user_id: str = "default",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> ActionResult:
        """Dispatch *transcript* through all action layers and return an :class:`ActionResult`.

        Args:
            intent:         :class:`~rex.intent.router.IntentResult` from the intent router.
                            Passed for context; the caller already handled ``handled=True`` cases.
            context:        :class:`~rex.context.builder.ContextPackage` from the context builder.
                            Used as a fallback when no auto tool context is available.
            transcript:     The user's original message.  (The PRD spec refers to this
                            parameter as ``llm_response`` but it carries the *input* transcript
                            so that pre-LLM handlers can pattern-match against it.)
            voice_mode:     Append the voice-concise instruction to the LLM prompt.
            active_user_id: Per-request user override for multi-user scenarios.
            user_id:        Default session user ID.
            loop:           Running event loop; obtained via ``asyncio.get_running_loop()`` when
                            *None*.

        Returns:
            :class:`ActionResult` with ``success=True`` and the final response string.
        """
        _loop = loop or asyncio.get_running_loop()
        effective_user = active_user_id or user_id

        # 1. Skill training: intercept natural-language skill creation before LLM
        if self._skill_trainer is not None and self._skill_registry is not None:
            training_response = self._skill_trainer.handle_if_training_request(
                transcript, self._skill_registry
            )
            if training_response is not None:
                return ActionResult(
                    success=True,
                    response=str(training_response),
                    actions_taken=["skill_training"],
                )

        # 2. Skill invocation: check registered skill triggers before the LLM
        if self._skill_router is not None:
            matched_skill = self._skill_router.match(transcript)
            if matched_skill is not None:
                skill_response = str(self._skill_router.execute(matched_skill, transcript))
                return ActionResult(
                    success=True,
                    response=skill_response,
                    actions_taken=["skill_invocation"],
                )

        # 3. Shopping list voice commands
        if self._shopping_list_handler is not None:
            _sl_response = self._shopping_list_handler.handle(transcript, user_id=effective_user)
            if _sl_response is not None:
                return ActionResult(
                    success=True,
                    response=str(_sl_response),
                    actions_taken=["shopping_list"],
                )

        # 4. Music Assistant voice commands
        if self._music_handler is not None:
            _music_response = self._music_handler.handle(transcript)
            if _music_response is not None:
                return ActionResult(
                    success=True,
                    response=str(_music_response),
                    actions_taken=["music"],
                )

        # 5. Device state queries
        if self._device_state_handler is not None:
            _ds_response = self._device_state_handler.handle(transcript)
            if _ds_response is not None:
                return ActionResult(
                    success=True,
                    response=str(_ds_response),
                    actions_taken=["device_state"],
                )

        # 6. Auto tool dispatch: build pre-LLM tool context string
        _tool_context: str | None = None
        if self._tool_dispatcher is not None:
            _selected_tools = self._tool_dispatcher.select_tools(transcript)
            if _selected_tools:
                _tool_results = await _loop.run_in_executor(
                    None,
                    functools.partial(
                        self._tool_dispatcher.execute_tools,
                        _selected_tools,
                        transcript,
                        user_id=effective_user,
                    ),
                )
                _tool_context = self._tool_dispatcher.format_tool_context(_tool_results) or None

        completion: str | None = None

        # 7. HA command routing (including undo and proactive suggestion injection)
        if self._ha_bridge is not None and self._ha_bridge.enabled:
            if _UNDO_PATTERN.match(transcript):
                completion = await _loop.run_in_executor(
                    None,
                    self._ha_bridge.undo_last,
                )
            else:
                _hist_len_before = len(getattr(self._ha_bridge, "_command_history", None) or [])
                completion = await _loop.run_in_executor(
                    None,
                    self._ha_bridge.process_transcript,
                    transcript,
                )
                if completion is not None:
                    _cmd_hist = getattr(self._ha_bridge, "_command_history", None)
                    if _cmd_hist is not None and len(_cmd_hist) > _hist_len_before:
                        try:
                            _last = _cmd_hist._entries[-1]
                            from rex.suggestions.pattern_detector import PatternEntry

                            self._pattern_entries.append(
                                PatternEntry(
                                    entity_id=_last.entity_id,
                                    service=_last.service,
                                    timestamp=time.time(),
                                )
                            )
                        except Exception:
                            pass
                    if self._suggestion_engine is not None:
                        try:
                            from rex.suggestions.pattern_detector import detect_patterns

                            _patterns = detect_patterns(self._pattern_entries)
                            _sug = self._suggestion_engine.get_suggestion(_patterns)
                            if _sug is not None:
                                _, _spoken = _sug
                                completion = f"{completion} {_spoken}"
                        except Exception:
                            pass

        # 8. LLM call (if no pre-LLM handler produced a completion)
        if completion is None:
            # Rebuild context with tool_context if auto-dispatch populated it.
            if _tool_context:
                ctx = self._context_builder.build(
                    transcript,
                    voice_mode=voice_mode,
                    active_user_id=active_user_id,
                    tool_context=_tool_context,
                )
            else:
                ctx = context

            messages = ctx.messages
            prompt = ctx.prompt
            try:
                completion = await _loop.run_in_executor(
                    None,
                    lambda: self._llm.generate(messages=messages),
                )
            except TypeError:
                completion = await _loop.run_in_executor(
                    None,
                    lambda: self._llm.generate(prompt),
                )

            # 9. Post-process LLM output (TOOL_REQUEST resolution, OpenClaw bridge)
            plugin_enrichments: list[str] = []
            if self._run_plugins_fn is not None:
                plugin_enrichments = await self._run_plugins_fn(transcript)

            tool_context_dict: dict = (
                self._build_tool_context_fn() if self._build_tool_context_fn else {}
            )
            model_call_fn = (
                self._model_call_fn_builder(transcript)
                if self._model_call_fn_builder is not None
                else None
            )
            completion = await self._result_handler.process(
                transcript,
                completion,
                tool_context=tool_context_dict,
                model_call_fn=model_call_fn,
                plugin_enrichments=plugin_enrichments,
            )

        return ActionResult(
            success=True,
            response=completion,
            actions_taken=["llm"] if not completion else ["dispatch"],
        )
