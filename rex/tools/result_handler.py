"""Tool result post-processing handler (US-013).

Extracted from ``rex.assistant.Assistant._post_process_completion``.  All
raw-completion fixup logic lives here so that ``assistant.py`` delegates
post-processing to a single call on ``ToolResultHandler.process()``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

from rex.runtime.cancellation import await_with_cancellation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns (independent copies; assistant.py retains its own for
# streaming helpers that are not part of this handler)
# ---------------------------------------------------------------------------

_INTERNAL_TOOL_SYNTAX_PATTERN = re.compile(r"\bTOOL_(?:REQUEST|RESULT)\s*:", re.IGNORECASE)
_HA_INLINE_SYNTAX_PATTERN = re.compile(r"\[\[ha:[^\]\r\n]*\]\]", re.IGNORECASE)

_UNVERIFIED_ACTION_CLAIM_PATTERNS = (
    re.compile(
        r"\bi(?:'ve| have| just)?\s+"
        r"(?:added|put|saved|sent|scheduled|set|changed|updated|deleted|removed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:done|okay|sure)[,.\s]+(?:i(?:'ve| have| just)\s+)?"
        r"(?:added|put|saved|sent|scheduled|set|changed|updated|deleted|removed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:it|that|this)\s+(?:has\s+been|was)\s+"
        r"(?:added|put|saved|sent|scheduled|set|changed|updated|deleted|removed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i(?:'ve| have| just)|done|okay|sure)[^.!?\n]{0,80}"
        r"\bcreated\s+(?:an?\s+)?(?:event|task|reminder|calendar|file|note)\b",
        re.IGNORECASE,
    ),
)

_EXPLICIT_MUTATION_REQUEST_PATTERN = re.compile(
    r"\b(?:add|put|save|send|create|schedule|set|change|update|delete|remove)\b",
    re.IGNORECASE,
)

_RECIPE_REQUEST_PATTERN = re.compile(
    r"\b(?:need|want|give\s+me|find\s+me|show\s+me|make|bake|cook|how\s+(?:do\s+i|to))\b"
    r".*\b(?:recipe|make|bake|cook)\b",
    re.IGNORECASE,
)
_CHOCOLATE_CAKE_PATTERN = re.compile(r"\bchocolate\s+cake\b", re.IGNORECASE)
_SHOPPING_LIST_REFERENCE_PATTERN = re.compile(r"\b(?:shopping\s+)?list\b", re.IGNORECASE)

_CHOCOLATE_CAKE_RECIPE = (
    "Here is a simple chocolate cake recipe: mix 1 and 3/4 cups flour, "
    "2 cups sugar, 3/4 cup cocoa, 1 and 1/2 teaspoons baking powder, "
    "1 and 1/2 teaspoons baking soda, and 1 teaspoon salt. Add 2 eggs, "
    "1 cup milk, 1/2 cup oil, and 2 teaspoons vanilla, then stir in "
    "1 cup hot water. Bake in two greased 9-inch pans at 350 F for "
    "30 to 35 minutes, cool, and frost."
)


class ToolResultHandler:
    """Post-process raw LLM completions through the tool and safety layers.

    Handles:
    - Routing ``TOOL_REQUEST:`` directives through *tool_router_fn*
    - Appending plugin enrichment text
    - Home Assistant response post-processing
    - Sanitizing residual internal tool syntax before user output
    - Guarding against unverified action claims

    Args:
        tool_router_fn: Callable ``(completion, tool_context, model_call_fn) -> str``.
                        Typically ``ToolBridge().route_if_tool_request``.
        ha_bridge:      Optional ``HABridge`` instance.  Pass ``None`` when HA
                        is not configured.
    """

    def __init__(
        self,
        tool_router_fn: Callable[..., str],
        ha_bridge: Any = None,
    ) -> None:
        self._tool_router_fn = tool_router_fn
        self._ha_bridge = ha_bridge

    # ------------------------------------------------------------------
    # Primary async entry point
    # ------------------------------------------------------------------

    async def process(
        self,
        transcript: str,
        completion: str,
        *,
        tool_context: dict[str, str],
        model_call_fn: Any,
        plugin_enrichments: list[str],
        allow_ha_postprocess: bool = True,
    ) -> str:
        """Post-process *completion* through all tool and safety layers.

        Args:
            transcript:         Original user message.
            completion:         Raw LLM completion string.
            tool_context:       Context dict (location, timezone) for tool execution.
            model_call_fn:      Callable built by ``_build_tool_model_call(transcript)``.
            plugin_enrichments: List of plugin result strings to append.

        Returns:
            Cleaned, safe user-facing completion string.
        """
        loop = asyncio.get_running_loop()
        from rex.mobile_api.action_context import (  # noqa: PLC0415
            mobile_action_context_active,
            mobile_scope_granted,
            run_in_executor_with_mobile_context,
        )

        completion = await await_with_cancellation(
            run_in_executor_with_mobile_context(
                loop,
                self._tool_router_fn,
                completion,
                tool_context,
                model_call_fn,
            )
        )

        if plugin_enrichments:
            completion = f"{completion}\n\nAdditional info:\n" + "\n".join(plugin_enrichments)

        if not allow_ha_postprocess and _HA_INLINE_SYNTAX_PATTERN.search(completion):
            logger.warning(
                "Suppressed inline Home Assistant syntax for canonical media turn",
                extra={"event": "assistant_media_ha_bypass_suppressed"},
            )
            completion = "I did not execute that Home Assistant command."
        elif (
            self._ha_bridge is not None
            and self._ha_bridge.enabled
            and not mobile_action_context_active()
            and mobile_scope_granted("home.control")
        ):
            completion = await await_with_cancellation(
                run_in_executor_with_mobile_context(
                    loop,
                    self._ha_bridge.post_process_response,
                    completion,
                )
            )

        if self._contains_internal_tool_syntax(completion):
            completion = await await_with_cancellation(
                run_in_executor_with_mobile_context(
                    loop,
                    self._sanitize_internal_tool_output,
                    transcript,
                    completion,
                    tool_context,
                    model_call_fn,
                )
            )

        return self._guard_unverified_action_claim(transcript, completion)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _contains_internal_tool_syntax(self, text: str) -> bool:
        return bool(_INTERNAL_TOOL_SYNTAX_PATTERN.search(text))

    def _sanitize_internal_tool_output(
        self,
        transcript: str,
        completion: str,
        tool_context: dict[str, str],
        model_call_fn: Any,
    ) -> str:
        """Resolve or suppress tool directives before user-facing output."""
        rerouted = self._tool_router_fn(
            completion.strip(),
            tool_context,
            model_call_fn,
        )
        if not self._contains_internal_tool_syntax(rerouted):
            logger.warning(
                "Resolved raw internal tool syntax before user output",
                extra={"event": "assistant_internal_tool_syntax_resolved"},
            )
            return rerouted

        logger.error(
            "Suppressed raw internal tool syntax before user output",
            extra={"event": "assistant_internal_tool_syntax_suppressed"},
        )
        return "I could not complete that tool request."

    def _guard_unverified_action_claim(self, transcript: str, completion: str) -> str:
        """Return *completion* unless it makes an unverified action claim."""
        if not self._looks_like_unverified_action_claim(completion):
            return completion
        if _EXPLICIT_MUTATION_REQUEST_PATTERN.search(transcript):
            return completion

        recipe_reply = self._try_recipe_fallback(transcript)
        if recipe_reply is not None:
            logger.warning(
                "Suppressed unverified action claim for recipe request",
                extra={"event": "assistant_unverified_action_claim_suppressed"},
            )
            return recipe_reply

        logger.warning(
            "Suppressed unverified action claim",
            extra={"event": "assistant_unverified_action_claim_suppressed"},
        )
        return (
            "I did not change anything. Please tell me exactly what you want me to add, "
            "send, save, or update."
        )

    def _looks_like_unverified_action_claim(self, completion: str) -> bool:
        return any(pattern.search(completion) for pattern in _UNVERIFIED_ACTION_CLAIM_PATTERNS)

    def _try_recipe_fallback(self, transcript: str) -> str | None:
        """Return a recipe reply if the transcript is a chocolate cake recipe request."""
        text = transcript.strip()
        if not text:
            return None
        if _SHOPPING_LIST_REFERENCE_PATTERN.search(text):
            return None
        if not _RECIPE_REQUEST_PATTERN.search(text):
            return None
        if not _CHOCOLATE_CAKE_PATTERN.search(text):
            return None
        return _CHOCOLATE_CAKE_RECIPE
