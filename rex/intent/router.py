"""Intent routing for direct-reply shortcuts (US-015).

Handles time/date queries, greeting detection, and recipe shortcuts
without an LLM round trip.  Logic moved from ``rex.assistant``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex constants (moved from rex.assistant)
# ---------------------------------------------------------------------------

_DIRECT_TIME_PATTERNS = (
    re.compile(r"\bwhat\s+time\s+is\s+it\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is)\s+(?:the\s+)?time\b", re.IGNORECASE),
    re.compile(r"\bcurrent\s+(?:local\s+)?time\b", re.IGNORECASE),
    re.compile(r"\btime\s+(?:is\s+it\s+)?(?:now|currently)\b", re.IGNORECASE),
)
_DIRECT_DATE_PATTERNS = (
    re.compile(r"\bwhat(?:'s|s| is)\s+(?:today'?s\s+|todays\s+|the\s+)?date\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+date\s+is\s+(?:it\s+)?(?:today)?\b", re.IGNORECASE),
    re.compile(r"\bcurrent\s+date\b", re.IGNORECASE),
    re.compile(r"\bdate\s+today\b", re.IGNORECASE),
    re.compile(r"\b(?:today'?s|todays)\s+date\b", re.IGNORECASE),
)
_DIRECT_DAY_PATTERNS = (
    re.compile(r"\bwhat\s+day\s+is\s+it(?:\s+today)?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+day\s+is\s+today\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s|s| is)\s+(?:the\s+)?day(?:\s+today)?\b", re.IGNORECASE),
    re.compile(r"\bday\s+of\s+(?:the\s+)?week\b", re.IGNORECASE),
)
_TIME_LOCATION_PATTERN = re.compile(r"\bin\s+([^?.!]+)[?.!]*\s*$", re.IGNORECASE)
_TIME_LOCATION_SUFFIXES = (
    "right now",
    "now",
    "today",
    "currently",
    "please",
    "for me",
    "at the moment",
)
_DIRECT_GREETING_PATTERN = re.compile(r"^\s*(?:hello|hey)\s*[!.?]*\s*$", re.IGNORECASE)
_DIRECT_WELLBEING_PATTERN = re.compile(
    r"^\s*(?:how\s+are\s+you|how'?s\s+it\s+going|how\s+are\s+things)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_DIRECT_CREATOR_PATTERN = re.compile(
    r"^\s*who\s+(?:created|made|built)\s+you\s*[?.!]*\s*$",
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


# ---------------------------------------------------------------------------
# IntentResult
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    """Result of an intent routing check."""

    handled: bool
    response: str | None
    intent_type: str | None


# ---------------------------------------------------------------------------
# IntentRouter
# ---------------------------------------------------------------------------


class IntentRouter:
    """Routes common user intents to direct responses, bypassing the LLM.

    Args:
        tool_context_fn: Optional callable returning a ``dict`` with
                         ``"location"`` and ``"timezone"`` keys, used for
                         time/date queries.  When *None*, time shortcuts
                         still work but use the local clock without location.
    """

    def __init__(
        self,
        *,
        tool_context_fn: Callable[[], dict] | None = None,
    ) -> None:
        self._tool_context_fn = tool_context_fn

    def route(self, user_message: str, context: Any = None) -> IntentResult:
        """Check *user_message* for recognized shortcut intents.

        Returns an :class:`IntentResult` with ``handled=True`` and the
        pre-built ``response`` when an intent is recognized, or
        ``handled=False`` when the message should proceed to the LLM.

        Args:
            user_message: The user's raw message.
            context:      Optional :class:`~rex.context.builder.ContextPackage`
                          (reserved for future use; not currently consumed).
        """
        text = user_message.strip()
        if not text:
            return IntentResult(handled=False, response=None, intent_type=None)

        resp = self._try_time(text)
        if resp is not None:
            return IntentResult(handled=True, response=resp, intent_type="time_query")

        resp = self._try_greeting(text)
        if resp is not None:
            return IntentResult(handled=True, response=resp, intent_type="greeting")

        resp = self._try_recipe(text)
        if resp is not None:
            return IntentResult(handled=True, response=resp, intent_type="recipe")

        return IntentResult(handled=False, response=None, intent_type=None)

    # ------------------------------------------------------------------
    # Private helpers — time / date
    # ------------------------------------------------------------------

    def _try_time(self, text: str) -> str | None:
        """Answer simple clock/date queries without an LLM round trip."""
        wants_time = any(pattern.search(text) for pattern in _DIRECT_TIME_PATTERNS)
        wants_date = any(pattern.search(text) for pattern in _DIRECT_DATE_PATTERNS)
        wants_day = any(pattern.search(text) for pattern in _DIRECT_DAY_PATTERNS)
        if not wants_time and not wants_date and not wants_day:
            return None

        context = self._tool_context_fn() if self._tool_context_fn is not None else {}
        location = self._extract_time_location(text) or context.get("location")
        args = {"location": location} if location else {}

        try:
            from rex.openclaw.tool_executor import execute_tool

            result = execute_tool(
                {"tool": "time_now", "args": args},
                context,
                skip_policy_check=True,
                skip_credential_check=True,
                skip_audit_log=True,
            )
        except Exception as exc:
            logger.debug("direct time reply failed: %s", exc)
            return None

        if "error" in result:
            fallback = self._fallback_local_time(location, context)
            if fallback is None:
                logger.debug("direct time reply returned error: %s", result["error"])
                return None
            result = fallback

        return self._format_time_reply(
            result,
            location=location,
            wants_date=wants_date and not wants_time,
            wants_day=wants_day and not wants_time,
        )

    def _fallback_local_time(
        self,
        location: str | None,
        context: dict,
    ) -> dict[str, object] | None:
        configured_location = context.get("location")
        if location and configured_location:
            same_location = location.strip().casefold() == configured_location.strip().casefold()
            if not same_location:
                return None

        try:
            now = datetime.now().astimezone()
        except Exception as exc:
            logger.debug("local clock fallback failed: %s", exc)
            return None

        timezone = str(now.tzinfo) if now.tzinfo is not None else context.get("timezone", "local")
        return {
            "local_time": now.strftime("%Y-%m-%d %H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "timezone": timezone,
        }

    def _extract_time_location(self, transcript: str) -> str | None:
        match = _TIME_LOCATION_PATTERN.search(transcript)
        if not match:
            return None
        location = match.group(1).strip(" \t,")
        while True:
            updated = location
            for suffix in _TIME_LOCATION_SUFFIXES:
                suffix_text = f" {suffix}"
                if updated.lower().endswith(suffix_text):
                    updated = updated[: -len(suffix_text)].strip(" \t,")
                    break
            if updated == location:
                break
            location = updated
        if not location:
            return None
        if location.lower().split(maxsplit=1)[0] in {"my", "your", "the", "a", "an"}:
            return None
        return location

    def _format_time_reply(
        self,
        result: dict[str, object],
        *,
        location: str | None,
        wants_date: bool,
        wants_day: bool,
    ) -> str:
        local_time = str(result.get("local_time") or "")
        try:
            when = datetime.strptime(local_time, "%Y-%m-%d %H:%M")
        except ValueError:
            return str(result.get("local_time") or result.get("date") or "")

        place = f" in {location}" if location else ""
        if wants_day:
            date_text = f"{when.strftime('%B')} {when.day}, {when.year}"
            return f"Today is {when.strftime('%A')}, {date_text}{place}."

        if wants_date:
            date_text = f"{when.strftime('%B')} {when.day}, {when.year}"
            return f"Today is {date_text}{place}."

        time_text = when.strftime("%I:%M %p").lstrip("0")
        return f"It's {time_text}{place}."

    # ------------------------------------------------------------------
    # Private helpers — greetings
    # ------------------------------------------------------------------

    def _try_greeting(self, text: str) -> str | None:
        """Handle common greetings without invoking an unstable chat model."""
        if _DIRECT_GREETING_PATTERN.match(text):
            return "Hello. How can I help?"
        if _DIRECT_WELLBEING_PATTERN.match(text):
            return "I'm here and ready to help."
        if _DIRECT_CREATOR_PATTERN.match(text):
            return (
                "I'm AskRex, a local assistant running from this project. "
                "The repo owner and project contributors configure the models and integrations I use."
            )
        return None

    # ------------------------------------------------------------------
    # Private helpers — recipes
    # ------------------------------------------------------------------

    def _try_recipe(self, text: str) -> str | None:
        """Handle common recipe requests without tool or shopping-list routing."""
        if _SHOPPING_LIST_REFERENCE_PATTERN.search(text):
            return None
        if not _RECIPE_REQUEST_PATTERN.search(text):
            return None
        if not _CHOCOLATE_CAKE_PATTERN.search(text):
            return None
        return _CHOCOLATE_CAKE_RECIPE
