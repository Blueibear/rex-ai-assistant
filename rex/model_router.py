"""Model router — classify user messages into task categories for model selection.

Classification is heuristic-only (keyword/pattern matching) with no LLM call,
giving deterministic, zero-latency routing.
"""

from __future__ import annotations

import re
from enum import auto
from typing import Sequence

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


class TaskCategory(StrEnum):
    coding = auto()
    reasoning = auto()
    search = auto()
    vision = auto()
    fast = auto()
    default = auto()


# ---------------------------------------------------------------------------
# Keyword / pattern tables
# ---------------------------------------------------------------------------

_CODING_PATTERNS: Sequence[str] = [
    r"\bcode\b",
    r"\bwrite\s+(a\s+)?(function|class|script|program|module|test)\b",
    r"\bdebugg?\b",
    r"\brefactor\b",
    r"\bimplement\b",
    r"\bsyntax\s+error\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\btypescript\b",
    r"\bgolang?\b",
    r"\brust\b",
    r"\bsql\b",
    r"\bshell\s+script\b",
    r"\bbash\b",
    r"\bregex\b",
    r"\bapi\s+endpoint\b",
    r"\bunit\s+test\b",
    r"\bpytest\b",
    r"\bgit\s+(commit|push|pull|merge|rebase|branch)\b",
    r"\bdocker(file)?\b",
    r"\bkubernetes\b",
]

_REASONING_PATTERNS: Sequence[str] = [
    r"\banalyze\b",
    r"\bcompare\b",
    r"\bpros?\s+and\s+cons?\b",
    r"\btrade.?offs?\b",
    r"\bstep.by.step\b",
    r"\bexplain\s+(how|why|the\s+reason)\b",
    r"\bplan\b",
    r"\bstrateg(y|ize)\b",
    r"\bevaluat(e|ion)\b",
    r"\bbreak\s+(down|it\s+down)\b",
    r"\bcomplex\b",
    r"\bmulti.step\b",
    r"\bdiagnos(e|is)\b",
    r"\bdecid(e|ing)\b",
    r"\bweigh\s+(the\s+)?options?\b",
]

_SEARCH_PATTERNS: Sequence[str] = [
    r"\bsearch\s+(for|the\s+web|online|internet)\b",
    r"\blook\s+up\b",
    r"\bfind\s+(me\s+)?(information|info|news|articles?)\b",
    r"\bwhat('s|\s+is)\s+the\s+latest\b",
    r"\bcurrent\s+(news|events?|status)\b",
    r"\brecent(ly)?\b.*\b(happened|released|announced)\b",
    r"\bbrowse\b",
    r"\bweb\s+(search|results?)\b",
    r"\bgoogle\b",
    r"\bwikipedia\b",
]

_VISION_PATTERNS: Sequence[str] = [
    r"\bimage\b",
    r"\bphoto\b",
    r"\bpicture\b",
    r"\bscreenshot\b",
    r"\bdescribe\s+(this|what\s+you\s+see|the\s+image)\b",
    r"\bwhat('s|\s+is)\s+in\s+(the\s+)?(image|photo|picture)\b",
    r"\bvisual\b",
    r"\bocr\b",
    r"\bread\s+(the\s+)?text\s+(in|from)\b",
    r"\bdiagram\b",
    r"\bchart\b",
]

_FAST_PATTERNS: Sequence[str] = [
    r"^(hi|hello|hey|howdy|yo|sup)\b",
    r"^(thanks?|thank\s+you)\b",
    r"^(ok(ay)?|got\s+it|sounds?\s+good)\b",
    r"\bwhat\s+(time|day|date)\s+is\s+it\b",
    r"\bwhat('s|\s+is)\s+(today|the\s+time|the\s+date)\b",
    r"\bhow\s+(are\s+you|old|tall|far|long|many|much)\b",
    r"^\s*\d[\d\s\+\-\*\/\.\(\)]*\s*[=\?]?\s*$",  # arithmetic
    r"\byes\b",
    r"\bno\b",
    r"\bmaybe\b",
    r"\brepeat\s+(that|it|yourself)\b",
    r"\bwhat\s+did\s+you\s+say\b",
]


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


class ModelRouter:
    """Classifies a user message into a TaskCategory using keyword heuristics.

    The classifier is intentionally simple — no LLM calls, no network I/O,
    pure regex matching.  Matches are checked in priority order:
    vision > coding > search > reasoning > fast > default.
    """

    def classify(self, message: str) -> TaskCategory:
        """Return the TaskCategory that best fits *message*."""
        if _matches_any(message, _VISION_PATTERNS):
            return TaskCategory.vision
        if _matches_any(message, _CODING_PATTERNS):
            return TaskCategory.coding
        if _matches_any(message, _SEARCH_PATTERNS):
            return TaskCategory.search
        if _matches_any(message, _REASONING_PATTERNS):
            return TaskCategory.reasoning
        if _matches_any(message, _FAST_PATTERNS):
            return TaskCategory.fast
        return TaskCategory.default
