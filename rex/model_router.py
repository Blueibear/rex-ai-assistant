"""Model router — classify user messages into task categories for model selection.

Classification is heuristic-only (keyword/pattern matching) with no LLM call,
giving deterministic, zero-latency routing.

Ollama availability is checked at init time (if needed) and refreshed every
60 seconds via a daemon background thread.  OpenAI and local-Transformers
models are assumed always available — no network call is made for them.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from enum import StrEnum, auto

logger = logging.getLogger(__name__)


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

# ---------------------------------------------------------------------------
# Ollama model detection
# ---------------------------------------------------------------------------

# Model ID prefixes that identify non-Ollama providers.
_OPENAI_PREFIXES = (
    "gpt-",
    "text-",
    "o1",
    "o3",
    "babbage",
    "davinci",
    "curie",
    "ada",
    "claude-",
    "gemini-",
    "mistral-",  # Mistral cloud API (vs local Ollama "mistral")
)

_ROUTING_FIELDS = ("default", "coding", "reasoning", "search", "vision", "fast")


def _is_ollama_model(model_id: str) -> bool:
    """Return True if *model_id* looks like an Ollama model (not OpenAI / HuggingFace)."""
    if not model_id:
        return False
    lower = model_id.lower()
    # HuggingFace / local Transformers paths contain a slash (org/model-name)
    if "/" in model_id:
        return False
    for prefix in _OPENAI_PREFIXES:
        if lower.startswith(prefix):
            return False
    return True


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class ModelRouter:
    """Classifies user messages and resolves target model with Ollama availability check.

    Classification uses keyword heuristics (no LLM call).  Priority order:
    vision > coding > search > reasoning > fast > default.

    Ollama availability is probed at init time only when at least one routing
    target is an Ollama model.  OpenAI and HuggingFace models skip the check.
    The available-model list is refreshed every *refresh_interval* seconds via
    a daemon thread.
    """

    _DEFAULT_OLLAMA_URL = "http://localhost:11434"
    _DEFAULT_REFRESH_INTERVAL = 60

    _DEFAULT_COOLDOWN_SECONDS = 3600  # 1 hour

    def __init__(
        self,
        ollama_base_url: str = _DEFAULT_OLLAMA_URL,
        routing_config: object = None,
        refresh_interval: int = _DEFAULT_REFRESH_INTERVAL,
        cooldown_seconds: int = _DEFAULT_COOLDOWN_SECONDS,
        notify_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._refresh_interval = refresh_interval
        self._available_ollama_models: set[str] = set()
        self._stop_event = threading.Event()
        self._refresh_thread: threading.Thread | None = None

        # Cloud rate-limit cooldown state
        self._cooldown_seconds = cooldown_seconds
        self._cloud_limited_until: float = 0.0  # epoch seconds
        self._notify_callback = notify_callback

        # Determine whether any routing target could be an Ollama model.
        self._needs_ollama = self._any_ollama_target(routing_config)

        if self._needs_ollama:
            self._fetch_ollama_models()
            self._start_refresh_thread()

    # ------------------------------------------------------------------
    # Ollama availability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _any_ollama_target(routing_config: object) -> bool:
        if routing_config is None:
            return False
        return any(_is_ollama_model(getattr(routing_config, f, "") or "") for f in _ROUTING_FIELDS)

    def _fetch_ollama_models(self) -> None:
        """Fetch available Ollama models from ``/api/tags`` and update the cache."""
        url = f"{self._ollama_base_url}/api/tags"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read())
            entries: list[dict[str, str]] = data.get("models", [])
            names: set[str] = set()
            for entry in entries:
                name = entry.get("name", "")
                if name:
                    names.add(name)
                    # Also store the name without the tag (e.g. "llama3" from "llama3:latest")
                    names.add(name.split(":")[0])
            self._available_ollama_models = names
            logger.debug("model_router: Ollama models available: %s", sorted(names))
        except Exception as exc:
            logger.warning("model_router: failed to fetch Ollama model list from %s: %s", url, exc)
            self._available_ollama_models = set()

    def _start_refresh_thread(self) -> None:
        def _loop() -> None:
            while not self._stop_event.wait(self._refresh_interval):
                self._fetch_ollama_models()

        t = threading.Thread(target=_loop, daemon=True, name="ModelRouter-ollama-refresh")
        t.start()
        self._refresh_thread = t

    def _is_available(self, model_id: str) -> bool:
        """Return True if *model_id* is usable.

        Non-Ollama models (OpenAI, HuggingFace) are always considered available.
        Ollama models are checked against the cached ``/api/tags`` response.
        """
        if not model_id:
            return False
        if not _is_ollama_model(model_id):
            return True  # OpenAI / HuggingFace — always available
        name_no_tag = model_id.split(":")[0]
        return (
            model_id in self._available_ollama_models
            or name_no_tag in self._available_ollama_models
        )

    # ------------------------------------------------------------------
    # Cloud rate-limit / quota fallback (US-045)
    # ------------------------------------------------------------------

    def cloud_limit_hit(self, status_code: int) -> None:
        """Record that the cloud API returned *status_code* 429 or 402.

        Activates a cooldown period during which :py:meth:`route` will route
        to the local model instead of cloud.  Calls *notify_callback* (if
        supplied) with a human-readable message so callers can surface the
        notification to the user.
        """
        if status_code not in (429, 402):
            return
        self._cloud_limited_until = time.monotonic() + self._cooldown_seconds
        msg = (
            f"Cloud limit reached (HTTP {status_code}), switching to local model. "
            f"Cloud will be retried in {self._cooldown_seconds // 60} min."
        )
        logger.warning("model_router: %s", msg)
        if self._notify_callback is not None:
            self._notify_callback(msg)

    def _cloud_in_cooldown(self) -> bool:
        """Return True if cloud is currently under a rate-limit cooldown."""
        return time.monotonic() < self._cloud_limited_until

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def resolve_model(self, category: TaskCategory, routing_config: object) -> str:
        """Resolve the model identifier for *category* with availability fallback.

        Returns an empty string when no routing override is configured,
        meaning the caller should use the globally-configured LLM model.

        If the preferred model for *category* is unavailable (Ollama not
        running or model not pulled), falls back to ``routing_config.default``
        and logs a warning.
        """
        if routing_config is None:
            return ""

        model: str = getattr(routing_config, str(category), "") or ""
        default: str = getattr(routing_config, "default", "") or ""

        if model:
            if self._is_available(model):
                return model
            logger.warning(
                "model_router: model %r for category %r is not available in Ollama; "
                "falling back to default %r",
                model,
                str(category),
                default,
            )

        # Return default regardless of its availability (caller handles final fallback)
        return default

    # ------------------------------------------------------------------
    # Smart local/cloud routing (US-044)
    # ------------------------------------------------------------------

    _SIMPLE_TOKEN_THRESHOLD = 200

    @staticmethod
    def estimate_complexity(message: str, *, requires_tools: bool = False) -> str:
        """Return ``"simple"`` or ``"complex"`` for *message*.

        A query is **simple** when:
        - estimated token count < 200 (approximated as ``len(message.split()) * 1.3``)
        - no tools are required

        Everything else is **complex**.
        """
        estimated_tokens = len(message.split()) * 1.3
        if requires_tools or estimated_tokens >= ModelRouter._SIMPLE_TOKEN_THRESHOLD:
            return "complex"
        return "simple"

    def route(
        self,
        message: str,
        *,
        local_model: str,
        cloud_model: str,
        routing_mode: str = "local_preferred",
        requires_tools: bool = False,
    ) -> str:
        """Return the model identifier to use for *message*.

        Args:
            message: The user message to route.
            local_model: Ollama model identifier for local inference.
            cloud_model: Cloud provider model identifier (e.g. ``"gpt-4o"``).
            routing_mode: One of ``"local_preferred"``, ``"cloud_only"``,
                ``"local_only"``.  Invalid values fall back to
                ``"local_preferred"`` with a warning.
            requires_tools: Whether the query needs tool dispatch.

        Returns:
            Model identifier string.
        """
        valid_modes = {"local_preferred", "cloud_only", "local_only"}
        if routing_mode not in valid_modes:
            logger.warning(
                "model_router: unknown routing_mode %r; falling back to 'local_preferred'",
                routing_mode,
            )
            routing_mode = "local_preferred"

        cloud_available = cloud_model and not self._cloud_in_cooldown()

        if routing_mode == "cloud_only":
            if cloud_available:
                return cloud_model
            # Cloud is in cooldown — fall back to local with notification
            logger.warning(
                "model_router: cloud_only mode but cloud is in cooldown; "
                "falling back to local model %r",
                local_model,
            )
            return local_model or cloud_model

        if routing_mode == "local_only":
            if local_model and self._is_available(local_model):
                return local_model
            logger.warning(
                "model_router: local_only mode but local model %r is unavailable; "
                "falling back to cloud model %r",
                local_model,
                cloud_model,
            )
            return cloud_model

        # local_preferred
        complexity = self.estimate_complexity(message, requires_tools=requires_tools)
        local_available = bool(local_model) and self._is_available(local_model)

        if complexity == "simple":
            if local_available:
                return local_model
            logger.warning(
                "model_router: local model %r is unavailable; routing simple query to cloud",
                local_model,
            )
            return cloud_model
        else:
            # complex query → prefer cloud if configured and not in cooldown
            if cloud_available:
                return cloud_model
            if local_available:
                return local_model
            logger.warning(
                "model_router: no cloud model available and local model %r is unavailable",
                local_model,
            )
            return local_model or cloud_model
