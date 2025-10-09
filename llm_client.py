"""Language model client with pluggable backends (Transformers, OpenAI, Echo)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from assistant_errors import ConfigurationError
from config import AppConfig, load_config
from logging_utils import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    import torch
except ImportError:  # pragma: no cover - dependency optional
    torch = None

try:  # pragma: no cover - dependency optional
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = hf_pipeline = None

try:  # pragma: no cover - dependency optional
    from openai import OpenAI
except ImportError:
    OpenAI = None

TORCH_AVAILABLE = torch is not None
TRANSFORMERS_AVAILABLE = all([AutoTokenizer, AutoModelForCausalLM, hf_pipeline])
OPENAI_AVAILABLE = OpenAI is not None


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int


class LLMStrategy(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        ...


# === Backend: Echo (fallback) ===
class EchoStrategy:
    name = "echo"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        _config: GenerationConfig,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        text = prompt.strip()
        if not text and messages:
            text = "\n".join(item.get("content", "").strip() for item in messages).strip()
        if not text:
            return "(silence)"
        return f"[{self.model_name}] {text}"


class OfflineTransformersStrategy:
    """Deterministic fallback when transformers dependencies are unavailable."""

    name = "offline-transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        _config: GenerationConfig,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        text = prompt.strip()
        if not text and messages:
            text = "\n".join(item.get("content", "").strip() for item in messages).strip()
        return f"(offline) {text}" if text else "(offline)"


# === Backend: Transformers ===
class TransformersStrategy:
    name = "transformers"

    def __init__(self, model_name: str) -> None:
        if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            raise ConfigurationError("Transformers backend requires `torch` and `transformers`.")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1

        self.pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if torch is None:  # pragma: no cover - defensive
            raise ConfigurationError("Transformers backend requires torch.")

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        do_sample = config.temperature > 0 or config.top_p < 1.0
        generate_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(config.temperature, 1e-4)
            generate_kwargs["top_p"] = config.top_p
            generate_kwargs["top_k"] = config.top_k
        else:
            generate_kwargs["temperature"] = 1.0
            generate_kwargs["top_p"] = 1.0
            generate_kwargs["top_k"] = 0

        outputs = self.pipeline(prompt, **generate_kwargs)
        text = outputs[0]["generated_text"]
        return text[len(prompt):].strip() or "(silence)"


class OpenAIStrategy:
    name = "openai"

    def __init__(self, model_name: str, client_factory) -> None:
        self.model_name = model_name
        self._client_factory = client_factory
        self._cached_client: Any | None = None

    def _get_client(self) -> Any:
        if self._cached_client is None:
            self._cached_client = self._client_factory()
        return self._cached_client

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        payload = messages or [{"role": "user", "content": prompt}]
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=payload,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
        )
        content = getattr(response.choices[0].message, "content", "") or ""
        return content.strip() or "(silence)"


_STRATEGIES: Dict[str, type[LLMStrategy]] = {
    EchoStrategy.name: EchoStrategy,
    TransformersStrategy.name: TransformersStrategy,
}


def register_strategy(name: str, strategy: type[LLMStrategy]) -> None:
    _STRATEGIES[name] = strategy


class LanguageModel:
    """LLM wrapper supporting multiple backends (OpenAI, Transformers, Echo)."""

    def __init__(self, config: Optional[AppConfig] = None, **overrides) -> None:
        self.config = config or load_config()
        self.provider = (overrides.get("provider") or self.config.llm_provider).lower()
        self.model_name = overrides.get("model") or self.config.llm_model
        self.api_key = (
            overrides.get("openai_api_key")
            or self.config.openai_api_key
            or os.getenv("OPENAI_API_KEY")
        )

        self.generation = GenerationConfig(
            max_new_tokens=overrides.get("max_new_tokens", self.config.llm_max_tokens),
            temperature=overrides.get("temperature", self.config.llm_temperature),
            top_p=overrides.get("top_p", self.config.llm_top_p),
            top_k=overrides.get("top_k", self.config.llm_top_k),
            seed=overrides.get("seed", self.config.llm_seed),
        )

        self._openai_client: Any | None = None
        self.strategy = self._init_strategy()

    def _init_strategy(self) -> LLMStrategy:
        if self.provider == "openai":
            return OpenAIStrategy(self.model_name, self._ensure_openai_client)

        strategy_cls = _STRATEGIES.get(self.provider)
        if strategy_cls is None:
            logger.warning("Unknown LLM provider '%s'. Falling back to echo mode.", self.provider)
            return EchoStrategy(self.model_name)

        if strategy_cls is TransformersStrategy and not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            logger.warning(
                "Transformers dependencies missing; using offline fallback for model '%s'.",
                self.model_name,
            )
            return OfflineTransformersStrategy(self.model_name)

        try:
            return strategy_cls(self.model_name)
        except Exception as exc:
            logger.warning("LLM backend init failed (%s). Falling back to echo. (%s)", self.provider, exc)
            return EchoStrategy(self.model_name)

    def _ensure_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        if not OPENAI_AVAILABLE:
            raise ConfigurationError("OpenAI backend requires the `openai` package.")
        if not self.api_key:
            raise ConfigurationError("Missing OpenAI API key.")
        self._openai_client = OpenAI(api_key=self.api_key)
        return self._openai_client

    def _format_messages(self, messages: Sequence[Dict[str, str]]) -> str:
        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user").strip() or "user"
            content = (message.get("content") or "").strip()
            parts.append(f"{role.capitalize()}: {content}".strip())
        return "\n".join(part for part in parts if part).strip()

    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if messages is not None:
            prompt_text = self._format_messages(messages)
            normalized_messages: Optional[List[Dict[str, str]]] = []
            for entry in messages:
                if isinstance(entry, dict):
                    normalized_messages.append(
                        {
                            "role": str(entry.get("role", "")),
                            "content": str(entry.get("content", "")),
                        }
                    )
            if not normalized_messages:
                normalized_messages = None
        elif isinstance(prompt, str):
            prompt_text = prompt
            normalized_messages = None
        else:
            raise ValueError("Prompt or messages must be provided.")

        if not prompt_text or not prompt_text.strip():
            raise ValueError("Prompt must not be empty.")

        active_config = config or self.generation

        try:
            return self.strategy.generate(
                prompt_text,
                active_config,
                messages=normalized_messages,
            )
        except TypeError:
            # Backwards compatibility for custom strategies that ignore ``messages``.
            return self.strategy.generate(prompt_text, active_config)


__all__ = [
    "LanguageModel",
    "GenerationConfig",
    "register_strategy",
    "TORCH_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]

