"""Language model utilities with pluggable backends.

Supports:
- EchoStrategy (testing, no real LLM)
- TransformersStrategy (Hugging Face models)
- OpenAIStrategy (OpenAI API)

Historically this logic lived in :mod:`rex.llm_client`. This module now
contains the implementation while the package module re-exports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Dict

from rex.config import settings
from rex.assistant_errors import ConfigurationError

import logging
logger = logging.getLogger(__name__)

# Optional dependencies
try:  # pragma: no cover - optional
    import torch
except (ImportError, OSError):
    torch = None

try:  # pragma: no cover - optional
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
except (ImportError, OSError):
    AutoTokenizer = AutoModelForCausalLM = hf_pipeline = None


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int


class LLMStrategy(Protocol):
    name: str
    def generate(self, prompt: str, config: GenerationConfig) -> str: ...


# ────────────────────────────── STRATEGIES ──────────────────────────────

class EchoStrategy:
    """Dummy backend that just echoes the input."""
    name = "echo"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, _: GenerationConfig) -> str:
        if not prompt.strip():
            return "(silence)"
        return f"[{self.model_name}] {prompt.strip()}"


class TransformersStrategy:
    """Hugging Face transformers backend."""
    name = "transformers"

    def __init__(self, model_name: str) -> None:
        if not (AutoTokenizer and AutoModelForCausalLM and hf_pipeline and torch):
            raise ConfigurationError("Transformers backend requires `torch` and `transformers`.")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1

        self.pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        outputs = self.pipeline(
            prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = outputs[0]["generated_text"]
        return text[len(prompt):].strip() or "(silence)"


class OpenAIStrategy:
    """OpenAI ChatCompletion backend."""
    name = "openai"

    def __init__(self, model_name: str, api_key: Optional[str]) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ConfigurationError("OpenAI backend requires the `openai` package.")
        if not api_key:
            raise ConfigurationError("Missing OpenAI API key.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
        )
        return (response.choices[0].message.content or "").strip() or "(silence)"


# Strategy registry
_STRATEGIES: Dict[str, type[LLMStrategy]] = {
    EchoStrategy.name: EchoStrategy,
    TransformersStrategy.name: TransformersStrategy,
    OpenAIStrategy.name: OpenAIStrategy,
}


def register_strategy(name: str, strategy: type[LLMStrategy]) -> None:
    _STRATEGIES[name] = strategy


# ────────────────────────────── WRAPPER ──────────────────────────────

class LanguageModel:
    """Unified LLM wrapper supporting multiple backends (Echo, Transformers, OpenAI)."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: str | None = None,
        api_key: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.llm_model
        self._config = GenerationConfig(
            max_new_tokens=max_new_tokens or settings.llm_max_tokens,
            temperature=temperature if temperature is not None else settings.temperature,
            top_p=top_p or settings.llm_top_p,
            top_k=top_k or settings.llm_top_k,
            seed=seed or settings.llm_seed,
        )
        self._backend_name = (backend or settings.llm_backend).lower()
        self.api_key = api_key or getattr(settings, "openai_api_key", None)
        self._strategy = self._initialise_strategy(self._backend_name)

    def _initialise_strategy(self, backend: str) -> LLMStrategy:
        factory = _STRATEGIES.get(backend)
        if not factory:
            logger.warning("Unknown LLM backend '%s'. Falling back to echo.", backend)
            return EchoStrategy(self.model_name)
        try:
            if backend == OpenAIStrategy.name:
                return factory(self.model_name, self.api_key)  # type: ignore
            return factory(self.model_name)  # type: ignore
        except Exception as exc:
            logger.warning("LLM backend init failed (%s). Falling back to echo. (%s)", backend, exc)
            return EchoStrategy(self.model_name)

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        return self._strategy.generate(prompt, config or self._config)


__all__ = ["LanguageModel", "GenerationConfig", "register_strategy"]

