"""Language model utilities with pluggable backends.

Historically this logic lived in :mod:`rex.llm_client`. During previous
refactors the root module became a thin re-export, which caused merge
conflicts and made the original entry point harder to understand. The
implementation now lives here again while the package module simply imports
from this file for backwards compatibility.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

logger = logging.getLogger(__name__)

from assistant_errors import ConfigurationError
from config import AppConfig, load_config

try:  # Optional dependencies
    import torch  # type: ignore[import-not-found]
except (ImportError, OSError):  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - transformers optional
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )
except (ImportError, OSError):
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    hf_pipeline = None  # type: ignore[assignment]

DEFAULT_PROVIDER = "transformers"
DEFAULT_MODEL = "distilgpt2"
DEFAULT_MAX_TOKENS = 120
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_SEED = 42


class LLMDependencyError(RuntimeError):
    """Raised when a backend cannot initialise due to missing dependencies."""

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int


class LLMStrategy(Protocol):
    name: str

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        ...


class EchoStrategy:
    name = "echo"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, _: GenerationConfig) -> str:
        content = prompt.strip()
        if not content:
            return "(silence)"
        return f"[{self.model_name}] {content}"


class TransformersStrategy:
    name = "transformers"

    def __init__(self, model_name: str) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or hf_pipeline is None:
            raise LLMDependencyError("transformers backend requires the transformers package")
        if torch is None:
            raise LLMDependencyError("transformers backend requires torch to be installed")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise LLMDependencyError(str(exc)) from exc

        device_index = 0 if torch.cuda.is_available() else -1
        self.pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
        )
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        torch.manual_seed(config.seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        outputs = self.pipeline(
            prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        if not outputs:  # pragma: no cover - defensive guard
            raise RuntimeError("Language model returned no candidates")
        generated = outputs[0]["generated_text"]
        return generated[len(prompt) :].strip() or "(silence)"


class OpenAIStrategy:
    name = "openai"

    def __init__(self, model_name: str, api_key: Optional[str]) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMDependencyError("openai backend requires the openai package") from exc

        if not api_key:
            raise LLMDependencyError("openai backend requires an API key")

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
        message = response.choices[0].message.content or ""
        return message.strip() or "(silence)"


# === Backend: Ollama Local REST ===
class OllamaStrategy:
    name = "ollama"

    def __init__(self, model_name: str, url: str) -> None:
        if requests is None:
            raise ConfigurationError("Ollama backend requires the 'requests' package.")
        self.model_name = model_name
        self.url = url.rstrip("/")

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_new_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except Exception as exc:
            raise ConfigurationError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        text = (data.get("response") or "").strip()
        if not text:
            text = "(silence)"
        return text


# === Strategy Registry ===
_STRATEGIES: Dict[str, type[LLMStrategy]] = {
    EchoStrategy.name: EchoStrategy,
    TransformersStrategy.name: TransformersStrategy,
    OpenAIStrategy.name: OpenAIStrategy,
    OllamaStrategy.name: OllamaStrategy,
}


def register_strategy(name: str, strategy: type[LLMStrategy]) -> None:
    """Register a custom LLM backend at runtime."""

    _STRATEGIES[name] = strategy


class LanguageModel:
    """Unified LLM wrapper supporting transformers, OpenAI, Ollama, and custom backends."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        provider: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
        **extras,
    ) -> None:
        cfg = config or load_config()
        self.config = cfg

        base_provider = getattr(cfg, "llm_backend", DEFAULT_PROVIDER)
        base_model = getattr(cfg, "llm_model", DEFAULT_MODEL)
        base_max_tokens = getattr(cfg, "llm_max_tokens", DEFAULT_MAX_TOKENS)
        base_temperature = getattr(cfg, "llm_temperature", DEFAULT_TEMPERATURE)
        base_top_p = getattr(cfg, "llm_top_p", DEFAULT_TOP_P)
        base_top_k = getattr(cfg, "llm_top_k", DEFAULT_TOP_K)
        base_seed = getattr(cfg, "llm_seed", DEFAULT_SEED)
        base_openai_key = openai_api_key or getattr(cfg, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")
        base_ollama_url = ollama_url or extras.get("ollama_url") or getattr(cfg, "llm_url", "http://localhost:11434")

        provider_override = provider or backend or extras.get("provider") or extras.get("backend")
        model_override = model_name or extras.get("model")

        self.provider = (provider_override or base_provider or DEFAULT_PROVIDER).lower()
        self.model_name = model_override or base_model or DEFAULT_MODEL
        self.api_key = base_openai_key
        self.ollama_url = base_ollama_url.rstrip("/")

        max_tokens = extras.get("max_new_tokens") or max_new_tokens or base_max_tokens
        temp_value = extras.get("temperature")
        if temp_value is None:
            temp_value = temperature if temperature is not None else base_temperature
        top_p_value = extras.get("top_p") or top_p or base_top_p
        top_k_value = extras.get("top_k") or top_k or base_top_k
        seed_value = extras.get("seed") or seed or base_seed

        self.generation = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temp_value,
            top_p=top_p_value,
            top_k=top_k_value,
            seed=seed_value,
        )

        self.strategy = self._init_strategy()

    def _init_strategy(self) -> LLMStrategy:
        strategy_cls = _STRATEGIES.get(self.provider)
        if not strategy_cls:
            logger.warning("Unknown LLM provider '%s'. Falling back to echo.", self.provider)
            return EchoStrategy(self.model_name)

        try:
            if strategy_cls is OpenAIStrategy or self.provider == OpenAIStrategy.name:
                return strategy_cls(self.model_name, self.api_key)  # type: ignore[call-arg]
            if strategy_cls is OllamaStrategy or self.provider == OllamaStrategy.name:
                return strategy_cls(self.model_name, self.ollama_url)  # type: ignore[call-arg]
            return strategy_cls(self.model_name)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - fallback guard
            logger.warning("LLM backend init failed (%s). Falling back to echo. (%s)", self.provider, exc)
            return EchoStrategy(self.model_name)

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        return self.strategy.generate(prompt, config or self.generation)


__all__ = [
    "LanguageModel",
    "GenerationConfig",
    "register_strategy",
    "LLMDependencyError",
]
