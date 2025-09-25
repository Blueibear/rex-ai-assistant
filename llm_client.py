"""Language model utilities with pluggable backends.

Historically this logic lived in :mod:`rex.llm_client`.  During previous
refactors the root module became a thin re-export, which caused merge
conflicts and made the original entry point harder to understand.  The
implementation now lives here again while the package module simply imports
from this file for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from rex.config import settings

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import-not-found]
except (ImportError, OSError):  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )
except (ImportError, OSError):  # pragma: no cover - transformers optional
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    hf_pipeline = None  # type: ignore[assignment]


class LLMDependencyError(RuntimeError):
    """Raised when a backend cannot initialise due to missing dependencies."""


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


class _EchoTokenizer:
    pad_token_id: int | None
    eos_token_id: int | None

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0


class EchoStrategy:
    name = "echo"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = _EchoTokenizer()

    def generate(self, prompt: str, _: GenerationConfig) -> str:
        completion = (
            f"[{self.model_name}] {prompt.strip()}" if prompt.strip() else f"[{self.model_name}]"
        )
        if prompt and not prompt.endswith(" "):
            prompt = f"{prompt} "
        return f"{prompt}{completion}"[len(prompt) :].strip() or "(silence)"


class TransformersStrategy:
    name = "transformers"

    def __init__(self, model_name: str) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or hf_pipeline is None:
            raise LLMDependencyError("transformers is not available")
        if torch is None:
            raise LLMDependencyError("torch is not available")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            device_index = 0 if torch.cuda.is_available() else -1
            self.pipeline = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_index,
            )
            self.tokenizer = self.pipeline.tokenizer
        except Exception as exc:  # pragma: no cover - defensive guard
            raise LLMDependencyError(str(exc)) from exc

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
        if not outputs:
            raise RuntimeError("Language model returned no candidates.")
        generated = outputs[0]["generated_text"]
        return generated[len(prompt) :].strip() or "(silence)"


_STRATEGIES: Dict[str, type[LLMStrategy]] = {
    EchoStrategy.name: EchoStrategy,
    TransformersStrategy.name: TransformersStrategy,
}


def register_strategy(name: str, factory: type[LLMStrategy]) -> None:
    _STRATEGIES[name] = factory


class LanguageModel:
    """Thin wrapper that exposes a deterministic ``generate`` method."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: str | None = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.llm_model
        self._config = GenerationConfig(
            max_new_tokens=max_new_tokens or 120,
            temperature=temperature if temperature is not None else settings.temperature,
            top_p=top_p or 0.9,
            top_k=top_k or 50,
            seed=seed or 42,
        )
        self._backend_name = backend or settings.llm_backend
        self._strategy = self._initialise_strategy(self._backend_name)

    def _initialise_strategy(self, backend: str) -> LLMStrategy:
        factory = _STRATEGIES.get(backend)
        if factory is None:
            raise ValueError(f"Unknown LLM backend '{backend}'")
        try:
            return factory(self.model_name)  # type: ignore[call-arg]
        except LLMDependencyError:
            if backend != EchoStrategy.name:
                return self._initialise_strategy(EchoStrategy.name)
            raise

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        active_config = config or self._config
        return self._strategy.generate(prompt, active_config)


__all__ = ["LanguageModel", "GenerationConfig", "register_strategy", "LLMDependencyError"]
