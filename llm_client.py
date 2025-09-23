"""Utility wrapper around transformer language models used by Rex."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:  # pragma: no cover - exercised indirectly via fallback
    import torch  # type: ignore[import-not-found]
except (ImportError, OSError):  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised indirectly via fallback
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )
except (ImportError, OSError):  # pragma: no cover - transformers are optional
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    hf_pipeline = None  # type: ignore[assignment]


class _EchoTokenizer:
    """Minimal tokenizer used when transformer dependencies are unavailable."""

    pad_token_id: int | None
    eos_token_id: int | None

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0


class _EchoPipeline:
    """Simple text-generation shim that echoes prompts for deterministic tests."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = _EchoTokenizer()

    def __call__(self, prompt: str, **_: Any) -> list[dict[str, str]]:
        completion = f"[{self.model_name}] {prompt.strip()}"
        return [{"generated_text": f"{prompt}{' ' if prompt and not prompt.endswith(' ') else ''}{completion}"}]


DEFAULT_MODEL_NAME = os.getenv("REX_LLM_MODEL", "distilgpt2")
DEFAULT_MAX_TOKENS = int(os.getenv("REX_LLM_MAX_TOKENS", "120"))
DEFAULT_TEMPERATURE = float(os.getenv("REX_LLM_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("REX_LLM_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("REX_LLM_TOP_K", "50"))
DEFAULT_SEED = int(os.getenv("REX_LLM_SEED", "42"))


@dataclass
class GenerationConfig:
    """Configuration options for sampling from the language model."""

    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    seed: int = DEFAULT_SEED


class LanguageModel:
    """Thin wrapper that exposes a deterministic ``generate`` method."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.model_name = model_name
        self._config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        self._torch = torch
        self._pipeline = self._build_pipeline(model_name)
        self._tokenizer = self._pipeline.tokenizer
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            self._tokenizer.pad_token_id = getattr(
                self._tokenizer, "eos_token_id", 0
            )

    def _build_pipeline(self, model_name: str) -> Callable[..., list[dict[str, str]]]:
        if AutoTokenizer is None or AutoModelForCausalLM is None or hf_pipeline is None:
            return _EchoPipeline(model_name)

        if self._torch is None:
            return _EchoPipeline(model_name)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            device_index = 0 if self._torch.cuda.is_available() else -1
            return hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_index,
            )
        except Exception:
            return _EchoPipeline(model_name)

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        """Generate text continuations for a given prompt."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        active_config = config or self._config
        if self._torch is not None:
            self._torch.manual_seed(active_config.seed)
            if hasattr(self._torch, "cuda") and self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(active_config.seed)

        outputs = self._pipeline(
            prompt,
            max_new_tokens=active_config.max_new_tokens,
            do_sample=active_config.temperature > 0,
            temperature=active_config.temperature,
            top_p=active_config.top_p,
            top_k=active_config.top_k,
            pad_token_id=getattr(self._tokenizer, "pad_token_id", None),
        )
        if not outputs:
            raise RuntimeError("Language model returned no candidates.")

        generated = outputs[0]["generated_text"]
        completion = generated[len(prompt) :].strip()
        if not completion:
            return "(silence)"
        return completion
