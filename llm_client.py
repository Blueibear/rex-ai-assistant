"""Utility wrapper around transformer language models used by Rex."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - import guard
    import torch
except ImportError:  # pragma: no cover - dependency optional for tests
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:  # pragma: no cover - dependency optional for tests
    AutoModelForCausalLM = AutoTokenizer = pipeline = None  # type: ignore[assignment]


TORCH_AVAILABLE = torch is not None
TRANSFORMERS_AVAILABLE = pipeline is not None and AutoTokenizer is not None


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
        self._transformer_available = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
        self._pipeline = None
        self._tokenizer = None
        self._device = "cpu"

        if self._transformer_available:
            assert torch is not None  # for type checkers
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = self._build_pipeline(model_name)
            self._tokenizer = self._pipeline.tokenizer
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _build_pipeline(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device_index = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
        )

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        """Generate text continuations for a given prompt."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        if not self._transformer_available:
            return self._fallback_response(prompt)

        active_config = config or self._config
        assert self._pipeline is not None
        assert self._tokenizer is not None
        assert torch is not None

        torch.manual_seed(active_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(active_config.seed)
        outputs = self._pipeline(
            prompt,
            max_new_tokens=active_config.max_new_tokens,
            do_sample=active_config.temperature > 0,
            temperature=active_config.temperature,
            top_p=active_config.top_p,
            top_k=active_config.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if not outputs:
            raise RuntimeError("Language model returned no candidates.")

        generated = outputs[0]["generated_text"]
        completion = generated[len(prompt) :].strip()
        if not completion:
            return "(silence)"
        return completion

    def _fallback_response(self, prompt: str) -> str:
        """Return a deterministic response when transformer deps are missing."""

        last_line = ""
        for line in reversed(prompt.splitlines()):
            if line.strip():
                last_line = line.strip()
                break
        if last_line.lower().startswith("user:"):
            message = last_line.split(":", 1)[1].strip()
        else:
            message = last_line

        if not message:
            message = "your request"

        return (
            "(offline) I'm running without PyTorch/transformers, "
            f"so here's a simple acknowledgement: You said '{message}'."
        )
