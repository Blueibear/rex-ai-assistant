"""Strategy based language model interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

from .config import settings
from .logging_utils import configure_logger

LOGGER = configure_logger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = settings.temperature
    top_p: float = 0.95
    top_k: int = 50
    seed: int = 7


class LLMBackend(Protocol):
    """Protocol implemented by each backend strategy."""

    name: str

    def warm_up(self) -> None:  # pragma: no cover - optional
        """Prepare expensive resources if necessary."""

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        """Return a completion for ``prompt``."""


BackendFactory = Callable[[str], LLMBackend]


class TransformersBackend:
    """HuggingFace Transformers implementation."""

    name = "transformers"

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers package is required for TransformersBackend") from exc

        try:  # pragma: no cover - torch import triggers hardware probing
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required for TransformersBackend") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device_index = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
        )
        self._tokenizer = self._pipeline.tokenizer
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def warm_up(self) -> None:  # pragma: no cover - warmup not required
        return

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        cfg = config or GenerationConfig()
        outputs = self._pipeline(
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if not outputs:
            raise RuntimeError("Language model returned no candidates")

        generated = outputs[0]["generated_text"]
        completion = generated[len(prompt) :].strip()
        return completion or "(silence)"


class EchoBackend:
    """Fallback backend used when heavyweight models are unavailable."""

    name = "echo"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def warm_up(self) -> None:
        LOGGER.info("Using EchoBackend for model '%s'", self._model_name)

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        return f"[{self._model_name}] {prompt.strip()}"


class OpenAIBackend:
    """Backend that delegates to the OpenAI chat completions API."""

    name = "openai"

    def __init__(self, model_name: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for OpenAIBackend") from exc

        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be configured to use OpenAIBackend")
        self._model_name = model_name
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
        )
        choices = response.choices
        if not choices:
            raise RuntimeError("OpenAI returned no candidates")
        message = choices[0].message
        return (message.content or "").strip() or "(silence)"


class BackendRegistry:
    """Manage backend registrations and creation."""

    def __init__(self) -> None:
        self._registry: Dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        self._registry[name.lower()] = factory

    def create(self, name: str, model_name: str) -> LLMBackend:
        key = name.lower()
        if key not in self._registry:
            raise RuntimeError(f"Unknown LLM backend: {name}")
        return self._registry[key](model_name)


registry = BackendRegistry()
registry.register("transformers", TransformersBackend)
registry.register("echo", EchoBackend)
registry.register("openai", OpenAIBackend)


class LLMClient:
    """High level facade around the backend strategies."""

    def __init__(self, model_name: Optional[str] = None, backend: Optional[str] = None) -> None:
        backend_name = backend or settings.llm_backend
        model_name = model_name or settings.llm_model
        try:
            self._backend = registry.create(backend_name, model_name)
        except Exception as exc:
            LOGGER.warning("Falling back to EchoBackend due to: %s", exc)
            self._backend = EchoBackend(model_name)
        self._backend.warm_up()

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        return self._backend.generate(prompt, config=config)


__all__ = [
    "GenerationConfig",
    "LLMBackend",
    "LLMClient",
    "registry",
]
