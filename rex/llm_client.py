"""Language model client with pluggable backends (Transformers, OpenAI, Echo)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from rex.assistant_errors import ConfigurationError
from rex.config import AppConfig, load_config
from rex.logging_utils import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:  # pragma: no cover
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = hf_pipeline = None

try:  # pragma: no cover
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:  # pragma: no cover
    import ollama
    from ollama import Client as OllamaClient
except ImportError:
    ollama = OllamaClient = None

TORCH_AVAILABLE = torch is not None
TRANSFORMERS_AVAILABLE = all([AutoTokenizer, AutoModelForCausalLM, hf_pipeline])
OPENAI_AVAILABLE = OpenAI is not None
OLLAMA_AVAILABLE = ollama is not None


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


class EchoStrategy:
    name = "echo"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, _config: GenerationConfig, *, messages=None) -> str:
        text = prompt.strip()
        if not text and messages:
            text = "\n".join(m.get("content", "").strip() for m in messages).strip()
        return f"[{self.model_name}] {text or '(silence)'}"


class OfflineTransformersStrategy:
    name = "offline-transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, _config: GenerationConfig, *, messages=None) -> str:
        text = prompt.strip()
        if not text and messages:
            text = "\n".join(m.get("content", "").strip() for m in messages).strip()
        return f"(offline) {text or ''}".strip() or "(offline)"


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

    def generate(self, prompt: str, config: GenerationConfig, *, messages=None) -> str:
        if torch is None:
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
            generate_kwargs.update({
                "temperature": max(config.temperature, 1e-4),
                "top_p": config.top_p,
                "top_k": config.top_k,
            })

        outputs = self.pipeline(prompt, **generate_kwargs)
        text = outputs[0]["generated_text"]
        return text[len(prompt):].strip() or "(silence)"


class OpenAIStrategy:
    name = "openai"

    def __init__(self, model_name: str, client_factory) -> None:
        self.model_name = model_name
        self._client_factory = client_factory
        self._cached_client = None

    def _get_client(self) -> Any:
        if self._cached_client is None:
            self._cached_client = self._client_factory()
        return self._cached_client

    def generate(self, prompt: str, config: GenerationConfig, *, messages=None) -> str:
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


class OllamaStrategy:
    name = "ollama"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        use_cloud: bool = False
    ) -> None:
        if not OLLAMA_AVAILABLE:
            raise ConfigurationError("Ollama backend requires the `ollama` package.")
        self.model_name = model_name
        self.base_url = base_url
        self.use_cloud = use_cloud
        self.api_key = api_key

        if use_cloud and not api_key:
            raise ConfigurationError("Ollama cloud requires an API key.")

        try:
            if use_cloud:
                self._client = None  # cloud handled via ollama.chat
            else:
                self._client = OllamaClient(host=self.base_url)
                logger.info("Ollama local client initialized at %s", self.base_url)
        except Exception as exc:
            raise ConfigurationError(f"Failed to initialize Ollama client: {exc}") from exc

    def generate(self, prompt: str, config: GenerationConfig, *, messages=None) -> str:
        payload = messages or [{"role": "user", "content": prompt}]
        options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "seed": config.seed,
        }
        try:
            if self.use_cloud:
                response = ollama.chat(
                    model=self.model_name,
                    messages=payload,
                    options=options,
                    api_key=self.api_key,
                )
            else:
                response = self._client.chat(
                    model=self.model_name,
                    messages=payload,
                    options=options,
                )
            content = response.get("message", {}).get("content", "")
            return content.strip() or "(silence)"
        except Exception as exc:
            logger.error("Ollama generation failed (model=%s, host=%s): %s", self.model_name, self.base_url, exc)
            return (
                f"I had trouble contacting the Ollama server at {self.base_url}. "
                f"Make sure `ollama serve` is running and the `{self.model_name}` model is pulled."
            )


_STRATEGIES: Dict[str, type[LLMStrategy]] = {
    EchoStrategy.name: EchoStrategy,
    TransformersStrategy.name: TransformersStrategy,
}


def register_strategy(name: str, strategy: type[LLMStrategy]) -> None:
    _STRATEGIES[name] = strategy


class LanguageModel:
    def __init__(self, config: Optional[AppConfig] = None, **overrides) -> None:
        self.config = config or load_config()
        self.provider = (overrides.get("provider") or self.config.llm_provider).lower()
        self.model_name = overrides.get("model") or self.config.llm_model
        self.api_key = overrides.get("openai_api_key") or self.config.openai_api_key or os.getenv("OPENAI_API_KEY")

        self.generation = GenerationConfig(
            max_new_tokens=overrides.get("max_new_tokens", self.config.llm_max_tokens),
            temperature=overrides.get("temperature", self.config.llm_temperature),
            top_p=overrides.get("top_p", self.config.llm_top_p),
            top_k=overrides.get("top_k", self.config.llm_top_k),
            seed=overrides.get("seed", self.config.llm_seed),
        )

        self._openai_client = None
        self.strategy = self._init_strategy()

    def _init_strategy(self) -> LLMStrategy:
        if self.provider == "openai":
            return OpenAIStrategy(self.model_name, self._ensure_openai_client)

        if self.provider == "ollama":
            return OllamaStrategy(
                self.model_name,
                api_key=self.config.ollama_api_key,
                base_url=self.config.ollama_base_url,
                use_cloud=self.config.ollama_use_cloud,
            )

        strategy_cls = _STRATEGIES.get(self.provider)
        if strategy_cls is None:
            logger.warning("Unknown LLM provider '%s'. Falling back to echo mode.", self.provider)
            return EchoStrategy(self.model_name)

        if strategy_cls is TransformersStrategy and not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            logger.warning("Transformers missing; using offline fallback for model '%s'.", self.model_name)
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
        base_url = self.config.openai_base_url
        self._openai_client = OpenAI(api_key=self.api_key, base_url=base_url) if base_url else OpenAI(api_key=self.api_key)
        return self._openai_client

    def _format_messages(self, messages: Sequence[Dict[str, str]]) -> str:
        return "\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('content', '').strip()}" for m in messages if isinstance(m, dict)).strip()

    def generate(self, prompt: Optional[str] = None, *, messages: Optional[Sequence[Dict[str, str]]] = None, config: Optional[GenerationConfig] = None) -> str:
        if messages is not None:
            prompt_text = self._format_messages(messages)
            normalized_messages = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages if isinstance(m, dict)]
        elif isinstance(prompt, str):
            prompt_text = prompt
            normalized_messages = None
        else:
            raise ValueError("Prompt or messages must be provided.")

        if not prompt_text.strip():
            raise ValueError("Prompt must not be empty.")

        try:
            return self.strategy.generate(prompt_text, config or self.generation, messages=normalized_messages)
        except TypeError:
            return self.strategy.generate(prompt_text, config or self.generation)


__all__ = [
    "LanguageModel",
    "GenerationConfig",
    "register_strategy",
    "TORCH_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]

