"""Utility wrapper around language models used by Rex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from assistant_errors import ConfigurationError
from config import AppConfig, load_config
from logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class GenerationConfig:
    """Configuration options for sampling from the language model."""

    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int


class LanguageModel:
    """Thin wrapper that exposes a deterministic ``generate`` method."""

    def __init__(self, config: Optional[AppConfig] = None, **overrides) -> None:
        self._config = config or load_config()
        self._provider = overrides.get("provider", self._config.llm_provider).lower()
        self.model_name = overrides.get("model", self._config.llm_model)
        self.model_loaded = False

        self.generation = GenerationConfig(
            max_new_tokens=overrides.get("max_new_tokens", self._config.llm_max_tokens),
            temperature=overrides.get("temperature", self._config.llm_temperature),
            top_p=overrides.get("top_p", self._config.llm_top_p),
            top_k=overrides.get("top_k", self._config.llm_top_k),
            seed=overrides.get("seed", self._config.llm_seed),
        )

        if self._provider == "transformers":
            self._initialise_transformers()
        elif self._provider == "openai":
            self._initialise_openai()
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {self._provider}")

    def __str__(self):
        return f"<LanguageModel provider={self._provider}, model={self.model_name}>"

    def _initialise_transformers(self) -> None:
        if torch is None:
            raise ConfigurationError("The 'torch' package is required for the transformers provider.")

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        LOGGER.info("Loading HuggingFace model: %s", self.model_name)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        device_index = 0 if torch.cuda.is_available() else -1
        if device_index == -1:
            device_index = None  # Use CPU

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
        )
        self._tokenizer = self._pipeline.tokenizer
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self.model_loaded = True
        LOGGER.info("Transformers model ready.")

    def _initialise_openai(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError("The 'openai' package is required for OpenAI provider.") from exc

        api_key = self._config.openai_api_key
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY must be set for the OpenAI provider.")

        self._openai_client = OpenAI(api_key=api_key)
        self.model_loaded = True
        LOGGER.info("OpenAI client initialized.")

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        """Generate text completions for a given prompt."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        active_config = config or self.generation

        if self._provider == "transformers":
            return self._generate_transformers(prompt, active_config)
        elif self._provider == "openai":
            return self._generate_openai(prompt, active_config)

        raise ConfigurationError(f"Unknown provider: {self._provider}")

    def _generate_transformers(self, prompt: str, config: GenerationConfig) -> str:
        if torch is None:
            raise ConfigurationError("The 'torch' package is required for transformers provider.")

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        outputs = self._pipeline(
            prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        if not outputs:
            raise RuntimeError("Transformers model returned no candidates.")

        generated = outputs[0]["generated_text"]
        completion = generated[len(prompt):].strip()

        return completion or "(silence)"

    def _generate_openai(self, prompt: str, config: GenerationConfig) -> str:
        response = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
        )

        choices = response.choices
        if not choices:
            raise RuntimeError("OpenAI returned no candidates.")

        message = choices[0].message
        return (message.content or "").strip() or "(silence)"

