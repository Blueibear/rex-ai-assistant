"""Language model client with pluggable backends (Transformers, OpenAI, Echo)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, Dict

from config import AppConfig, load_config
from assistant_errors import ConfigurationError
from logging_utils import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
except ImportError:
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


# === Backend: Echo (fallback) ===
class EchoStrategy:
    name = "echo"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, _: GenerationConfig) -> str:
        if not prompt.strip():
            return "(silence)"
        return f"[{self.model_name}] {prompt.strip()}"


# === Backend: Transformers ===
class TransformersStrategy:
    name = "transformers"

    def __init__(self, model_name: str) -> None:
        if not (AutoTokenizer and AutoModelForCausalLM and hf_pipeline and torch):
            raise ConfigurationError("Transformers backend requires `torch` and `transformers`.")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1

        self.pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
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


# === Backend: OpenAI Chat Completion ===
class OpenAIStrategy:
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


# === Strategy Registry ===
_STRATEGIES: Dict[str, type[LLMStrategy]] = {
    EchoStrategy.name: EchoStrategy,
    TransformersStrategy.name: TransformersStrategy,
    OpenAIStrategy.name: OpenAIStrategy,
}


def register_strategy(name: str, strategy: type[LLMStrategy]) -> None:
    _STRATEGIES[name] = strategy


# === LLM Wrapper ===
class LanguageModel:
    """LLM wrapper supporting multiple backends (OpenAI, Transformers, Echo)."""

    def __init__(self, config: Optional[AppConfig] = None, **overrides) -> None:
        self.config = config or load_config()
        self.provider = (overrides.get("provider") or self.config.llm_provider).lower()
        self.model_name = overrides.get("model") or self.config.llm_model
        self.api_key = self.config.openai_api_key if self.provider == "openai" else None

        self.generation = GenerationConfig(
            max_new_tokens=overrides.get("max_new_tokens", self.config.llm_max_tokens),
            temperature=overrides.get("temperature", self.config.llm_temperature),
            top_p=overrides.get("top_p", self.config.llm_top_p),
            top_k=overrides.get("top_k", self.config.llm_top_k),
            seed=overrides.get("seed", self.config.llm_seed),
        )

        self.strategy = self._init_strategy()

    def _init_strategy(self) -> LLMStrategy:
        strategy_cls = _STRATEGIES.get(self.provider)
        if not strategy_cls:
            logger.warning("Unknown LLM provider '%s'. Falling back to echo mode.", self.provider)
            return EchoStrategy(self.model_name)

        try:
            if self.provider == "openai":
                return strategy_cls(self.model_name, self.api_key)
            return strategy_cls(self.model_name)
        except Exception as exc:
            logger.warning("LLM backend init failed (%s). Falling back to echo. (%s)", self.provider, exc)
            return EchoStrategy(self.model_name)

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        return self.strategy.generate(prompt, config or self.generation)

