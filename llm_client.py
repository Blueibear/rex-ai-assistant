"""Utility wrapper around language models used by Rex."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

try:  # pragma: no cover - import guard
    import torch
except ImportError:  # pragma: no cover - dependency optional for tests
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:  # pragma: no cover - dependency optional for tests
    AutoModelForCausalLM = AutoTokenizer = pipeline = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


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
    """Expose a deterministic ``generate`` method across providers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        seed: int = DEFAULT_SEED,
        provider: Optional[str] = None,
    ) -> None:
        self._logger = logging.getLogger("rex.llm")
        if not self._logger.handlers:
            logging.basicConfig(
                level=os.getenv("REX_LOG_LEVEL", "INFO").upper(),
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )

        self._model_pattern = re.compile(r"^[\w\-./]+(\/[\w\-./]+)*$")
        self._provider = (provider or self._infer_provider(model_name)).lower()
        self._openai_client = None
        self._openai_model: Optional[str] = None

        if self._provider == "openai":
            self._openai_model = self._normalise_openai_model(model_name)
            self.model_name = self._openai_model
        else:
            self.model_name = self._validate_model_name(model_name)

        self._config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        self._transformer_available = (
            self._provider == "transformers"
            and TORCH_AVAILABLE
            and TRANSFORMERS_AVAILABLE
        )
        self._pipeline = None
        self._tokenizer = None
        self._device = "cpu"

        if self._transformer_available:
            assert torch is not None  # for type checkers
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = self._build_pipeline(self.model_name)
            self._tokenizer = self._pipeline.tokenizer
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Provider helpers
    # ------------------------------------------------------------------
    def _infer_provider(self, model_name: str) -> str:
        lowered = model_name.lower()
        if lowered.startswith("openai:") or lowered.startswith("openai/"):
            return "openai"
        return "transformers"

    def _normalise_openai_model(self, model_name: str) -> str:
        if ":" in model_name:
            _, _, remainder = model_name.partition(":")
        elif "/" in model_name:
            parts = model_name.split("/", 1)
            remainder = parts[1] if len(parts) > 1 else model_name
        else:
            remainder = model_name
        clean = remainder.strip()
        if not clean:
            raise ValueError("OpenAI model name must not be empty.")
        return clean

    def _validate_model_name(self, model_name: str) -> str:
        if not self._model_pattern.match(model_name):
            raise ValueError(
                "Model name may only contain letters, numbers, '-', '_', '.' and '/'."
            )
        if ".." in model_name or model_name.startswith(('/', '.')):
            raise ValueError("Model name must not contain path traversal segments.")
        return model_name

    def _build_pipeline(self, model_name: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - network/dependency issues
            raise RuntimeError(
                f"Failed to load transformer model '{model_name}': {exc}"
            )
        device_index = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
        )

    def _ensure_openai_client(self):
        if self._openai_client is None:
            if OpenAI is None:
                raise RuntimeError(
                    "openai package is not installed; install 'openai' to use this provider."
                )
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable is required for the OpenAI provider."
                )
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    # ------------------------------------------------------------------
    # Prompt & message preparation helpers
    # ------------------------------------------------------------------
    def _messages_to_prompt(self, messages: Sequence[Dict[str, str]]) -> str:
        system_blocks: List[str] = []
        conversation: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                system_blocks.append(content)
            else:
                label = "User" if role == "user" else "Assistant"
                conversation.append(f"{label}: {content}")
        prompt_sections: List[str] = []
        if system_blocks:
            prompt_sections.append("\n\n".join(system_blocks))
        if conversation:
            prompt_sections.append("\n".join(conversation))
        else:
            prompt_sections.append("User: \nAssistant:")
        prompt = "\n\n".join(prompt_sections)
        if not prompt.strip().endswith("Assistant:"):
            prompt += "\nAssistant:"
        return prompt

    def _prepare_prompt(
        self,
        prompt: Optional[str],
        messages: Optional[Sequence[Dict[str, str]]],
    ) -> str:
        if prompt and prompt.strip():
            return prompt
        if not messages:
            raise ValueError("Either a prompt string or chat messages must be provided.")
        return self._messages_to_prompt(messages)

    def _prepare_messages(
        self,
        prompt: Optional[str],
        messages: Optional[Sequence[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        if messages:
            return [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ]
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty when no chat messages are supplied.")
        return [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------------
    # Generation APIs
    # ------------------------------------------------------------------
    def _generate_openai(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        config: Optional[GenerationConfig],
    ) -> str:
        client = self._ensure_openai_client()
        cfg = config or self._config
        response = client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        if not response.choices:
            raise RuntimeError("OpenAI returned no choices.")
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if not content:
            raise RuntimeError("OpenAI response did not include content.")
        return content.strip()

    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        config: Optional[GenerationConfig] = None,
        messages: Optional[Sequence[Dict[str, str]]] = None,
    ) -> str:
        """Generate text continuations for a prompt or structured messages."""

        if self._provider == "openai":
            try:
                chat_messages = self._prepare_messages(prompt, messages)
                return self._generate_openai(chat_messages, config=config)
            except Exception as exc:
                self._logger.warning("OpenAI generation failed: %s", exc)
                fallback_prompt = prompt or self._messages_to_prompt(
                    self._prepare_messages(prompt, messages)
                )
                return self._fallback_response(fallback_prompt)

        prepared_prompt = self._prepare_prompt(prompt, messages)

        if not self._transformer_available:
            return self._fallback_response(prepared_prompt)

        active_config = config or self._config
        assert self._pipeline is not None
        assert self._tokenizer is not None
        assert torch is not None

        torch.manual_seed(active_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(active_config.seed)
        outputs = self._pipeline(
            prepared_prompt,
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
        completion = generated[len(prepared_prompt) :].strip()
        if not completion:
            return "(silence)"
        return completion

    # ------------------------------------------------------------------
    # Fallback behaviour
    # ------------------------------------------------------------------
    def _fallback_response(self, prompt: str) -> str:
        """Return a deterministic response when model dependencies are missing."""

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
            "(offline) I'm running without a compatible language model, "
            f"so here's a simple acknowledgement: You said '{message}'."
        )


__all__ = ["LanguageModel", "GenerationConfig", "TORCH_AVAILABLE", "TRANSFORMERS_AVAILABLE"]
