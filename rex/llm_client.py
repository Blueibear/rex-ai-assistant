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
        self.tools = []  # Will be set by LanguageModel

    def _get_client(self) -> Any:
        if self._cached_client is None:
            self._cached_client = self._client_factory()
        return self._cached_client

    def generate(self, prompt: str, config: GenerationConfig, *, messages=None, tools=None) -> str:
        payload = messages or [{"role": "user", "content": prompt}]

        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": payload,
            "temperature": config.temperature,
            "max_tokens": config.max_new_tokens,
            "top_p": config.top_p,
        }

        # Add tools if available
        if tools or self.tools:
            api_params["tools"] = tools or self.tools
            # Don't force tool use, let model decide
            # api_params["tool_choice"] = "auto"

        response = self._get_client().chat.completions.create(**api_params)
        message = response.choices[0].message

        # Check if model wants to call a tool
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return message  # Return full message object for tool handling

        content = getattr(message, "content", "") or ""
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
        self._tools = []  # OpenAI tool definitions
        self._tool_functions = {}  # Map of tool name -> callable
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

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], function: Any) -> None:
        """Register a tool/function that the LLM can call.

        Args:
            name: Function name (e.g., "web_search")
            description: What the function does
            parameters: JSON schema for parameters
            function: Callable to execute when tool is invoked
        """
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        }
        self._tools.append(tool_def)
        self._tool_functions[name] = function

        # Update strategy if it supports tools
        if hasattr(self.strategy, 'tools'):
            self.strategy.tools = self._tools

        logger.info(f"Registered tool: {name}")

    def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute a tool call and return the result."""
        import json

        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        logger.info(f"Executing tool: {function_name} with args: {function_args}")

        if function_name not in self._tool_functions:
            return f"Error: Unknown function {function_name}"

        try:
            result = self._tool_functions[function_name](**function_args)
            return str(result) if result is not None else "No results found"
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error executing {function_name}: {str(e)}"

    def _format_messages(self, messages: Sequence[Dict[str, str]]) -> str:
        return "\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('content', '').strip()}" for m in messages if isinstance(m, dict)).strip()

    def generate(self, prompt: Optional[str] = None, *, messages: Optional[Sequence[Dict[str, str]]] = None, config: Optional[GenerationConfig] = None, max_tool_rounds: int = 3) -> str:
        if messages is not None:
            prompt_text = self._format_messages(messages)
            normalized_messages = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages if isinstance(m, dict)]
        elif isinstance(prompt, str):
            prompt_text = prompt
            normalized_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Prompt or messages must be provided.")

        if not prompt_text.strip():
            raise ValueError("Prompt must not be empty.")

        # Tool calling loop (only for OpenAI strategy with tools)
        if self.provider == "openai" and self._tools:
            current_messages = list(normalized_messages)

            for round_num in range(max_tool_rounds):
                try:
                    result = self.strategy.generate(prompt_text, config or self.generation, messages=current_messages, tools=self._tools)
                except TypeError:
                    # Fallback for strategies that don't support tools parameter
                    result = self.strategy.generate(prompt_text, config or self.generation, messages=current_messages)

                # Check if result is a message object with tool calls
                if hasattr(result, 'tool_calls') and result.tool_calls:
                    logger.info(f"Tool call round {round_num + 1}: Model requested {len(result.tool_calls)} tool(s)")

                    # Add assistant message to conversation
                    assistant_msg = {
                        "role": "assistant",
                        "content": result.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in result.tool_calls
                        ]
                    }
                    current_messages.append(assistant_msg)

                    # Execute each tool call and add results
                    for tool_call in result.tool_calls:
                        tool_result = self._execute_tool_call(tool_call)
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })

                    # Continue loop to get final response with tool results
                    continue

                # No tool calls - return final response
                return result.strip() if isinstance(result, str) else (result.content or "(silence)").strip()

            # Max rounds reached
            logger.warning(f"Max tool rounds ({max_tool_rounds}) reached")
            return "I apologize, but I encountered too many tool calls. Please try rephrasing your question."

        # No tools or not OpenAI - use original simple generation
        try:
            result = self.strategy.generate(prompt_text, config or self.generation, messages=normalized_messages)
            return result.strip() if isinstance(result, str) else str(result).strip()
        except TypeError:
            result = self.strategy.generate(prompt_text, config or self.generation)
            return result.strip() if isinstance(result, str) else str(result).strip()


__all__ = [
    "LanguageModel",
    "GenerationConfig",
    "register_strategy",
    "TORCH_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]

