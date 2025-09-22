"""Backwards compatibility shim."""

from rex.llm_client import GenerationConfig, LLMClient, registry

__all__ = ["GenerationConfig", "LLMClient", "registry"]
