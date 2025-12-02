"""Backward compatibility wrapper - imports from rex.llm_client.

New code should import directly from rex.llm_client.
"""

from __future__ import annotations

# Re-export all classes and functions from the rex.llm_client package
from rex.llm_client import (
    GenerationConfig,
    LanguageModel,
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    register_strategy,
)

__all__ = [
    "LanguageModel",
    "GenerationConfig",
    "register_strategy",
    "TORCH_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]

