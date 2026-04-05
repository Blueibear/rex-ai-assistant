"""Backward compatibility wrapper - imports from rex.llm_client.

New code should import directly from rex.llm_client.

.. deprecated::
    Import from ``rex.llm_client`` instead. This shim will be removed in a future cycle.
    References still exist in: tests/test_llm_client.py,
    tests/test_us013_openai_provider.py, tests/test_us014_anthropic_provider.py,
    tests/test_us015_local_llm_provider.py, tests/test_us016_provider_routing.py
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from root-level 'llm_client' is deprecated. "
    "Use 'from rex.llm_client import ...' instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all classes and functions from the rex.llm_client package
from rex.llm_client import (
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    GenerationConfig,
    LanguageModel,
    register_strategy,
)

__all__ = [
    "LanguageModel",
    "GenerationConfig",
    "register_strategy",
    "TORCH_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]
