"""Configuration validation and startup checks for Rex AI Assistant."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .config import Settings, settings

logger = logging.getLogger(__name__)


def check_optional_dependency(module_name: str) -> bool:
    """Check if an optional dependency is installed."""
    return importlib.util.find_spec(module_name) is not None


def validate_config(config: Optional[Settings] = None) -> list[str]:
    """
    Validate configuration and return list of error messages.

    Performs conditional validation based on selected backends and features.
    Returns empty list if all validations pass.

    Args:
        config: Settings object to validate (defaults to global settings)

    Returns:
        List of validation error messages (empty if valid)
    """
    cfg = config or settings
    errors = []

    # Backend-specific validation
    if cfg.llm_backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            errors.append(
                "OPENAI_API_KEY environment variable required when llm_backend='openai'. "
                "Set via: export OPENAI_API_KEY=sk-..."
            )

    # Validate model paths for local models
    if cfg.llm_backend == "transformers":
        # Model name should not contain path traversal
        if ".." in cfg.llm_model or cfg.llm_model.startswith("/"):
            errors.append(
                f"llm_model '{cfg.llm_model}' appears to be a file path. "
                "Use Hugging Face model IDs (e.g., 'gpt2', 'facebook/opt-125m')"
            )

    # Validate audio device indices if specified
    if cfg.input_device is not None:
        try:
            if cfg.input_device < 0:
                errors.append(f"input_device must be >= 0, got {cfg.input_device}")
        except (TypeError, ValueError):
            errors.append(f"input_device must be an integer, got {type(cfg.input_device)}")

    if cfg.output_device is not None:
        try:
            if cfg.output_device < 0:
                errors.append(f"output_device must be >= 0, got {cfg.output_device}")
        except (TypeError, ValueError):
            errors.append(f"output_device must be an integer, got {type(cfg.output_device)}")

    # Validate numeric ranges
    if not (0.0 <= cfg.temperature <= 2.0):
        errors.append(f"temperature must be in [0.0, 2.0], got {cfg.temperature}")

    if not (0.0 < cfg.wakeword_threshold <= 1.0):
        errors.append(
            f"wakeword_threshold must be in (0.0, 1.0], got {cfg.wakeword_threshold}"
        )

    if cfg.max_memory_items < 1:
        errors.append(f"max_memory_items must be >= 1, got {cfg.max_memory_items}")

    # Validate paths exist or can be created
    transcripts_dir = Path(cfg.transcripts_dir)
    if transcripts_dir.exists() and not transcripts_dir.is_dir():
        errors.append(
            f"transcripts_dir '{transcripts_dir}' exists but is not a directory"
        )

    return errors


def check_feature_availability() -> dict[str, bool]:
    """
    Check which optional features are available based on installed dependencies.

    Returns:
        Dictionary mapping feature names to availability (True/False)
    """
    return {
        "audio_input": check_optional_dependency("sounddevice"),
        "audio_output": check_optional_dependency("simpleaudio"),
        "speech_to_text": check_optional_dependency("whisper"),
        "text_to_speech": check_optional_dependency("TTS"),
        "wakeword_detection": check_optional_dependency("openwakeword"),
        "web_search": check_optional_dependency("bs4"),  # beautifulsoup4
        "cuda": check_optional_dependency("torch") and _check_cuda(),
    }


def _check_cuda() -> bool:
    """Check if CUDA is available for PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def print_startup_table(config: Optional[Settings] = None, verbose: bool = False) -> None:
    """
    Print a formatted table showing configuration and system status.

    Args:
        config: Settings object to display (defaults to global settings)
        verbose: If True, show all settings; if False, show key settings only
    """
    cfg = config or settings

    print("\n" + "=" * 70)
    print(" Rex AI Assistant - Startup Configuration".center(70))
    print("=" * 70)

    # Configuration section
    print("\n📋 CONFIGURATION:")
    print(f"  LLM Backend:        {cfg.llm_backend}")
    print(f"  LLM Model:          {cfg.llm_model}")
    print(f"  Temperature:        {cfg.temperature}")
    print(f"  Whisper Model:      {cfg.whisper_model}")
    print(f"  Whisper Device:     {cfg.whisper_device}")
    print(f"  Wakeword:           {cfg.wakeword_keyword} (threshold: {cfg.wakeword_threshold})")
    print(f"  User ID:            {cfg.user_id}")
    print(f"  Memory Limit:       {cfg.max_memory_items} turns")

    if verbose:
        print(f"  Sample Rate:        {cfg.sample_rate} Hz")
        print(f"  Capture Duration:   {cfg.capture_seconds}s")
        print(f"  Input Device:       {cfg.input_device if cfg.input_device is not None else 'default'}")
        print(f"  Output Device:      {cfg.output_device if cfg.output_device is not None else 'default'}")
        print(f"  Transcripts Dir:    {cfg.transcripts_dir}")
        print(f"  Log Path:           {cfg.log_path}")

    # Feature availability section
    features = check_feature_availability()
    print("\n✨ AVAILABLE FEATURES:")
    for feature, available in features.items():
        status = "✅" if available else "❌"
        feature_name = feature.replace("_", " ").title()
        print(f"  {status} {feature_name}")

    # Validation section
    validation_errors = validate_config(cfg)
    if validation_errors:
        print("\n⚠️  CONFIGURATION ERRORS:")
        for error in validation_errors:
            print(f"  • {error}")
        print("\n" + "=" * 70)
        print("❌ Configuration validation failed. Please fix errors above.")
        print("=" * 70 + "\n")
        sys.exit(1)
    else:
        print("\n✅ Configuration valid")

    print("=" * 70 + "\n")


def validate_and_exit_on_error(config: Optional[Settings] = None) -> None:
    """
    Validate configuration and exit with error message if invalid.

    Use this at application startup to fail fast with clear error messages.

    Args:
        config: Settings object to validate (defaults to global settings)
    """
    errors = validate_config(config)
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error("  - %s", error)
        sys.exit(1)


__all__ = [
    "validate_config",
    "check_feature_availability",
    "print_startup_table",
    "validate_and_exit_on_error",
]
