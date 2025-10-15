#!/usr/bin/env python3
"""Comprehensive import test for Rex modules - validates namespace structure."""
import sys
import traceback
from typing import List, Tuple

def test_import(module_path: str, item_name: str) -> bool:
    """Test importing a specific item from a module."""
    try:
        module = __import__(module_path, fromlist=[item_name])
        getattr(module, item_name)
        print(f"✓ {module_path}.{item_name}")
        return True
    except Exception as e:
        print(f"✗ {module_path}.{item_name}: {e}")
        traceback.print_exc()
        return False

def test_module(module_path: str) -> bool:
    """Test importing a module directly."""
    try:
        __import__(module_path)
        print(f"✓ {module_path}")
        return True
    except Exception as e:
        print(f"✗ {module_path}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Rex package imports...\n")
    
    # Test core rex package exports
    core_tests: List[Tuple[str, str]] = [
        ("rex", "settings"),
        ("rex", "reload_settings"),
        ("rex", "configure_logging"),
    ]
    
    # Test rex submodules
    submodule_tests: List[Tuple[str, str]] = [
        ("rex.config", "AppConfig"),
        ("rex.config", "Settings"),
        ("rex.config", "settings"),
        ("rex.config", "load_config"),
        ("rex.assistant_errors", "AssistantError"),
        ("rex.assistant_errors", "ConfigurationError"),
        ("rex.assistant_errors", "AuthenticationError"),
        ("rex.llm_client", "LanguageModel"),
        ("rex.llm_client", "GenerationConfig"),
        ("rex.memory_utils", "load_users_map"),
        ("rex.memory_utils", "extract_voice_reference"),
        ("rex.memory", "trim_history"),
        ("rex.logging_utils", "get_logger"),
        ("rex.logging_utils", "configure_logging"),
        ("rex.plugins", "Plugin"),
        ("rex.plugins", "load_plugins"),
        ("rex.assistant", "Assistant"),
    ]
    
    # Test backward compatibility wrappers
    compat_tests: List[Tuple[str, str]] = [
        ("config", "settings"),
        ("assistant_errors", "ConfigurationError"),
        ("llm_client", "LanguageModel"),
        ("memory_utils", "load_users_map"),
        ("logging_utils", "get_logger"),
    ]
    
    # Test top-level scripts
    script_tests: List[str] = [
        "rex_assistant",
        "rex_speak_api",
    ]
    
    results = []
    
    print("=== Core Rex Package ===")
    for module, item in core_tests:
        results.append(test_import(module, item))
    
    print("\n=== Rex Submodules ===")
    for module, item in submodule_tests:
        results.append(test_import(module, item))
    
    print("\n=== Backward Compatibility ===")
    for module, item in compat_tests:
        results.append(test_import(module, item))
    
    print("\n=== Top-Level Scripts ===")
    for module in script_tests:
        results.append(test_module(module))
    
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed ({passed*100//total}%)")
    print(f"{'='*50}")
    
    sys.exit(0 if all(results) else 1)
