"""Rex AI Assistant - Post-Deployment Validation Script

Run this after deploying to verify the runtime environment is correctly configured.
"""

import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Canonical torch version range (must match pyproject.toml ml extras)
TORCH_MIN = (2, 6, 0)
TORCH_MAX = (2, 9, 0)  # exclusive

# CLI entrypoints declared in pyproject.toml [project.scripts]
CLI_ENTRYPOINTS: list[tuple[str, str]] = [
    ("rex", "rex.cli"),
    ("rex-config", "rex.config"),
    ("rex-speak-api", "rex_speak_api"),
    ("rex-agent", "rex.computers.agent_server"),
    ("rex-gui", "rex.gui_app"),
    ("rex-tool-server", "rex.openclaw.tool_server"),
]


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {text}")


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> bool:
    """Verify Python version is 3.10+"""
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        print_success(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    print_error(f"Python {v.major}.{v.minor} (requires 3.10+)")
    return False


def check_core_files() -> bool:
    """Verify essential source files are present."""
    paths = [
        "rex/assistant_errors.py",
        "rex/config.py",
        "rex_speak_api.py",
        "pyproject.toml",
        ".gitignore",
    ]
    results = []
    for p in paths:
        full = PROJECT_ROOT / p
        if full.exists():
            print_success(p)
            results.append(True)
        else:
            print_error(f"{p} NOT FOUND")
            results.append(False)
    return all(results)


def check_config_json() -> bool:
    """Verify config/rex_config.json exists and is valid JSON with required keys."""
    config_path = PROJECT_ROOT / "config" / "rex_config.json"

    if not config_path.exists():
        print_error("config/rex_config.json not found")
        print_warning("  Copy config/rex_config.example.json to config/rex_config.json")
        return False

    print_success("config/rex_config.json exists")

    try:
        with config_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print_error(f"config/rex_config.json is not valid JSON: {exc}")
        return False

    print_success("config/rex_config.json is valid JSON")

    # Basic schema sanity: must be an object (dict)
    if not isinstance(data, dict):
        print_error("config/rex_config.json must be a JSON object at the top level")
        return False

    print_success(f"  Keys present: {', '.join(list(data.keys())[:8])}")
    return True


def check_dependencies() -> bool:
    """Verify core importable packages."""
    packages = ["flask", "pydantic", "dotenv", "requests", "cryptography"]
    results = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print_success(f"import {pkg}")
            results.append(True)
        except ImportError as exc:
            print_error(f"import {pkg} — {exc}")
            results.append(False)
    return all(results)


def check_pytorch() -> bool:
    """Verify PyTorch is installed and within the supported version range."""
    try:
        import torch
    except ImportError:
        print_error("PyTorch not installed (optional — required for voice/ML features)")
        return False

    version_str: str = torch.__version__
    print_success(f"PyTorch {version_str}")

    # Parse major.minor.patch (ignore +cu* suffix)
    raw = version_str.split("+")[0].split(".")
    try:
        parts = tuple(int(x) for x in raw[:3])
    except ValueError:
        print_warning(f"Could not parse PyTorch version: {version_str}")
        return False

    min_ok = parts >= TORCH_MIN
    max_ok = parts < TORCH_MAX

    if min_ok and max_ok:
        print_success(
            f"PyTorch version {version_str} is within supported range "
            f">={'.'.join(str(x) for x in TORCH_MIN)},<{'.'.join(str(x) for x in TORCH_MAX)}"
        )
    else:
        print_error(
            f"PyTorch {version_str} is outside supported range "
            f">={'.'.join(str(x) for x in TORCH_MIN)},<{'.'.join(str(x) for x in TORCH_MAX)}"
        )
        return False

    cuda_available: bool = torch.cuda.is_available()
    if cuda_available:
        device = torch.cuda.get_device_name(0)
        print_success(f"CUDA available: {device}")
    else:
        print_warning("CUDA not available (CPU mode)")

    return True


def check_config_loads() -> bool:
    """Verify rex.config can be imported and settings load without error."""
    try:
        from rex.config import AppConfig

        AppConfig()
        print_success("rex.config.AppConfig instantiates successfully")
        return True
    except Exception as exc:  # pragma: no cover
        print_error(f"rex.config.AppConfig failed: {exc}")
        return False


def check_cli_entrypoints() -> bool:
    """Verify all CLI entrypoints declared in pyproject.toml are importable."""
    results = []
    for name, module in CLI_ENTRYPOINTS:
        try:
            importlib.import_module(module)
            print_success(f"{name!r} → import {module}")
            results.append(True)
        except ImportError as exc:
            print_error(f"{name!r} → import {module} FAILED: {exc}")
            results.append(False)
        except Exception as exc:  # pragma: no cover
            print_warning(f"{name!r} → import {module} raised {type(exc).__name__}: {exc}")
            results.append(False)
    return all(results)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_validation() -> int:
    """Run all 7 validation checks and return exit code."""
    print_header("REX AI ASSISTANT — DEPLOYMENT VALIDATION")

    checks = [
        ("python_version", "1. Python Version", check_python_version),
        ("core_files", "2. Core Files", check_core_files),
        ("config_json", "3. Runtime Config (rex_config.json)", check_config_json),
        ("dependencies", "4. Core Python Dependencies", check_dependencies),
        ("pytorch", "5. PyTorch Version", check_pytorch),
        ("config_loads", "6. Config Module Loading", check_config_loads),
        ("entrypoints", "7. CLI Entrypoints Importable", check_cli_entrypoints),
    ]

    results = {}
    for key, header, fn in checks:
        print_header(header)
        results[key] = fn()

    print_header("VALIDATION SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)  # always 7

    for key, value in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if value else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"{key.upper():.<30} {status}")

    print(f"\n{Colors.BLUE}Score: {passed}/{total} checks passed{Colors.RESET}\n")

    if passed == total:
        print(f"{Colors.GREEN}All {total} checks passed. Deployment is valid.{Colors.RESET}")
        return 0

    if passed >= int(total * 0.8):
        print(
            f"{Colors.YELLOW}Most checks passed ({passed}/{total}). Review warnings above.{Colors.RESET}"
        )
        return 1

    print(
        f"{Colors.RED}Deployment validation FAILED ({passed}/{total}). Fix errors above.{Colors.RESET}"
    )
    return 2


if __name__ == "__main__":
    try:
        sys.exit(run_validation())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted{Colors.RESET}")
        sys.exit(130)
    except Exception as exc:  # pragma: no cover
        print(f"\n{Colors.RED}Validation error: {exc}{Colors.RESET}")
        sys.exit(1)
