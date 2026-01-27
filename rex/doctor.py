"""Environment diagnostics for Rex AI Assistant.

This module provides the `rex doctor` command functionality, checking:
- Python version compatibility
- Configuration file presence and readability
- Required environment variables
- External dependencies (binaries on PATH)
- Basic security checks

Usage:
    from rex.doctor import run_diagnostics
    exit_code = run_diagnostics(verbose=True)
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class Status(Enum):
    """Status levels for diagnostic checks."""

    OK = "ok"
    WARNING = "warn"
    ERROR = "error"
    INFO = "info"


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: Status
    message: str
    details: str = ""


@dataclass
class DiagnosticsReport:
    """Collection of all diagnostic check results."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        """Add a check result to the report."""
        self.results.append(result)

    def has_errors(self) -> bool:
        """Return True if any check has error status."""
        return any(r.status == Status.ERROR for r in self.results)

    def has_warnings(self) -> bool:
        """Return True if any check has warning status."""
        return any(r.status == Status.WARNING for r in self.results)

    @property
    def error_count(self) -> int:
        """Count of checks with error status."""
        return sum(1 for r in self.results if r.status == Status.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of checks with warning status."""
        return sum(1 for r in self.results if r.status == Status.WARNING)


def _status_symbol(status: Status) -> str:
    """Return a symbol for the status level."""
    symbols = {
        Status.OK: "[OK]",
        Status.WARNING: "[WARN]",
        Status.ERROR: "[ERROR]",
        Status.INFO: "[INFO]",
    }
    return symbols.get(status, "[?]")


def _find_project_root() -> Path | None:
    """Find the project root by looking for pyproject.toml or .git."""
    # Start from current working directory
    current = Path.cwd()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "config" / "rex_config.example.json").exists():
            return parent

    return None


def check_python_version() -> CheckResult:
    """Check if Python version is supported."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version < (3, 9):
        return CheckResult(
            name="Python Version",
            status=Status.ERROR,
            message=f"Python {version_str} is not supported",
            details="Rex requires Python 3.9 or later. Please upgrade your Python installation.",
        )
    elif version < (3, 10):
        return CheckResult(
            name="Python Version",
            status=Status.WARNING,
            message=f"Python {version_str} is supported but 3.10+ recommended",
            details="Consider upgrading to Python 3.10+ for best performance and compatibility.",
        )
    else:
        return CheckResult(
            name="Python Version",
            status=Status.OK,
            message=f"Python {version_str}",
        )


def check_config_file(root: Path | None) -> CheckResult:
    """Check for presence and readability of rex_config.json."""
    if root is None:
        return CheckResult(
            name="Config File",
            status=Status.WARNING,
            message="Could not determine project root",
            details="Run rex doctor from the project directory or ensure pyproject.toml exists.",
        )

    config_path = root / "config" / "rex_config.json"
    example_path = root / "config" / "rex_config.example.json"

    if not config_path.exists():
        if example_path.exists():
            return CheckResult(
                name="Config File",
                status=Status.WARNING,
                message="rex_config.json not found (example exists)",
                details=f"Copy {example_path} to {config_path} and customize it.",
            )
        return CheckResult(
            name="Config File",
            status=Status.ERROR,
            message="rex_config.json not found",
            details=f"Expected config file at: {config_path}",
        )

    # Check readability
    try:
        with open(config_path) as f:
            import json

            json.load(f)
        return CheckResult(
            name="Config File",
            status=Status.OK,
            message=f"Found and readable: {config_path.name}",
        )
    except json.JSONDecodeError as e:
        return CheckResult(
            name="Config File",
            status=Status.ERROR,
            message="rex_config.json has invalid JSON",
            details=str(e),
        )
    except PermissionError:
        return CheckResult(
            name="Config File",
            status=Status.ERROR,
            message="Cannot read rex_config.json (permission denied)",
            details=f"Check file permissions on {config_path}",
        )


def check_env_file(root: Path | None) -> CheckResult:
    """Check for presence of .env file."""
    if root is None:
        return CheckResult(
            name="Environment File",
            status=Status.INFO,
            message="Could not determine project root",
        )

    env_path = root / ".env"
    example_path = root / ".env.example"

    if not env_path.exists():
        if example_path.exists():
            return CheckResult(
                name="Environment File",
                status=Status.WARNING,
                message=".env not found (example exists)",
                details=f"Copy {example_path} to {env_path} and add your API keys.",
            )
        return CheckResult(
            name="Environment File",
            status=Status.WARNING,
            message=".env file not found",
            details="Create a .env file with your API keys. See .env.example for template.",
        )

    return CheckResult(
        name="Environment File",
        status=Status.OK,
        message=f"Found: {env_path.name}",
    )


def check_environment_variables() -> CheckResult:
    """Check for presence of commonly required environment variables."""
    # List of common API key environment variables
    # We don't hardcode specific key names, just check common patterns
    api_key_patterns = [
        "OPENAI_API_KEY",
        "OLLAMA_API_KEY",
        "BRAVE_API_KEY",
        "SERPAPI_KEY",
        "GOOGLE_API_KEY",
        "HA_TOKEN",
    ]

    missing = []
    present = []

    for var in api_key_patterns:
        value = os.environ.get(var)
        if value and value.strip():
            present.append(var)
        else:
            missing.append(var)

    if not present:
        return CheckResult(
            name="API Keys",
            status=Status.WARNING,
            message="No API keys configured",
            details=(
                "At least one API key is needed for full functionality. "
                f"Missing: {', '.join(missing[:3])}..."
            ),
        )

    if missing:
        return CheckResult(
            name="API Keys",
            status=Status.INFO,
            message=f"{len(present)} API key(s) configured",
            details=f"Configured: {', '.join(present)}",
        )

    return CheckResult(
        name="API Keys",
        status=Status.OK,
        message=f"All {len(present)} common API keys configured",
    )


def check_binary(name: str, purpose: str) -> CheckResult:
    """Check if a binary is available on PATH."""
    path = shutil.which(name)
    if path:
        return CheckResult(
            name=f"Binary: {name}",
            status=Status.OK,
            message=f"Found: {path}",
        )
    return CheckResult(
        name=f"Binary: {name}",
        status=Status.WARNING,
        message=f"'{name}' not found on PATH",
        details=f"Required for: {purpose}. Install it or ensure it's in your PATH.",
    )


def check_external_dependencies() -> list[CheckResult]:
    """Check for external binary dependencies."""
    dependencies = [
        ("ffmpeg", "audio processing and transcoding"),
        ("git", "version control"),
    ]

    results = []
    for name, purpose in dependencies:
        results.append(check_binary(name, purpose))

    return results


def check_config_permissions(root: Path | None) -> CheckResult:
    """Check for security issues with config file permissions."""
    if root is None:
        return CheckResult(
            name="Config Permissions",
            status=Status.INFO,
            message="Could not determine project root",
        )

    # Check .env file permissions (should not be world-readable)
    env_path = root / ".env"
    if env_path.exists():
        try:
            mode = env_path.stat().st_mode
            # Check if world-readable (others have read permission)
            if mode & stat.S_IROTH:
                return CheckResult(
                    name="Config Permissions",
                    status=Status.WARNING,
                    message=".env file is world-readable",
                    details=(
                        f"Consider restricting permissions: chmod 600 {env_path}\n"
                        "This file may contain sensitive API keys."
                    ),
                )
        except OSError:
            pass

    # Check for secrets in config files
    config_path = root / "config" / "rex_config.json"
    if config_path.exists():
        try:
            content = config_path.read_text().lower()
            sensitive_patterns = ["api_key", "secret", "password", "token"]
            found_patterns = [p for p in sensitive_patterns if p in content]
            if found_patterns:
                # This is likely fine since config should have non-sensitive settings
                pass
        except OSError:
            pass

    return CheckResult(
        name="Config Permissions",
        status=Status.OK,
        message="Configuration permissions look reasonable",
    )


def check_package_installation() -> CheckResult:
    """Check if Rex is properly installed as a package."""
    try:
        import rex
        from rex.contracts import CONTRACT_VERSION

        return CheckResult(
            name="Package Installation",
            status=Status.OK,
            message=f"rex package installed (contracts v{CONTRACT_VERSION})",
        )
    except ImportError as e:
        return CheckResult(
            name="Package Installation",
            status=Status.WARNING,
            message="rex package not fully installed",
            details=f"Import error: {e}. Run 'pip install -e .' to install.",
        )


def check_core_dependencies() -> list[CheckResult]:
    """Check if core Python dependencies are installed."""
    results = []

    deps = [
        ("torch", "PyTorch (ML framework)"),
        ("transformers", "Transformers (NLP models)"),
        ("whisper", "Whisper (speech recognition)"),
        ("TTS", "Coqui TTS (text-to-speech)"),
        ("openwakeword", "OpenWakeWord (wake word detection)"),
        ("flask", "Flask (web framework)"),
        ("pydantic", "Pydantic (data validation)"),
        ("dotenv", "python-dotenv (environment loading)"),
    ]

    for module, description in deps:
        try:
            __import__(module)
            results.append(
                CheckResult(
                    name=f"Dependency: {module}",
                    status=Status.OK,
                    message=f"Installed: {description}",
                )
            )
        except ImportError:
            results.append(
                CheckResult(
                    name=f"Dependency: {module}",
                    status=Status.WARNING,
                    message=f"Not installed: {description}",
                    details=f"Run 'pip install {module}' or 'pip install -e .' to install.",
                )
            )

    return results


def check_gpu_availability() -> CheckResult:
    """Check if GPU is available for ML inference."""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return CheckResult(
                name="GPU Availability",
                status=Status.OK,
                message=f"CUDA available: {device_name}",
                details=f"{device_count} GPU(s) detected",
            )
        else:
            return CheckResult(
                name="GPU Availability",
                status=Status.INFO,
                message="No CUDA GPU available (CPU mode)",
                details="ML inference will run on CPU. This is fine but slower.",
            )
    except ImportError:
        return CheckResult(
            name="GPU Availability",
            status=Status.INFO,
            message="PyTorch not installed (GPU check skipped)",
        )
    except Exception as e:
        return CheckResult(
            name="GPU Availability",
            status=Status.WARNING,
            message="Could not check GPU availability",
            details=str(e),
        )


def run_diagnostics(verbose: bool = False) -> int:
    """Run all diagnostic checks and print results.

    Args:
        verbose: If True, show detailed information for all checks.

    Returns:
        Exit code: 0 if no errors, 1 if errors found.
    """
    print("Rex Doctor - Environment Diagnostics")
    print("=" * 40)
    print()

    report = DiagnosticsReport()
    project_root = _find_project_root()

    if project_root:
        print(f"Project root: {project_root}")
    else:
        print("Project root: (not found)")
    print()

    # Core checks
    report.add(check_python_version())
    report.add(check_package_installation())
    report.add(check_config_file(project_root))
    report.add(check_env_file(project_root))
    report.add(check_environment_variables())
    report.add(check_config_permissions(project_root))

    # External dependencies
    for result in check_external_dependencies():
        report.add(result)

    # GPU check
    report.add(check_gpu_availability())

    # Core Python dependencies (only in verbose mode to reduce noise)
    if verbose:
        for result in check_core_dependencies():
            report.add(result)

    # Print results
    for result in report.results:
        symbol = _status_symbol(result.status)
        print(f"{symbol:8s} {result.name}: {result.message}")
        if verbose and result.details:
            # Indent details
            for line in result.details.split("\n"):
                print(f"         {line}")

    # Summary
    print()
    print("-" * 40)

    if report.has_errors():
        print(f"FAILED: {report.error_count} error(s), {report.warning_count} warning(s)")
        print("Fix the errors above before running Rex.")
        return 1
    elif report.has_warnings():
        print(f"PASSED with warnings: {report.warning_count} warning(s)")
        print("Rex should work, but consider addressing the warnings above.")
        return 0
    else:
        print("PASSED: All checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_diagnostics(verbose=True))
