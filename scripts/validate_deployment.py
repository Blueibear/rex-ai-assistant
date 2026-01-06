"""Rex AI Assistant - Post-Deployment Validation Script

Run this after deploying stabilization files to verify everything works.
"""

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


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


def load_env_from_file() -> None:
    env_path = PROJECT_ROOT / '.env'
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def check_python_version() -> bool:
    """Verify Python version is 3.10+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    print_error(f"Python {version.major}.{version.minor} (requires 3.10+)")
    return False


def check_file_exists(path: str) -> bool:
    """Check if a file exists"""
    if Path(path).exists():
        print_success(path)
        return True
    print_error(f"{path} NOT FOUND")
    return False


def check_import(module_name: str) -> bool:
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print_success(f"import {module_name}")
        return True
    except ImportError as exc:
        print_error(f"import {module_name} - {exc}")
        return False


def check_env_file() -> bool:
    """Check .env file exists and contains key variables"""
    env_path = PROJECT_ROOT / '.env'
    if not env_path.exists():
        print_error('.env file not found')
        return False

    print_success('.env file exists')

    required_vars = [
        'REX_SPEAK_API_KEY',
        'REX_ACTIVE_USER',
        'REX_WAKEWORD',
    ]

    env_content = env_path.read_text(encoding='utf-8')
    missing = [var for var in required_vars if var not in env_content]

    if missing:
        print_warning(f"Missing variables: {', '.join(missing)}")
        return False

    print_success('All critical variables present')
    return True


def check_pytorch() -> bool:
    """Verify PyTorch installation and CUDA"""
    try:
        import torch  # type: ignore
    except ImportError:
        print_error('PyTorch not installed')
        return False

    version = torch.__version__
    cuda_available = torch.cuda.is_available()  # type: ignore[attr-defined]

    print_success(f'PyTorch {version}')

    if cuda_available:
        device = torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
        print_success(f'CUDA available: {device}')
    else:
        print_warning('CUDA not available (CPU mode)')

    if version.startswith('2.5'):
        print_success('PyTorch version correct (2.5.x)')
        return True

    print_warning(f'PyTorch {version} (expected 2.5.x)')
    return False


def check_config_loads() -> bool:
    """Test configuration loading"""
    try:
        from rex.config import settings  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print_error(f'Config loading failed: {exc}')
        return False

    print_success('Config loads successfully')
    if getattr(settings, 'wakeword', None):
        print_success(f'  Wake word: {settings.wakeword}')
    if getattr(settings, 'llm_model', None):
        print_success(f'  LLM model: {settings.llm_model}')
    return True


def check_circular_imports() -> bool:
    """Verify no circular import issues"""
    test_imports = [
        'rex.assistant_errors',
        'rex.config',
        'llm_client',
        'memory_utils',
    ]

    results = []
    for module in test_imports:
        try:
            importlib.import_module(module)
            print_success(module)
            results.append(True)
        except ImportError as exc:
            print_error(f'{module} - {exc}')
            results.append(False)

    return all(results)


def run_validation() -> int:
    """Run all validation checks"""
    print_header('REX AI ASSISTANT - VALIDATION SCRIPT')
    print(f'Working directory: {os.getcwd()}\n')

    load_env_from_file()

    results = {}

    print_header('1. Python Version')
    results['python'] = check_python_version()

    print_header('2. Core Files')
    core_files = [
        'rex/assistant_errors.py',
        'rex/config.py',
        'rex_speak_api.py',
        'requirements.txt',
        '.gitignore',
    ]
    results['files'] = all(check_file_exists(path) for path in core_files)

    print_header('3. Environment Configuration')
    results['env'] = check_env_file()

    print_header('4. Python Dependencies')
    deps = ['flask', 'torch', 'transformers', 'sounddevice', 'whisper']
    results['deps'] = all(check_import(dep) for dep in deps)

    print_header('5. PyTorch & CUDA')
    results['pytorch'] = check_pytorch()

    print_header('6. Configuration Loading')
    results['config'] = check_config_loads()

    print_header('7. Import Structure')
    results['imports'] = check_circular_imports()

    print_header('VALIDATION SUMMARY')
    passed = sum(1 for value in results.values() if value)
    total = len(results)

    for name, value in results.items():
        status = f'{Colors.GREEN}PASS{Colors.RESET}' if value else f'{Colors.RED}FAIL{Colors.RESET}'
        print(f'{name.upper():.<20} {status}')

    print(f"\n{Colors.BLUE}Score: {passed}/{total} checks passed{Colors.RESET}\n")

    if passed == total:
        print(f'{Colors.GREEN}All checks passed! Deployment successful.{Colors.RESET}')
        return 0
    if passed >= int(total * 0.8):
        print(f'{Colors.YELLOW}Most checks passed. Review warnings above.{Colors.RESET}')
        return 1
    print(f'{Colors.RED}Deployment validation failed. Fix errors above.{Colors.RESET}')
    return 2


if __name__ == '__main__':
    try:
        sys.exit(run_validation())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted{Colors.RESET}")
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"\n{Colors.RED}Validation error: {exc}{Colors.RESET}")
        sys.exit(1)
