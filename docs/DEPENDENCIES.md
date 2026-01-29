# Dependency Management Guide

## Overview

Rex AI Assistant splits dependencies by install target so CPU-only installs and CI do not pull CUDA/GPU wheels. Base installs come from `pyproject.toml`, while optional ML/audio stacks live in split requirements files.

## Files

- **`requirements.txt`**: Pointer file with guidance (not used for installs)
- **`requirements-cpu.txt`**: CPU-only ML + audio stack (no CUDA index)
- **`requirements-gpu-cu124.txt`**: CUDA 12.4 GPU stack (Windows RTX 3060)
- **`requirements-gpu.txt`**: CUDA 11.8 GPU stack (Linux)
- **`requirements-dev.txt`**: Dev tooling via extras
- **`requirements.in`**: Top-level dependencies with semver ranges (optional)
- **`Pipfile`**: Pipenv dependency specification (exact pins for Dependabot)
- **`Pipfile.lock`**: Pipenv lockfile with all transitive dependencies (for Dependabot)

**Note:** `Pipfile.lock` enables GitHub Dependabot to automatically detect and propose updates for vulnerable dependencies.

## Updating Dependencies

### Manual Updates

1. **Edit the relevant requirements file** with new version numbers
2. **Test thoroughly:**
   ```bash
   pip install .
   pip install -r requirements-cpu.txt
   pip install -e ".[dev]"
   python -m compileall .
   pytest
   python -c "import rex; from rex.llm_client import *"
   ```
3. **Commit changes** with explanation of what was updated and why

### Security Updates

When GitHub Dependabot alerts you to vulnerabilities:

1. Check the **recommended version** in the Dependabot alert
2. Update the relevant requirements file to that version or higher
3. **Test for breaking changes** (see Testing section below)
4. Review security advisory to understand the risk
5. Commit with reference to CVE/GHSA ID

## Important Constraints

### PyTorch (CPU-only builds)

Rex uses CPU-only PyTorch builds for compatibility:

```bash
pip install -r requirements-cpu.txt
```

**Version constraints:**
- `torch`, `torchvision`, `torchaudio` must be compatible versions
- Check [PyTorch version compatibility](https://pytorch.org/get-started/previous-versions/)

### Transformers + TTS Compatibility

**Issue:** Hugging Face Transformers 4.38+ moved `BeamSearchScorer` from top-level exports to internal modules. Coqui TTS still expects it at top level.

**Solution:** Compatibility shim in `rex/compat/transformers_shims.py`

- Call `ensure_transformers_compatibility()` before importing libraries that require it
- Allows upgrading to transformers 4.48+ for security fixes
- No changes needed to TTS library code

**Testing the shim:**
```bash
python -c "from transformers import BeamSearchScorer; print('OK')"
```

## Testing After Updates

### Minimal Smoke Test
```bash
python -c "import rex; from rex.llm_client import *; print('Imports OK')"
```

### Full Test Suite
```bash
python -m compileall .  # Check all Python files compile
pytest                  # Run unit tests
python gui.py           # Test GUI loads
```

### Integration Test
1. Start the assistant
2. Test wake word detection
3. Test speech-to-text
4. Test LLM response generation
5. Test text-to-speech output

## Security Best Practices

1. **Subscribe to Dependabot alerts** in GitHub repository settings
2. **Review changelogs** before major version bumps
3. **Test breaking changes** in development environment first
4. **Pin transitive dependencies** if they cause issues
5. **Document compatibility constraints** in this file

## Known Compatibility Issues

### Resolved

✅ **transformers BeamSearchScorer** - Shim in `rex/compat/transformers_shims.py`

### Active Constraints

- **torch**: Must use CPU builds (no CUDA)
- **TTS**: Requires transformers compatibility shim
- **numpy**: Keep <2.0 for compatibility with older scientific packages

## Dependabot Configuration

GitHub Dependabot requires exact pins (`==`) to function. It will:
- Detect outdated dependencies
- Propose version bumps via pull requests
- Flag security vulnerabilities

**To enable Dependabot:**
1. Ensure the split requirements files use exact pins where needed
2. Enable Dependabot in repository settings
3. Review and merge Dependabot PRs after testing

## Version History

| Date       | Transformers | Torch | TTS   | Notes                          |
|------------|--------------|-------|-------|--------------------------------|
| 2026-01-09 | 4.57.3       | 2.6.0 | 0.22.0| Pin CUDA 12.4, fix torchvision constraint |
| 2026-01-09 | 4.57.3       | 2.8.0 | 0.22.0| 15 CVEs fixed, Pipfile.lock added |
| 2026-01-08 | 4.57.3       | 2.9.1 | 0.22.0| Security fixes, shim added     |
| 2025-12-XX | 4.37.2       | 2.7.1 | 0.18.0| Pre-security audit             |

## Questions?

See [README_STABILIZATION.md](README_STABILIZATION.md) for configuration details or open an issue on GitHub.


## Using Pipenv (Optional)

Pipenv provides an alternative dependency management workflow with lockfile support. The `Pipfile.lock` is primarily used by GitHub Dependabot to track updates.

### Installation with Pipenv

```bash
# Install pipenv
pip install pipenv

# Install dependencies from Pipfile
pipenv install

# Install dev dependencies
pipenv install --dev

# Activate virtual environment
pipenv shell

# Run commands in pipenv environment
pipenv run python gui.py
```

### Updating Dependencies with Pipenv

```bash
# Update a specific package
pipenv update flask

# Update all packages
pipenv update

# Lock dependencies after manual edit to Pipfile
pipenv lock

# Generate CPU requirements from Pipfile.lock (if using Pipenv)
pipenv requirements > requirements-cpu.txt
```

### Keeping Pipfile and split requirements in Sync

**Primary workflow:** Edit the split requirements files directly for dependency updates.

To sync Pipfile when requirements files change:

```bash
# Update Pipfile manually or regenerate from requirements-cpu.txt
pipenv install -r requirements-cpu.txt --skip-lock

# Lock dependencies
pipenv lock
```

**Note:** `requirements.txt` is a pointer file. Pipfile/Pipfile.lock exist primarily to enable Dependabot.
