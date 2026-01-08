# Dependency Management Guide

## Overview

Rex AI Assistant uses **pinned dependencies** in `requirements.txt` for reproducible builds and security. All versions are explicitly specified with `==` to enable GitHub Dependabot to track and propose updates for vulnerable packages.

## Files

- **`requirements.txt`**: Pinned versions for production deployment
- **`requirements.in`**: (Optional) Top-level dependencies with semver ranges

## Updating Dependencies

### Manual Updates

1. **Edit `requirements.txt`** with new version numbers
2. **Test thoroughly:**
   ```bash
   pip install -r requirements.txt
   python -m compileall .
   pytest
   python -c "import rex; from rex.llm_client import *"
   ```
3. **Commit changes** with explanation of what was updated and why

### Security Updates

When GitHub Dependabot alerts you to vulnerabilities:

1. Check the **recommended version** in the Dependabot alert
2. Update `requirements.txt` to that version or higher
3. **Test for breaking changes** (see Testing section below)
4. Review security advisory to understand the risk
5. Commit with reference to CVE/GHSA ID

## Important Constraints

### PyTorch (CPU-only builds)

Rex uses CPU-only PyTorch builds for compatibility:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Version constraints:**
- `torch`, `torchvision`, `torchaudio` must be compatible versions
- Check [PyTorch version compatibility](https://pytorch.org/get-started/previous-versions/)

### Transformers + TTS Compatibility

**Issue:** Hugging Face Transformers 4.38+ moved `BeamSearchScorer` from top-level exports to internal modules. Coqui TTS still expects it at top level.

**Solution:** Compatibility shim in `rex/compat/transformers_shims.py`

- Automatically patches `transformers.BeamSearchScorer` when `rex` is imported
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
1. Ensure `requirements.txt` uses exact pins
2. Enable Dependabot in repository settings
3. Review and merge Dependabot PRs after testing

## Version History

| Date       | Transformers | Torch | TTS   | Notes                          |
|------------|--------------|-------|-------|--------------------------------|
| 2026-01-08 | 4.57.3       | 2.9.1 | 0.22.0| Security fixes, shim added     |
| 2025-12-XX | 4.37.2       | 2.7.1 | 0.18.0| Pre-security audit             |

## Questions?

See [README_STABILIZATION.md](README_STABILIZATION.md) for configuration details or open an issue on GitHub.
