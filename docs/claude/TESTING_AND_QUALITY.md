# Claude Reference: Testing and Quality

This file is reference material.
Use it when a task touches tests, CI, lint, formatting, typing, repo integrity, or verification rules.

## Pytest source of truth
- Pytest configuration source of truth is `[tool.pytest.ini_options]` in `pyproject.toml`
- Do not reintroduce `pytest.ini`
- `pytest -q` should work after a normal editable install without requiring coverage plugins

## Default local validation
```bash
pytest -q
python -m rex --help
python scripts/doctor.py
python scripts/security_audit.py