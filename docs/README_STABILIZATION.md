# Historical Stabilization Note

This file used to contain a 2025 stabilization handoff that instructed readers
to copy generated files from an `outputs/` directory and pin older dependency
versions. That workflow is no longer current.

Current sources of truth:

- `README.md` for overview and quick start
- `INSTALL.md` for setup
- `RUNNING.md` for runtime modes
- `CONFIGURATION.md` for runtime configuration
- `docs/DEPLOYMENT_CHECKLIST.md` for deployment checks
- `docs/runbook.md` for day-2 operations
- `docs/ARCHITECTURE.md` for architecture

Do not follow old instructions that mention:

- copying files from `outputs/`
- replacing `rex/config.py`, `rex/assistant_errors.py`, or `rex_speak_api.py`
- installing `torch==2.5.1` as the current recommended GPU path
- posting to `http://localhost:5000/speak`
- running `python scripts/doctor.py` as the primary doctor command

Current quick checks:

```bash
python -m rex --help
python -m rex doctor
python -m rex tools --all
```

Current TTS API default:

```text
http://127.0.0.1:5005/speak
```
