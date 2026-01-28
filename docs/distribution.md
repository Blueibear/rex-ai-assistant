# Distribution & Installation

## Installation options

### Pip installation
```bash
pip install .
pip install .[sms,devtools]
```

### Full installation script
```bash
./install_full.sh
```

Environment overrides:
```bash
REX_SERVICE_PORT=8765 REX_SKIP_SERVICE=1 ./install_full.sh
```

### Lean installation script
```bash
./install_lean.sh
```

Environment overrides:
```bash
REX_SERVICES=event_bus,workflow_runner,memory_store,credential_manager ./install_lean.sh
```

## Optional dependencies
- `sms`: Twilio support (`twilio` package).
- `devtools`: Build and lint tooling (`build`, `ruff`, `black`, `mypy`).

## Remote node setup
Lean nodes run a trimmed service set and register with the Rex gateway.

See `node_installers/README.md` for platform-specific installer instructions.

## Customization
- Use `--services` with `rex.app` to control which services are supervised.
- Adjust health check port with `--port`.
- Modify `.env.node` for gateway registration details.
