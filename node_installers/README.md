# Rex Node Installers

This directory contains scripts and templates to deploy a **lean Rex node** for remote environments.

## Included files
- `install_debian.sh`: Lean node installer for Debian/Ubuntu systems.
- `install_windows.ps1`: Lean node installer for Windows.
- `.env.node`: Environment template for registering the node.

## Environment variables
Update `.env.node` with the gateway URL and registration token before running installers.

| Variable | Description |
| --- | --- |
| `REX_GATEWAY_URL` | Base URL for the Rex gateway. |
| `REX_NODE_ID` | Identifier for this node. |
| `REX_NODE_TOKEN` | Registration token (stubbed). |
| `REX_NODE_ROLE` | Role label (`lean`). |
| `REX_HEALTH_PORT` | Health check port for the local service. |

## Debian installer
```bash
cd node_installers
./install_debian.sh
```

Optional overrides:
```bash
REX_ROOT=/opt/rex-node \
REX_PACKAGE_SOURCE=/path/to/rex_ai_assistant.whl \
REX_SERVICE_PORT=8765 \
REX_SERVICES=event_bus,workflow_runner,memory_store,credential_manager \
./install_debian.sh
```

## Windows installer
Open PowerShell as Administrator:
```powershell
cd node_installers
.\install_windows.ps1 -RexRoot "C:\RexNode" -PackageSource rex-ai-assistant
```

Use `-DryRun` to preview actions.

## Manual registration (stub)
Once installed, register the node with the gateway:
```bash
curl -X POST "$REX_GATEWAY_URL/api/nodes/register" \
  -H "Authorization: Bearer $REX_NODE_TOKEN"
```

```powershell
Invoke-RestMethod -Method Post -Uri $env:REX_GATEWAY_URL/api/nodes/register \
  -Headers @{Authorization="Bearer $env:REX_NODE_TOKEN"}
```
