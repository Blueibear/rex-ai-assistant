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

`install_windows.ps1` normalizes `-RexRoot` to an absolute path before it builds the virtual-environment and service-registration paths. The Windows service is installed and started with the fully qualified `<RexRoot>\venv\Scripts\python.exe`; paths containing spaces are quoted, and a real registration fails closed if that interpreter is missing. This remains true when the installer is invoked from a working directory outside the repository or when `-RexRoot` is supplied as a relative path.

Use `-DryRun` to preview the exact normalized interpreter path and service commands without creating or registering the service. CI exercises that dry run on a Windows runner with a relative root containing spaces, and the packaged Windows artifact gate verifies the installed application without relying on machine Python or Node.

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