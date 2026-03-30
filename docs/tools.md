# Tool Registry

Rex uses a centralized tool registry to manage available tools, their metadata, and health checks. This document explains the tool registry system and how to register new tools.

> **Migration note:** The `rex/tool_registry.py` and `rex/tool_router.py` modules are marked `# OPENCLAW-REPLACE` and are frozen (no new features). They will be replaced by OpenClaw's tool system in a future migration phase. The public API documented here remains valid until that migration is complete.

## Overview

The `ToolRegistry` provides:
- Structured metadata about available tools
- Health check functionality to verify tool availability
- Credential requirement tracking
- Integration with the tool router for pre-execution validation

## CLI Command

View all registered tools and their status:

```bash
# List all tools with status
rex tools

# Verbose output with details
rex tools -v

# Include disabled tools
rex tools -a
```

Example output:

```
Rex Tool Registry
============================================================

time_now [READY]
  Description: Get the current time for a specified location

weather_now [NO CREDS]
  Description: Get current weather conditions for a location

send_email [NO CREDS]
  Description: Send an email message

Total: 3 tools, 1 ready
```

### Status Icons

| Icon         | Meaning                                      |
|--------------|----------------------------------------------|
| `[READY]`    | Tool is enabled, healthy, and has credentials|
| `[NO CREDS]` | Tool is missing required credentials         |
| `[UNHEALTHY]`| Tool's health check failed                   |
| `[DISABLED]` | Tool is disabled                             |

## Registering New Tools

### Basic Registration

```python
from rex.openclaw.tool_registry import ToolMeta, register_tool

# Register a simple tool
register_tool(ToolMeta(
    name="my_tool",
    description="Does something useful",
))
```

### Full Registration with Metadata

```python
from rex.openclaw.tool_registry import ToolMeta, register_tool

def my_health_check() -> tuple[bool, str]:
    """Check if the service is available."""
    try:
        # Check service availability
        return True, "Service is running"
    except Exception as e:
        return False, f"Service unavailable: {e}"

register_tool(ToolMeta(
    name="weather_api",
    description="Get weather data from external API",
    required_credentials=["weather_api_key"],
    capabilities=["read", "network", "weather"],
    health_check=my_health_check,
    version="1.0.0",
    enabled=True,
))
```

### ToolMeta Fields

| Field                 | Type                      | Description                                    |
|-----------------------|---------------------------|------------------------------------------------|
| `name`                | `str`                     | Unique identifier for the tool (required)      |
| `description`         | `str`                     | Human-readable description (required)          |
| `required_credentials`| `list[str]`               | Service names needed from CredentialManager    |
| `capabilities`        | `list[str]`               | Capability tags (e.g., "read", "write")        |
| `health_check`        | `() -> tuple[bool, str]`  | Function that checks tool availability         |
| `version`             | `str`                     | Tool version (default: "1.0.0")                |
| `enabled`             | `bool`                    | Whether the tool is active (default: True)     |

## Health Checks

Health checks verify that a tool is operational before use.

### Writing a Health Check

```python
def my_health_check() -> tuple[bool, str]:
    """
    Returns:
        Tuple of (ok: bool, message: str)
        - ok: True if the tool is healthy
        - message: Description of health status
    """
    try:
        # Check external service
        response = check_service()
        if response.ok:
            return True, "Service responding normally"
        return False, f"Service error: {response.status}"
    except ConnectionError:
        return False, "Cannot connect to service"
    except Exception as e:
        return False, f"Health check failed: {e}"
```

### Running Health Checks

```python
from rex.openclaw.tool_registry import get_tool_registry

registry = get_tool_registry()

# Check single tool
ok, message = registry.check_health("my_tool")

# Check all tools
results = registry.check_all_health()
for name, (ok, message) in results.items():
    status = "OK" if ok else "FAIL"
    print(f"{name}: {status} - {message}")
```

## Credential Integration

Tools can declare required credentials that are checked before execution:

```python
from rex.openclaw.tool_registry import ToolMeta, get_tool_registry

# Register a tool that needs credentials
register_tool(ToolMeta(
    name="email_sender",
    description="Send emails via SMTP",
    required_credentials=["email", "smtp_password"],
))

# Check if credentials are available
registry = get_tool_registry()
all_available, missing = registry.check_credentials("email_sender")

if not all_available:
    print(f"Missing credentials: {', '.join(missing)}")
```

## Tool Router Integration

The tool router automatically checks credentials before execution:

```python
from rex.tool_router import execute_tool, CredentialMissingError

try:
    result = execute_tool(
        {"tool": "email_sender", "args": {"to": "user@example.com"}},
        default_context={},
    )
except CredentialMissingError as e:
    print(f"Cannot execute {e.tool}: missing {e.missing_credentials}")
```

To skip credential checking (use with caution):

```python
result = execute_tool(
    request,
    default_context,
    skip_credential_check=True,
)
```

## Built-in Tools

Rex registers several built-in tools:

| Tool            | Description                           | Credentials Required |
|-----------------|---------------------------------------|---------------------|
| `time_now`      | Get current time for a location       | None                |
| `weather_now`   | Get current weather (stub)            | `weather_api`       |
| `web_search`    | Search the web                        | `brave` or `serpapi`|
| `send_email`    | Send email messages                   | `email`             |
| `home_assistant`| Control Home Assistant devices        | `home_assistant`    |

## Tool Status

Get comprehensive status for a tool:

```python
from rex.openclaw.tool_registry import get_tool_registry

registry = get_tool_registry()
status = registry.get_tool_status("my_tool")

print(f"Name: {status['name']}")
print(f"Ready: {status['ready']}")
print(f"Enabled: {status['enabled']}")
print(f"Credentials available: {status['credentials_available']}")
print(f"Health OK: {status['health_ok']}")
print(f"Health message: {status['health_message']}")
```

A tool is "ready" if:
- It is enabled
- All required credentials are available
- Health check passes

## Plugin Registration

Plugins can register tools during initialization:

```python
# In plugins/my_plugin.py
from rex.openclaw.tool_registry import ToolMeta, register_tool

def register():
    """Called when plugin is loaded."""
    register_tool(ToolMeta(
        name="plugin_tool",
        description="Tool provided by my plugin",
        required_credentials=["plugin_api_key"],
        health_check=lambda: (True, "Plugin ready"),
    ))
```

## Managing the Registry

### Unregistering Tools

```python
from rex.openclaw.tool_registry import get_tool_registry

registry = get_tool_registry()
registry.unregister_tool("obsolete_tool")
```

### Listing Tools

```python
from rex.openclaw.tool_registry import get_tool_registry

registry = get_tool_registry()

# List enabled tools only
for tool in registry.list_tools():
    print(f"{tool.name}: {tool.description}")

# Include disabled tools
for tool in registry.list_tools(include_disabled=True):
    print(f"{tool.name} (enabled={tool.enabled})")
```

### Using a Custom Registry

```python
from rex.openclaw.tool_registry import ToolRegistry, set_tool_registry

# Create a custom registry
custom_registry = ToolRegistry()
custom_registry.register_tool(...)

# Set as the global registry
set_tool_registry(custom_registry)
```

## Best Practices

1. **Use descriptive names**
   - Tool names should be lowercase with underscores
   - Names should indicate the action (e.g., `send_email`, `get_weather`)

2. **Write meaningful health checks**
   - Check actual service connectivity
   - Return informative error messages
   - Handle exceptions gracefully

3. **Declare all required credentials**
   - List all services the tool needs
   - Let the registry validate availability

4. **Use capability tags**
   - Tag tools with their capabilities
   - Helps with filtering and policy decisions

5. **Version your tools**
   - Track breaking changes
   - Enable compatibility checks

## Troubleshooting

### Tool Not Found

```python
from rex.openclaw.tool_registry import ToolNotFoundError

try:
    status = registry.get_tool_status("unknown_tool")
except ToolNotFoundError:
    print("Tool is not registered")
```

### Missing Credentials

```python
from rex.tool_router import CredentialMissingError

try:
    execute_tool(request, context)
except CredentialMissingError as e:
    print(f"Configure these credentials: {e.missing_credentials}")
```

### Health Check Failures

1. Check network connectivity
2. Verify external service is running
3. Check API keys and authentication
4. Review health check implementation

## Related Documentation

- [credentials.md](credentials.md) - Credential management
- [policy.md](policy.md) - Policy engine for tool execution
- [audit.md](audit.md) - Audit logging of tool calls
