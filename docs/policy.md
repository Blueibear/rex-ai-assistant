# Policy Engine

The Rex Policy Engine provides a permissions and safety layer for tool execution. It enforces least-privilege principles by classifying actions by risk, maintaining allow/deny lists, and requiring approval for higher-risk operations.

## Overview

All tool calls in Rex are evaluated by the policy engine before execution. Based on configured policies, the engine determines whether to:

- **Auto-execute**: Proceed immediately without user intervention
- **Require approval**: Pause and wait for user confirmation
- **Deny**: Block the action entirely

## Risk Levels

Actions are classified into three risk levels:

| Level | Description | Examples |
|-------|-------------|----------|
| `low` | Read-only operations, no external effects | `time_now`, `weather_now`, `web_search` |
| `medium` | External effects but reversible/limited scope | `send_email`, `calendar_create_event` |
| `high` | Significant external effects, potentially destructive | `execute_command`, `file_delete` |

## Policy Structure

Policies are defined using the `ActionPolicy` model:

```python
from rex.policy import ActionPolicy
from rex.contracts import RiskLevel

policy = ActionPolicy(
    tool_name="send_email",           # Tool this policy applies to
    risk=RiskLevel.MEDIUM,            # Risk classification
    allow_auto=False,                 # Whether to auto-execute (only for low risk)
    allowed_recipients=None,          # Whitelist of allowed recipients
    denied_recipients=["spam@x.com"], # Blacklist of denied recipients
    allowed_domains=None,             # Whitelist of allowed domains
    denied_domains=["malicious.com"], # Blacklist of denied domains
)
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool this policy applies to |
| `risk` | `RiskLevel` | Risk level (low, medium, high) |
| `allow_auto` | `bool` | If `True` and risk is `low`, auto-execute without approval |
| `allowed_recipients` | `list[str] \| None` | If set, only these recipients are allowed |
| `denied_recipients` | `list[str] \| None` | If set, these recipients are blocked |
| `allowed_domains` | `list[str] \| None` | If set, only these domains are allowed |
| `denied_domains` | `list[str] \| None` | If set, these domains are blocked |

## Decision Types

The policy engine returns a `PolicyDecision` with these outcomes:

### Allowed (Auto-Execute)
```python
PolicyDecision(
    allowed=True,
    denied=False,
    requires_approval=False,
    reason="Low-risk tool with auto-execute enabled"
)
```
The tool executes immediately without user intervention.

### Requires Approval
```python
PolicyDecision(
    allowed=True,
    denied=False,
    requires_approval=True,
    reason="Medium-risk action requires user approval"
)
```
The tool execution is paused until the user grants approval.

### Denied
```python
PolicyDecision(
    allowed=False,
    denied=True,
    requires_approval=False,
    reason="Recipient 'spam@example.com' is on the deny list"
)
```
The tool execution is blocked and cannot proceed.

## Default Policies

The following default policies are configured:

### Low-Risk (Auto-Execute)
- `time_now` - Get current time
- `weather_now` - Get weather information
- `web_search` - Search the web

### Medium-Risk (Requires Approval)
- `send_email` - Send email messages
- `calendar_create_event` - Create calendar events
- `calendar_delete_event` - Delete calendar events
- `home_assistant_call_service` - Call Home Assistant services

### High-Risk (Requires Approval)
- `execute_command` - Execute shell commands
- `file_write` - Write files
- `file_delete` - Delete files

## Usage

### Basic Usage

```python
from rex.policy_engine import PolicyEngine, get_policy_engine
from rex.contracts import ToolCall

# Use the default singleton
engine = get_policy_engine()

# Create a tool call
tool_call = ToolCall(tool="send_email", args={"to": "user@example.com"})

# Evaluate the policy
metadata = {"recipient": "user@example.com"}
decision = engine.decide(tool_call, metadata)

if decision.denied:
    print(f"Action denied: {decision.reason}")
elif decision.requires_approval:
    print(f"Approval required: {decision.reason}")
else:
    print("Auto-executing...")
```

### Custom Policies

```python
from rex.policy import ActionPolicy
from rex.policy_engine import PolicyEngine
from rex.contracts import RiskLevel

# Create custom policies
custom_policies = [
    ActionPolicy(
        tool_name="send_email",
        risk=RiskLevel.LOW,  # Downgrade risk for trusted tool
        allow_auto=True,
        allowed_domains=["company.com"],  # Only allow company emails
    ),
    ActionPolicy(
        tool_name="custom_tool",
        risk=RiskLevel.HIGH,
        allow_auto=False,
    ),
]

# Create engine with custom policies
engine = PolicyEngine(policies=custom_policies)
```

### Integration with Tool Router

The tool router automatically integrates with the policy engine:

```python
from rex.tool_router import execute_tool, PolicyDeniedError, ApprovalRequiredError

try:
    result = execute_tool(
        {"tool": "send_email", "args": {"to": "user@example.com"}},
        default_context={},
    )
except PolicyDeniedError as e:
    print(f"Tool denied: {e.reason}")
except ApprovalRequiredError as e:
    print(f"Approval needed: {e.reason}")
```

## Extending Policies

### Adding Policies at Runtime

```python
engine = get_policy_engine()
engine.add_policy(ActionPolicy(
    tool_name="new_tool",
    risk=RiskLevel.MEDIUM,
))
```

### Removing Policies

```python
engine.remove_policy("tool_name")
```

### Custom Default Policy

```python
# Set a custom default for unknown tools
default = ActionPolicy(
    tool_name="__default__",
    risk=RiskLevel.HIGH,  # Treat unknown tools as high-risk
    allow_auto=False,
)
engine = PolicyEngine(default_policy=default)
```

## Future Enhancements

### Dynamic Policy Loading (Planned)

Policies could be loaded from a configuration file at `config/policy.json`:

```json
{
    "policies": [
        {
            "tool_name": "send_email",
            "risk": "medium",
            "allow_auto": false,
            "denied_domains": ["malicious.com"]
        }
    ]
}
```

This would be implemented via a `PolicyEngine.load_from_file()` class method.

### Pattern Matching (Planned)

Future versions may support glob or regex patterns for recipients and domains:

```python
ActionPolicy(
    tool_name="send_email",
    denied_recipients=["*@competitor.com"],  # Block all competitor emails
    allowed_domains=["*.company.com"],       # Allow all subdomains
)
```

## Security Considerations

1. **Default Deny for Unknown Tools**: Unknown tools default to medium-risk with approval required
2. **Case-Insensitive Matching**: Email addresses and domains are matched case-insensitively
3. **Deny Lists Take Precedence**: Deny lists are checked before allow lists
4. **No Secrets in Policies**: Never store credentials or API keys in policy configurations

## Testing

Run policy tests with:

```bash
pytest tests/test_policy.py -v
pytest tests/test_tool_router_policy.py -v
```

## Related Documentation

- [Contracts](contracts.md) - Core data models including `ToolCall` and `RiskLevel`
- [Workflow Engine](workflow-engine.md) - Multi-step task execution with policy enforcement
- [Architecture](ARCHITECTURE.md) - System architecture overview
