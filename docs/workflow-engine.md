# Rex Workflow Engine

The Rex Workflow Engine provides a structured, persistent system for executing multi-step tasks with policy enforcement, approval gates, and recovery capabilities.

## Overview

The workflow engine is designed to:

- **Sequence complex tasks** - Chain multiple tool calls together with dependencies
- **Enforce safety** - Integrate with the policy engine to gate risky actions
- **Support approvals** - Block execution for user review of sensitive operations
- **Enable recovery** - Persist state to disk so workflows can resume after restarts
- **Provide visibility** - Dry-run mode shows what will happen before execution

## Key Concepts

### Workflow

A `Workflow` is a named sequence of steps to be executed. Each workflow has:

- **workflow_id**: Unique identifier
- **title**: Human-readable description
- **status**: Current state (queued, running, blocked, completed, failed, canceled)
- **steps**: Ordered list of `WorkflowStep` objects
- **current_step_index**: Which step to execute next (for resume)
- **state**: Dictionary for storing intermediate values between steps
- **idempotency_key**: Optional key for preventing duplicate execution

### WorkflowStep

A `WorkflowStep` represents a single action in a workflow:

- **step_id**: Unique identifier within the workflow
- **description**: What this step does
- **tool_call**: The tool to invoke (optional - steps without tool calls are checkpoints)
- **precondition**: Name of a condition function to check before execution
- **postcondition**: Name of a condition function to validate after execution
- **idempotency_key**: Optional key for skipping already-executed steps

### Conditions

Preconditions and postconditions are named functions registered with the condition registry. They receive the workflow state dictionary and return True/False.

Built-in conditions:
- `always_true`: Always returns True
- `always_false`: Always returns False

Register custom conditions:

```python
from rex.workflow import register_condition

def check_data_ready(state):
    return "data" in state and state["data"] is not None

register_condition("check_data_ready", check_data_ready)
```

### Approvals

When the policy engine determines a step requires approval:

1. A `WorkflowApproval` object is created and saved to `data/approvals/`
2. The workflow status is set to "blocked"
3. The approval_id is recorded for resumption

Approvals can be approved/denied via:
- CLI: `rex approvals --approve <id>` or `rex approvals --deny <id>`
- Programmatically: `approve_workflow(id)` or `deny_workflow(id)`
- Manually: Edit the JSON file in `data/approvals/`

## Defining Workflows

### JSON Format

Workflows can be defined as JSON files:

```json
{
  "workflow_id": "wf_morning_routine",
  "title": "Morning briefing workflow",
  "steps": [
    {
      "step_id": "get_time",
      "description": "Get current local time",
      "tool_call": {
        "tool": "time_now",
        "args": {"location": "Dallas, TX"}
      }
    },
    {
      "step_id": "check_weather",
      "description": "Get weather forecast",
      "tool_call": {
        "tool": "weather_now",
        "args": {"location": "Dallas, TX"}
      },
      "idempotency_key": "weather_check_2024_01_15"
    },
    {
      "step_id": "send_summary",
      "description": "Email morning summary",
      "tool_call": {
        "tool": "send_email",
        "args": {
          "to": "user@example.com",
          "subject": "Morning Briefing",
          "body": "Good morning! Here's your update..."
        }
      }
    }
  ]
}
```

### Programmatic Creation

```python
from rex.workflow import Workflow, WorkflowStep
from rex.contracts import ToolCall

workflow = Workflow(
    title="Daily backup workflow",
    steps=[
        WorkflowStep(
            description="Check disk space",
            tool_call=ToolCall(tool="check_disk", args={}),
            postcondition="disk_has_space",
        ),
        WorkflowStep(
            description="Create backup",
            tool_call=ToolCall(tool="create_backup", args={"target": "/backup"}),
        ),
    ],
)

# Save to disk
workflow.save()
```

## Running Workflows

### CLI Commands

```bash
# Run a workflow
rex run-workflow workflow.json

# Preview without executing (dry-run)
rex run-workflow workflow.json --dry-run

# Resume a blocked workflow after approval
rex run-workflow workflow.json --resume
```

### Programmatic Execution

```python
from rex.workflow import Workflow
from rex.workflow_runner import WorkflowRunner

# Load workflow
workflow = Workflow.load_from_file("workflow.json")

# Create runner
runner = WorkflowRunner(workflow)

# Dry run first
dry_result = runner.dry_run()
for step in dry_result.steps:
    print(f"{step.step_id}: {step.policy_decision}")

# Execute
result = runner.run()
print(f"Status: {result.status}")
print(f"Executed: {result.steps_executed}/{result.steps_total}")
```

## Managing Approvals

### List Pending Approvals

```bash
rex approvals
```

Output:
```
Pending Approvals
============================================================

apr_abc123def456
  Workflow: wf_morning_routine
  Step: send_summary
  Description: Email morning summary
  Tool: send_email({'to': 'user@example.com', ...})
  Requested: 2024-01-15T10:30:00Z

Total: 1 pending approval(s)

To approve: rex approvals --approve <approval_id>
To deny:    rex approvals --deny <approval_id> --reason 'reason'
```

### Approve or Deny

```bash
# Approve
rex approvals --approve apr_abc123def456

# Deny with reason
rex approvals --deny apr_abc123def456 --reason "Not authorized for external emails"
```

### View Approval Details

```bash
rex approvals --show apr_abc123def456
```

## Workflow Status Management

### List Workflows

```bash
# List all workflows
rex workflows

# Filter by status
rex workflows --status blocked
rex workflows --status completed
```

### Workflow Statuses

| Status | Description |
|--------|-------------|
| queued | Created but not started |
| running | Currently executing |
| blocked | Waiting for approval |
| completed | All steps finished successfully |
| failed | A step failed or was denied |
| canceled | Manually canceled |

## Persistence and Recovery

Workflows persist their state to `data/workflows/{workflow_id}.json` after each step. This enables:

1. **Recovery after restart**: If Rex crashes or restarts, workflows resume from the last completed step
2. **Audit trail**: Each step's result is recorded
3. **Resume after approval**: Blocked workflows can be resumed once approved

Approvals are stored in `data/approvals/{approval_id}.json`.

## Idempotency

Steps can have an `idempotency_key` to prevent duplicate execution:

```json
{
  "step_id": "send_welcome",
  "description": "Send welcome email",
  "idempotency_key": "welcome_email_user123",
  "tool_call": {
    "tool": "send_email",
    "args": {"to": "user@example.com"}
  }
}
```

If a step with the same idempotency key was previously executed successfully in the workflow, it will be skipped on re-run.

## Policy Integration

The workflow runner uses the `PolicyEngine` to evaluate each step's tool call:

| Decision | Behavior |
|----------|----------|
| Allowed (auto-execute) | Step executes immediately |
| Requires approval | Workflow blocks, approval created |
| Denied | Step fails, workflow marked as failed |

Default policies (from `rex.policy_engine`):
- Low-risk tools (time_now, weather_now): Auto-execute
- Medium-risk tools (send_email, calendar_*): Requires approval
- High-risk tools (execute_command, file_delete): Requires approval

## Audit Logging

All workflow step executions are logged to the audit log (`data/logs/audit.log`), including:
- Tool name and arguments
- Policy decision
- Execution result or error
- Timestamps

## Current Limitations

1. **No UI for approvals**: Approvals must be managed via CLI or JSON files
2. **Sequential execution only**: Steps execute one at a time (no parallel steps)
3. **Limited tool implementations**: Only `time_now` is fully implemented currently
4. **No scheduled execution**: Workflows must be triggered manually (scheduler integration coming)

## Example: Complete Workflow

```json
{
  "workflow_id": "wf_daily_report",
  "title": "Daily Status Report",
  "steps": [
    {
      "step_id": "gather_metrics",
      "description": "Collect system metrics",
      "tool_call": {
        "tool": "time_now",
        "args": {"location": "Dallas, TX"}
      }
    },
    {
      "step_id": "generate_report",
      "description": "Generate report content",
      "tool_call": {
        "tool": "web_search",
        "args": {"query": "system status"}
      },
      "precondition": "always_true"
    },
    {
      "step_id": "send_report",
      "description": "Email daily report to team",
      "tool_call": {
        "tool": "send_email",
        "args": {
          "to": "team@company.com",
          "subject": "Daily Status Report",
          "body": "Here is today's report..."
        }
      },
      "idempotency_key": "daily_report_2024_01_15"
    }
  ],
  "state": {}
}
```

Run with:
```bash
# Preview first
rex run-workflow daily_report.json --dry-run

# Execute
rex run-workflow daily_report.json

# If blocked, approve and resume
rex approvals --approve apr_xxxxx
rex run-workflow daily_report.json --resume
```

## Related Documentation

- [Policy Engine](policy.md) - How tool calls are gated
- [Audit Logging](audit.md) - Execution history and traceability
- [Tools](tools.md) - Available tools for workflow steps
- [Credentials](credentials.md) - Managing secrets for tool execution
