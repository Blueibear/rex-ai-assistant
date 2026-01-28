# Autonomy Features in Rex

This document explains Rex's autonomy features introduced in Phase 9, including the planner, executor, and autonomy modes.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Planner](#planner)
- [Executor](#executor)
- [Autonomy Modes](#autonomy-modes)
- [Usage](#usage)
- [Configuration](#configuration)
- [Safety and Budgets](#safety-and-budgets)
- [Examples](#examples)
- [Limitations](#limitations)

## Overview

Rex's autonomy system enables the AI assistant to autonomously plan and execute multi-step workflows from high-level natural language goals. The system consists of three main components:

1. **Planner**: Converts natural language goals into structured workflows
2. **Executor**: Safely executes workflows with budget constraints
3. **Autonomy Modes**: Controls when and how workflows execute automatically

The system is designed with safety as a primary concern, using budgets, policy checks, and approval gates to ensure controlled execution.

## Architecture

```
User Goal → Planner → Workflow → Validator → Autonomy Check → Executor → Results
                                      ↓              ↓
                              Policy Engine    Budget Limits
                                      ↓              ↓
                              Approval Gates   Evidence Collection
```

### Key Design Principles

1. **Safety First**: All operations are gated by the policy engine
2. **Bounded Execution**: Budgets prevent runaway automation
3. **Transparency**: All actions are logged to the audit trail
4. **Recoverability**: Workflows can be paused and resumed
5. **User Control**: Autonomy modes give users fine-grained control

## Planner

The Planner converts natural language goals into structured workflows.

### How It Works

The planner uses a rule-based approach with pattern matching:

1. Parse the goal string against known patterns
2. Select appropriate tools from the ToolRegistry
3. Generate WorkflowStep objects with tool calls
4. Assign risk levels and idempotency keys
5. Validate against the policy engine

### Supported Goal Patterns

| Pattern | Example | Generated Steps |
|---------|---------|-----------------|
| Email | "send email to alice@example.com" | send_email |
| Newsletter | "send monthly newsletter" | web_search (contacts) → send_email |
| Calendar | "schedule meeting tomorrow" | calendar_create_event |
| Weather | "check weather in Dallas" | weather_now |
| Time | "what's the time in New York" | time_now |
| Home Control | "turn on living room lights" | home_assistant_call_service |
| Web Search | "search for Python tutorials" | web_search |
| Reports | "create monthly report" | web_search (data) → send_email |

### Validation

Before execution, the planner validates workflows:

- ✅ All tools exist in the registry
- ✅ Required credentials are available
- ✅ No steps are denied by policy
- ⚠️ Steps requiring approval are flagged

### Current Limitations

The planner has some limitations in Phase 9:

- **Rule-Based Only**: Uses pattern matching, not LLM-based planning
- **Fixed Patterns**: Only recognizes predefined goal patterns
- **No Optimization**: Doesn't optimize plans or learn from feedback
- **Simple Goals**: Works best with single, clear objectives

Future improvements may include LLM-based planning for more flexible goal understanding.

## Executor

The Executor runs workflows with safety constraints and evidence collection.

### Execution Flow

```
1. Load workflow
2. Create WorkflowRunner
3. For each step:
   a. Check budget limits
   b. Evaluate preconditions
   c. Check policy (via PolicyEngine)
   d. Create approval if needed (blocks execution)
   e. Execute tool call
   f. Collect evidence
   g. Update budget counters
   h. Log to audit trail
4. Generate summary
5. Return ExecutionResult
```

### Budget Enforcement

The executor enforces three types of budgets:

1. **max_actions**: Maximum number of tool calls
   - Prevents excessive API calls
   - Default: 0 (unlimited)

2. **max_messages**: Maximum number of messages sent
   - Prevents spam (email, SMS, notifications)
   - Default: 0 (unlimited)

3. **max_time_seconds**: Maximum execution time
   - Prevents hung workflows
   - Default: 0 (unlimited)

Budgets are checked before each step. If a limit is exceeded, execution stops gracefully.

### Evidence Collection

The executor automatically collects evidence:

- Tool call results
- Execution timestamps
- Success/failure status
- Error messages
- Screenshots (for browser automation)

Evidence is stored as `EvidenceRef` objects and can be used for:

- Debugging failed workflows
- Audit compliance
- User confirmation
- Replay and analysis

### Audit Logging

All executor actions are logged to the audit trail:

- Action ID and task ID
- Tool name and arguments (redacted)
- Policy decision
- Execution result (redacted)
- Duration and timestamp

Sensitive data (API keys, passwords) is automatically redacted before logging.

## Autonomy Modes

Autonomy modes control when workflows execute automatically.

### Mode Definitions

| Mode | Behavior | Use Cases |
|------|----------|-----------|
| **OFF** | No automatic execution, user must explicitly trigger | High-risk operations (OS commands, file operations) |
| **SUGGEST** | Generate plan and show to user, require confirmation | Medium-risk operations (email, calendar, home control) |
| **AUTO** | Automatically execute if policy allows | Low-risk operations (weather, time, web search) |

### Category Mapping

Workflows are categorized based on the tools they use:

| Category | Default Mode | Examples |
|----------|--------------|----------|
| `info.*` | AUTO | weather_now, time_now |
| `web.search` | AUTO | web_search |
| `email.*` | SUGGEST | send_email |
| `calendar.*` | SUGGEST | calendar_create_event |
| `home.*` | SUGGEST | home_assistant_call_service |
| `browser.*` | SUGGEST | browser automation |
| `os.*` | OFF | execute_command |
| `fs.*` | OFF | file operations |

### How Categories Are Inferred

The planner automatically infers categories from workflow properties:

```python
# Email workflow with "newsletter" in title
→ Category: email.newsletter

# Workflow using weather_now tool
→ Category: info.weather

# Workflow using home_assistant_call_service
→ Category: home.control
```

### Wildcard Matching

Categories support hierarchical wildcards:

- `email.*` matches `email.newsletter`, `email.send`, `email.report`
- `info.*` matches `info.weather`, `info.time`
- `os.*` matches `os.command`, `os.file_operation`

## Usage

### Planning a Workflow

```bash
# Generate a plan
rex plan "send monthly newsletter"

# Plan and save to disk
rex plan "check weather in Dallas" --save

# Plan and execute immediately
rex plan "turn on living room lights" --execute

# Execute with budgets
rex plan "send email" --execute --max-actions 5 --max-time 60
```

### Resuming Blocked Workflows

If a workflow is blocked pending approval:

```bash
# View pending approvals
rex approvals

# Approve the request
rex approvals --approve <approval_id>

# Resume execution
rex executor resume <workflow_id>
```

### Running Saved Workflows

```bash
# Run a workflow file
rex run-workflow data/workflows/wf_abc123.json

# Dry-run to preview
rex run-workflow data/workflows/wf_abc123.json --dry-run

# Resume a blocked workflow
rex run-workflow data/workflows/wf_abc123.json --resume
```

## Configuration

### Autonomy Configuration File

Edit `config/autonomy.json` to customize autonomy modes:

```json
{
  "default_mode": "suggest",
  "categories": {
    "info.*": "auto",
    "email.*": "suggest",
    "email.newsletter": "auto",
    "os.*": "off"
  }
}
```

### Per-Category Configuration

Override modes for specific categories:

```json
{
  "categories": {
    "email.newsletter": "auto",
    "email.send": "suggest",
    "email.*": "off"
  }
}
```

The most specific match wins:
- `email.newsletter` → `auto`
- `email.send` → `suggest`
- `email.report` → `off` (matches `email.*`)

### Programmatic Configuration

```python
from rex.autonomy_modes import AutonomyConfig, AutonomyMode

config = AutonomyConfig()
config.set_mode("email.newsletter", AutonomyMode.AUTO)
config.save("config/autonomy.json")
```

## Safety and Budgets

### Default Budget Recommendations

| Workflow Type | max_actions | max_messages | max_time_seconds |
|---------------|-------------|--------------|------------------|
| Simple query | 5 | 0 | 30 |
| Email workflow | 10 | 3 | 60 |
| Report generation | 20 | 5 | 300 |
| Complex automation | 50 | 10 | 600 |

### Policy Integration

The executor always respects policy decisions:

1. **Denied**: Step is skipped, workflow fails
2. **Requires Approval**: Workflow blocks until approved
3. **Allowed**: Step executes immediately

Policy checks happen before budget checks, so denied actions never consume budget.

### Approval Workflow

```
1. Executor encounters step requiring approval
2. Create WorkflowApproval object
3. Save approval to data/approvals/
4. Mark workflow as "blocked"
5. Exit executor, return blocking_approval_id
6. User reviews and approves/denies
7. Resume executor with approved workflow
```

## Examples

### Example 1: Simple Weather Check (AUTO)

```bash
$ rex plan "check weather in Dallas" --execute

Planning workflow for goal: check weather in Dallas
------------------------------------------------------------
Generated workflow: check weather in Dallas
  ID: wf_a1b2c3d4e5f6
  Steps: 1

1. Check weather in Dallas
   Tool: weather_now
   Args: {'location': 'Dallas'}

Validating workflow...
Workflow validation passed.

Autonomy mode: auto

Executing workflow with budget: ExecutionBudget(unlimited)
------------------------------------------------------------

Execution complete
Workflow: wf_a1b2c3d4e5f6
Status: completed
Actions taken: 1
Messages sent: 0
Elapsed time: 0.52s
Evidence: 1 items

Workflow completed successfully. Executed 1 of 1 steps.
```

### Example 2: Email (SUGGEST, requires approval)

```bash
$ rex plan "send email to alice@example.com" --execute

Planning workflow for goal: send email to alice@example.com
------------------------------------------------------------
Generated workflow: send email to alice@example.com
  ID: wf_x9y8z7w6v5u4
  Steps: 1

1. Send email to alice@example.com
   Tool: send_email
   Args: {'to': 'alice@example.com', 'subject': 'Message from Rex', 'body': '...'}
   [REQUIRES APPROVAL]

Validating workflow...
Workflow validation passed.

Autonomy mode: suggest

Executing workflow with budget: ExecutionBudget(unlimited)
------------------------------------------------------------

Execution complete
Workflow: wf_x9y8z7w6v5u4
Status: blocked
Actions taken: 0
Messages sent: 0
Elapsed time: 0.05s
Evidence: 0 items

To approve, run:
  rex approvals --approve apr_123456

Then resume with:
  rex executor resume wf_x9y8z7w6v5u4
```

### Example 3: Newsletter with Budget

```bash
$ rex plan "send monthly newsletter" --execute --max-actions 10 --max-messages 5

Planning workflow for goal: send monthly newsletter
------------------------------------------------------------
Generated workflow: send monthly newsletter
  ID: wf_n1n2n3n4n5n6
  Steps: 2

1. Search for newsletter contact list
   Tool: web_search
   Args: {'query': 'newsletter contacts'}

2. Send newsletter email
   Tool: send_email
   Args: {'to': 'newsletter@example.com', 'subject': 'Newsletter', 'body': '...'}
   [REQUIRES APPROVAL]

Validating workflow...
Workflow validation passed.

Autonomy mode: suggest

Executing workflow with budget: ExecutionBudget(actions=10, messages=5)
------------------------------------------------------------

Execution complete
Workflow: wf_n1n2n3n4n5n6
Status: blocked
Actions taken: 1
Messages sent: 0
Elapsed time: 1.23s
Evidence: 1 items

Workflow blocked pending approval. Executed 1 of 2 steps.
Budget used: 1/10 actions, 0/5 messages
```

## Limitations

### Current Limitations (Phase 9)

1. **Rule-Based Planning**
   - Only recognizes predefined patterns
   - Cannot handle complex or multi-step reasoning
   - No learning or optimization

2. **Limited Goal Understanding**
   - Cannot parse ambiguous goals
   - No clarification questions
   - No context awareness

3. **Fixed Tool Selection**
   - Tools are hard-coded in planning rules
   - No dynamic tool discovery
   - No tool chaining beyond predefined patterns

4. **No Replanning**
   - Cannot adjust plan if steps fail
   - No retry logic
   - No alternative path exploration

5. **Budget Granularity**
   - Budgets apply to entire workflow
   - Cannot set per-step limits
   - No dynamic budget adjustment

### Planned Improvements

Future phases may include:

- **LLM-Based Planning**: Use language models for flexible goal understanding
- **Dynamic Replanning**: Adjust plans based on execution results
- **Learn from Feedback**: Improve plans based on success/failure history
- **Multi-Goal Planning**: Handle complex goals with dependencies
- **Cost Optimization**: Minimize API calls and execution time
- **User Preference Learning**: Adapt autonomy modes based on user patterns

## Best Practices

### When to Use Autonomy

✅ **Good use cases:**
- Repetitive, well-defined tasks
- Low-risk operations (info queries, searches)
- Time-based automation (scheduled reports)
- Known, tested workflows

❌ **Avoid autonomy for:**
- One-time, critical operations
- Tasks with unknown outcomes
- Operations requiring human judgment
- High-risk actions (file deletion, payments)

### Budget Guidelines

1. **Start Conservative**: Use low budgets initially
2. **Monitor Execution**: Check audit logs for actual usage
3. **Adjust Gradually**: Increase budgets as confidence grows
4. **Set Time Limits**: Always set max_time_seconds to prevent hangs

### Security Considerations

1. **Review Approvals Carefully**: Don't approve blindly
2. **Check Autonomy Config**: Regularly review category modes
3. **Monitor Audit Logs**: Look for unexpected patterns
4. **Use Policy Engine**: Configure policies before enabling autonomy
5. **Test in Dry-Run**: Preview workflows before execution

## Troubleshooting

### "Unable to plan for goal"

**Cause**: Goal doesn't match any known patterns

**Solution**: Try rephrasing the goal to match supported patterns:
- "send email" instead of "email someone"
- "check weather in Dallas" instead of "Dallas weather"

### "Workflow validation failed"

**Cause**: Missing tools or credentials, or policy denials

**Solution**:
1. Check tool registry: `rex tools`
2. Verify credentials are configured
3. Review policy settings
4. Check audit logs for denial reasons

### "Budget exceeded"

**Cause**: Workflow consumed more resources than budgeted

**Solution**:
1. Review workflow complexity
2. Increase budget limits
3. Simplify the workflow
4. Check for loops or inefficiencies

### Workflow stuck in "blocked" state

**Cause**: Pending approval not resolved

**Solution**:
1. List approvals: `rex approvals`
2. Approve or deny: `rex approvals --approve <id>`
3. Resume: `rex executor resume <workflow_id>`

## See Also

- [Workflow Engine Documentation](workflow_engine.md)
- [Policy Engine Documentation](policy_engine.md)
- [Audit Logging Documentation](audit.md)
- [Tool Registry Documentation](tool_registry.md)
- [CLI Reference](cli.md)

## Support

For questions or issues:
- GitHub: https://github.com/Blueibear/rex-ai-assistant
- Documentation: https://github.com/Blueibear/rex-ai-assistant/tree/main/docs
