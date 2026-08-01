# REX Unified Build Spec and Must-Haves

This document combines `rex_assistant_build_spec.md` and `REX_Must_Haves.md` into one non-redundant master planning document.

It is meant to define what Rex should become, what must be built first, and the design rules that should guide future implementation.

---

## 1. Mission

Build Rex as a local-first, tool-using, memory-aware personal AI assistant that can operate through:

- Voice
- Browser
- Desktop GUI
- Mobile access
- API calls
- Home Assistant
- n8n workflows
- Business automations
- File and code tooling

Rex should not simply answer questions. Rex should understand intent, gather context, use tools, perform actions, verify outcomes, remember useful information, recover from failures, and adapt to James's household and business needs over time.

**Core standard:** Rex is only truly useful when it can safely do things, verify that they worked, and explain what happened.

---

## 2. Top 10 Build Priorities

If everything had to be reduced to the most important items, build these first:

1. Stable service mode
2. Reliable wake word to spoken response
3. Home Assistant natural language control
4. Tool verification before saying "done"
5. Error recovery with useful explanations
6. Context engine that pulls memory, device state, files, and recent history
7. Long-term searchable memory
8. Model router with fallback
9. Security, permissions, and confirmation gates
10. Observability dashboard and golden tests

These priorities should drive the order of work unless a blocker prevents progress.

---

## 3. Most Important Design Rule

## Rex must never pretend

Rex must clearly distinguish between:

1. Things it knows
2. Things it inferred
3. Things it guessed
4. Things it attempted
5. Things it verified

### Bad

> "I fixed the workflow."

### Good

> "I found the likely issue in the ComfyUI payload node. I updated the expression and ran a syntax check. I have not verified the full workflow execution yet."

This rule should apply to every tool call, automation, code edit, smart home command, and business workflow.

---

## 4. Core Architecture

Rex should be built as a layered assistant system.

| Layer | Purpose |
|---|---|
| Input Layer | Accepts voice, text, browser, mobile, API calls, and n8n triggers |
| Identity Layer | Knows who is speaking or using Rex |
| Intent Layer | Determines what the user wants |
| Context Layer | Pulls relevant memory, files, device states, and recent history |
| Model Layer | Chooses the right LLM or local model for the task |
| Planning Layer | Breaks complex tasks into safe, executable steps |
| Tool Layer | Gives Rex abilities outside the language model |
| Verification Layer | Confirms whether actions succeeded |
| Response Layer | Adapts answers for voice, screen, mobile, or automation |
| Memory Layer | Saves useful short-term and long-term information |
| Monitoring Layer | Logs errors, decisions, actions, and failures |
| Security Layer | Enforces permissions, confirmations, and safe defaults |
| Evaluation Layer | Tests whether Rex is getting better or regressing |

**Fact:** Without these layers, Rex is just a chatbot wrapper. With them, Rex becomes an assistant.

**Confidence: High**  
**What would raise confidence:** Repo validation showing that each layer exists as a real implementation and not just scaffolding.

---


### External Capability Ecosystems

Rex should support external capability ecosystems that allow new tools, skills, channels, and automations to be added without major modifications to the Rex core codebase.

Primary target ecosystem:
- OpenClaw + ClawHub

Architecture principles:
- Rex remains the orchestrator, verifier, memory system, permission layer, and user-facing assistant.
- External systems provide optional tools and capabilities.
- Rex must never lose core functionality if an external ecosystem becomes unavailable.
- Rex should treat OpenClaw as an expandable capability provider, not the core assistant brain.

Rex should support:
- Dynamic plugin discovery
- Dynamic skill discovery
- Capability synchronization
- Tool allowlists and denylists
- External tool health monitoring
- External tool verification
- Runtime capability refresh without restart


---

## 5. Stable Core Runtime

Rex needs one clean, predictable way to start and run.

### Must include

- Canonical startup modes:
  - CLI chat
  - Voice loop
  - API/server mode
  - Combined service mode
- One config source of truth
- `.env` loaded once, early, and consistently
- Doctor checks before long-running services start
- Windows-first service mode that survives reboot
- Clear startup, stop, restart, and status commands
- Health monitoring for long-running services
- Watchdog behavior if the voice loop dies

### Why it matters

Nothing else matters if Rex only works when manually babysat.

### Confidence

**High**  
**What would raise confidence:** Repeated successful startup tests from a clean reboot.

---

## 6. Voice Layer

Voice is Rex's front door.

### Must include

| Ability | Requirement |
|---|---|
| Wake word | "Hey Rex" detection that works reliably |
| Voice activity detection | Know when the user stopped speaking |
| Speech-to-text | Fast, accurate transcription |
| Text-to-speech | Natural voice output |
| Automatic playback | Audio plays without hunting for files |
| Barge-in | User can interrupt Rex |
| Short acknowledgments | "On it," "Done," "Checking" |
| Room awareness | Know where the command came from when possible |
| Multi-user support | Recognize James vs Cole eventually |
| Latency management | Speak quickly for simple tasks |
| Recovery | Handle missing mic, speaker, or audio device errors |

### Voice response modes

| Mode | Behavior |
|---|---|
| Voice-only | Complete spoken answer, never rely on a screen |
| Screen | Full visual answer with more detail |
| Hybrid | Short spoken summary plus full UI or mobile detail |
| Automation | Minimal status output plus logs |

### Required rule

A household voice assistant cannot depend on a screen.

Bad voice-only response:

> "Read what is on the screen."

Good voice-only response:

> "The kitchen light is on. I also found two related automations that may need cleanup."

### Confidence

**High**  
**What would raise confidence:** Wake word to spoken response tests across restarts, audio devices, and response modes.

---

## 7. Intent and Context Engine

Rex needs to know what kind of request it is handling before deciding what to do.

### Intent router must classify

- Smart home command
- General question
- Web research task
- Coding or debugging task
- File action
- Business or Nasteeshirts task
- Reminder or scheduling request
- Media request
- Risky or destructive action
- Multi-step workflow

### Context engine must gather

- Current user
- Current room or device context
- Conversation state
- Relevant memory
- Home Assistant entity states
- Project files and logs
- Recent task history
- Available tools
- Permissions
- Response mode

### Why it matters

Rex cannot be smart if every request starts from zero.

### Confidence

**High**  
**What would raise confidence:** Logs showing what context was retrieved and why it was used.

---

## 8. Model Layer

Rex should not depend on one model for everything.

### Required model roles

| Model Role | Purpose |
|---|---|
| Fast local model | Quick voice responses, basic commands, smart home control |
| Reasoning model | Planning, troubleshooting, coding, complex questions |
| Creative model | Stories, product descriptions, marketing copy |
| Embedding model | Search, memory recall, document retrieval |
| Vision model | Image understanding, screenshots, product images |
| Cloud fallback | Used only when local models are not good enough |

### Model router examples

| User Request | Best Model |
|---|---|
| "Turn on the kitchen lights" | Fast local model |
| "Why is this Python script failing?" | Reasoning model |
| "Write 10 funny shirt slogans" | Creative model |
| "What did I say about n8n last week?" | Retrieval plus reasoning model |
| "What is in this image?" | Vision model |

**Inference:** Model routing matters more than using one huge model for every task. A smaller model with the right tools can beat a larger model with poor orchestration.

**Confidence: High**  
**What would raise confidence:** Benchmarks comparing response quality, speed, and cost across routed tasks.

---

## 9. Tool Layer

The model should not be the whole assistant. The model should be the brain that chooses and coordinates tools.

### Essential tool categories

| Tool Area | Examples |
|---|---|
| Home Assistant | Lights, switches, thermostat, scenes, locks, speakers |
| Web and research | Search, summarize, compare sources, cite sources |
| Files and documents | Search, read, summarize, edit safely, diff changes |
| Code tools | Inspect code, modify code, run tests, explain errors |
| n8n | Trigger workflows, inspect failed nodes, review payloads |
| ComfyUI | Generate or process images, confirm output files |
| WooCommerce | Products, orders, inventory, descriptions, metadata |
| Nasteeshirts | Listings, tags, SEO, trends, social content |
| Plex and media | Search and play movies, shows, episodes, target devices |
| Calendar and email | Scheduling, reminders, inbox search, draft replies |
| Desktop control | Open apps, manage files, run safe commands |
| External capability ecosystems | OpenClaw plugins, ClawHub skills, distributed tool providers |


### External Tool Ecosystems

Rex should support modular external tool ecosystems such as OpenClaw and ClawHub.

These systems may provide:
- Browser automation
- Communication integrations
- Device integrations
- Workflow tools
- Distributed agents
- Voice gateways
- Automation triggers
- Retrieval systems
- Third-party plugins

Rex should treat these as optional external tool providers rather than part of the Rex core runtime.

Rex must:
- Verify actions performed through external tools
- Enforce permissions before tool execution
- Normalize responses into a consistent Rex tool format
- Distinguish between attempted and verified outcomes
- Gracefully recover if external ecosystems fail


### Tool requirements

Every tool should have:

- Clear name
- Description
- Required parameters
- Permission rules
- Risk level
- Confirmation rule
- Execution function
- Verification function
- Error handler
- Tests

### Confidence

**High**  
**What would raise confidence:** A working tool registry where tools are discoverable, callable, permission-aware, and verifiable.

---

## 10. Home Assistant Core

Home Assistant control is one of the highest priority Rex features because it is central to replacing Alexa.

### Must include

| Ability | Requirement |
|---|---|
| List devices | Pull current entities from Home Assistant |
| Control devices | Lights, switches, thermostat, scenes, locks, speakers |
| Understand aliases | "bar lights," "cart lights," "living room lamp" |
| Use context | Room, speaker location, recent commands, device state |
| Check state | Know whether something is already on or off |
| Confirm result | Verify the device state after acting |
| Clarify ambiguity | Ask only when needed |
| Suggest matches | Offer likely devices when exact match fails |
| Undo | Support reversible smart home actions when possible |
| Recover from failure | Explain what failed and why |

### Example successful command

> "Done. The living room lights are on."

### Example failed command

> "I could not turn on the bar lights because Home Assistant did not return that entity. I found similar entities named `light.bar_leds` and `switch.bar_cart`."

### Confidence

**High**  
**What would raise confidence:** Golden tests for natural-language Home Assistant commands.

---

## 11. Memory Layer

Rex needs several kinds of memory, not one pile of saved facts.

### Required memory types

| Memory Type | Purpose |
|---|---|
| Working memory | What is happening right now |
| Session memory | Current conversation context |
| User profile memory | Stable facts about James and Cole |
| Preference memory | How users like things done |
| Episodic memory | Important past events and conversations |
| Semantic memory | General facts Rex has learned |
| Procedural memory | How to perform recurring tasks |
| Project memory | Status of Rex, n8n, ComfyUI, Home Assistant, and Nasteeshirts work |
| Device memory | Home Assistant aliases and device behavior |
| Business memory | Nasteeshirts policies, tone, products, workflows, SEO patterns |

### Memory rules

Rex should:

1. Save only useful information.
2. Separate James and Cole's memory.
3. Ask before saving sensitive details.
4. Store timestamps and source context.
5. Allow memory search.
6. Allow memory editing and deletion.
7. Use memory only when relevant.
8. Avoid treating old memory as automatically current.

### Why it matters

Good memory is not "save everything." Good memory is selective, searchable, permission-aware, and regularly cleaned.

### Confidence

**High**  
**What would raise confidence:** Memory recall tests showing relevant memories are retrieved while irrelevant ones are ignored.

---

## 12. Retrieval Layer

Rex needs retrieval-augmented generation so it can search before answering when context matters.

### Required retrieval sources

| Source | Purpose |
|---|---|
| Local project files | Rex code, configs, logs |
| User documents | PDFs, notes, instructions |
| Nasteeshirts content | Product info, policies, designs |
| Home Assistant state | Devices, scenes, automations |
| Past conversations | Project decisions and history |
| Web search | Current information |
| Vector database | Semantic memory search |
| SQL database | Structured memory and logs |

### Retrieval requirements

Rex should:

- Search before answering when needed
- Rank results
- Detect stale information
- Respect permissions
- Summarize retrieved context
- Cite or identify sources
- Avoid bloating prompts with irrelevant material

### Why it matters

Rex should not hallucinate what it could have looked up.

### Confidence

**High**  
**What would raise confidence:** Retrieval tests against known files, memories, and logs.

---

## 13. Planning and Execution Layer

Rex needs to think before acting.

### Required action pipeline

```text
User request
→ Identify intent
→ Check identity and permission
→ Gather context
→ Decide whether confirmation is needed
→ Plan action
→ Execute tool call
→ Verify result
→ Report outcome
→ Log action
→ Save memory if useful
```

### Confirmation rules

| Command | Confirmation Needed? |
|---|---|
| Turn on a light | No |
| Check weather | No |
| Play music | Usually no |
| Set thermostat slightly | Maybe, depending on rule |
| Delete files | Yes |
| Send email | Yes |
| Buy something | Yes |
| Unlock door | Yes |
| Open garage | Yes |
| Change security settings | Yes |
| Modify code and commit | Yes |
| Push code to GitHub | Yes |
| Run destructive terminal commands | Yes |

### Why it matters

This lets Rex handle requests like:

- "Figure out why the n8n workflow failed and fix it."
- "Create a new Nasteeshirts product draft from this design."
- "Set up an automation that turns on the hallway light when I come home after dark."
- "Find the issue in the repo and write a prompt for Claude Code to fix it."

### Confidence

**High**  
**What would raise confidence:** End-to-end plan execution tests with safe mock tools.

---

## 14. Verification Layer

Rex must verify actions before saying they are done.

### Required verification behavior

| Action | Verification |
|---|---|
| Turn on light | Re-check Home Assistant entity state |
| Run n8n workflow | Confirm execution status |
| Generate image | Confirm output file exists |
| Edit code | Run syntax check or tests |
| Send API request | Check response code |
| Create file | Confirm path exists |
| Schedule task | Confirm task is saved |
| Start service | Confirm port is listening |

### Required distinction

Rex should separate:

- Attempted
- Succeeded
- Failed
- Verified
- Not verified

### Why it matters

Attempted is not the same as completed.

### Confidence

**High**  
**What would raise confidence:** Logs that separate attempted actions from verified outcomes.

---

## 15. Error Recovery Layer

Rex should be excellent at failing.

### Required failure handling

| Failure Type | Behavior |
|---|---|
| Tool timeout | Retry once, then explain |
| API error | Show exact error and likely cause |
| Missing config | Tell user what setting is missing |
| Missing file | Search likely locations |
| Bad command | Stop before damage |
| Speech recognition error | Ask for repeat |
| Device not found | Suggest close matches |
| Model failure | Fall back to another model |
| Workflow failure | Identify failing node and payload |

### Example

Instead of:

> "Something went wrong."

Rex should say:

> "The ComfyUI workflow failed at `Build ComfyUI Payload` because the JavaScript expression has an unexpected closing parenthesis. The workflow did not reach the image generation node."

### Confidence

**High**  
**What would raise confidence:** Failure-mode tests for common broken states.

---

## 16. Skill System

Rex should have modular skills instead of hardcoded one-off behavior.

### Each skill should include

| Part | Purpose |
|---|---|
| Name | Human-readable skill name |
| Description | What the skill does |
| Trigger examples | Phrases that activate it |
| Required tools | APIs or services needed |
| Permissions | What it can access |
| Parameters | Inputs it needs |
| Confirmation rules | When to ask first |
| Execution function | Actual code |
| Verification function | Confirms success |
| Error handler | Handles failure |
| Tests | Proves it works |

### Example skill

```json
{
  "name": "Turn On Home Assistant Device",
  "description": "Turns on a light, switch, or scene using Home Assistant.",
  "examples": [
    "Turn on the kitchen light",
    "Switch on the bar lights",
    "Turn the living room lamp on"
  ],
  "required_tools": ["home_assistant"],
  "requires_confirmation": false,
  "verification": "check_entity_state_after_action",
  "failure_behavior": "suggest_similar_entities"
}
```

### Why it matters

This lets Rex grow without becoming a junk drawer full of half-working integrations.

### Confidence

**High**  
**What would raise confidence:** A working skill registry with several real skills and tests.

---

## 17. Automation Layer

Rex should be proactive, but carefully.

### Required automation abilities

| Ability | Example |
|---|---|
| Scheduled tasks | Weekly Nasteeshirts trend report |
| Event triggers | New shirt added to WooCommerce |
| Device triggers | Garage opens after dark |
| Business triggers | Low sales, abandoned cart spike |
| Reminder triggers | Daily tasks, appointments |
| Workflow triggers | Start n8n workflow |
| Conditional alerts | "Notify me if ComfyUI fails" |

### Proactive behavior rules

Rex should act or notify only when:

1. It is clearly useful.
2. The user previously approved it.
3. The action is low-risk.
4. It has enough confidence.
5. It can explain why it acted.

### Examples

- "You usually turn on the hallway light around this time after sunset. Want me to automate that?"
- "The last three ComfyUI runs failed at the same node. Want me to inspect the payload?"
- "Nasteeshirts traffic is up but conversions are down. Want a quick report?"
- "You have a weekly trend report scheduled. I found a new phrase gaining traction."

### Confidence

**Medium to High**  
**What would raise confidence:** Reliable pattern detection with low false positives and clear approval flows.

---

## 18. Security and Permissions

This is non-negotiable.

### Required security features

| Feature | Requirement |
|---|---|
| Secret management | API keys in `.env` or vault, never hardcoded |
| User authentication | Especially web UI and remote access |
| Role permissions | James and Cole can have different access |
| Local-first design | Private data stays local when possible |
| HTTPS | Required for remote access |
| Audit logs | Track actions and tool calls |
| Confirmation gates | Required for risky actions |
| Rate limits | Prevent accidental loops |
| Backups | Memory, configs, database, workflows |
| Recovery mode | Disable tools if something behaves badly |
| Deny-by-default | Destructive actions disabled unless explicitly allowed |

### Dangerous actions requiring confirmation

| Action | Require Confirmation |
|---|---|
| Delete files | Yes |
| Send email | Yes |
| Buy product | Yes |
| Unlock door | Yes |
| Open garage | Yes |
| Change passwords | Yes |
| Modify DNS | Yes |
| Push code to GitHub | Yes |
| Run destructive terminal commands | Yes |

### Confidence

**High**  
**What would raise confidence:** Security tests, permission tests, and threat modeling for exposed endpoints.

---

## 19. UX and Response Layer

Rex needs to be understandable, not just powerful.

### User-facing behavior rules

| Rule | Why |
|---|---|
| Be concise for simple commands | Reduces annoyance |
| Explain failures clearly | Helps debugging |
| Avoid fake certainty | Builds trust |
| Ask follow-up questions only when necessary | Keeps flow fast |
| Use spoken summaries for long answers | Better voice UX |
| Show detailed answers in UI when available | Better screen UX |
| Remember user preferences | Feels personalized |
| Adapt tone by context | Business, coding, casual, urgent |
| Never rely on unseen UI in voice mode | Required for screenless use |

### Response examples

Smart home success:

> "Done. The living room lights are on."

Smart home failure:

> "I could not turn on the bar lights because Home Assistant did not return that entity. I found similar entities named `light.bar_leds` and `switch.bar_cart`."

Long answer in voice mode:

> "The short answer is yes. Rex should use a voice-only mode so it never tells you to read the screen when you are talking from another room. I can send the full version to the app."

---

## 20. Observability and Dashboard

Rex needs to show what is happening.

### Must monitor

- Active model
- Tool calls
- Errors
- Voice status
- Home Assistant status
- n8n status
- ComfyUI status
- Memory status
- Cloud tunnel status
- API health
- Service mode status
- Last successful action
- Last failed action

### Must log

```text
timestamp
active_user
input_type
transcript
intent
model_used
tools_called
tool_results
response_mode
spoken_response
screen_response
errors
verification_result
memory_updates
```

### Why it matters

When something breaks, Rex should help diagnose itself instead of making James dig through mystery logs like an archaeologist with admin rights.

### Confidence

**High**  
**What would raise confidence:** A working dashboard with live status and searchable logs.

---

## 21. Developer and Maintenance Layer

Rex must be maintainable.

### Required developer features

| Feature | Purpose |
|---|---|
| Clear folder structure | Avoid chaos |
| Config-driven behavior | Avoid hardcoded values |
| Plugin architecture | Add tools without rewriting core |
| Type hints | Reduce bugs |
| Unit tests | Catch regressions |
| Integration tests | Validate real workflows |
| Logging | Diagnose failures |
| Health checks | Know what is running |
| Version control | Track changes |
| Documentation | Future James does not hate Past James |
| CI coverage | Catch breakage before merging |
| Windows-first validation | Avoid local-only Windows surprises |

### Suggested folder structure

```text
rex/
  core/
    orchestrator.py
    intent_router.py
    model_router.py
    planner.py
    response_manager.py

  tools/
    home_assistant.py
    web_search.py
    browser.py
    filesystem.py
    code_runner.py
    n8n.py
    woocommerce.py
    comfyui.py
    plex.py
    email.py
    calendar.py

  memory/
    short_term.py
    long_term.py
    vector_store.py
    user_profiles.py
    memory_policy.py

  voice/
    wakeword.py
    stt.py
    tts.py
    voice_loop.py
    speaker_id.py

  safety/
    permissions.py
    confirmation.py
    secrets.py
    audit_log.py

  integrations/
    home_assistant/
    n8n/
    woo/
    comfyui/
    plex/

  ui/
    api.py
    websocket.py

  tests/
    unit/
    integration/
    fixtures/

config/
  rex_config.json
  tools.json
  permissions.json
  aliases.json

data/
  memory/
  logs/
  vector_db/
  backups/
```

---

## 22. Evaluation Layer

This is how Rex gets better without becoming a mess.

### Required test categories

| Test Type | Example |
|---|---|
| Smart home tests | "Turn on kitchen light" maps correctly |
| Voice tests | Wake word and STT accuracy |
| Tool tests | API calls succeed |
| Memory tests | Recalls correct user preference |
| Safety tests | Refuses risky actions without confirmation |
| Regression tests | Old working features stay working |
| Latency tests | Voice replies stay fast |
| Hallucination tests | Does not claim actions it did not perform |
| Error recovery tests | Explains failures and suggests next steps |

### Golden test set

```text
Turn on the kitchen lights.
Turn off all downstairs lights.
Set thermostat to 72.
Play Star Trek TNG season 7 episode 25.
What products did I add to Nasteeshirts this week?
Create a Facebook post for the newest shirt.
Why did the last n8n workflow fail?
Search the Rex repo for the wake word threshold.
Summarize what changed in the latest commit.
```

### Why it matters

If Rex cannot pass its most common real-world requests, then "smart" is just a costume.

### Confidence

**High**  
**What would raise confidence:** Automated golden tests running in CI and locally.

---

## 23. Recommended Build Roadmap

### Phase 1: Make Rex Reliable

1. Stable service mode
2. Reliable voice loop
3. Home Assistant command execution
4. Tool verification
5. Proper voice-only response mode
6. Logging and error reporting
7. Basic memory
8. Model routing

### Phase 2: Make Rex Useful

1. n8n integration
2. Nasteeshirts workflows
3. WooCommerce integration
4. ComfyUI integration
5. File and repo search
6. Browser and web search
7. Task scheduling
8. Plex and media control
9. Calendar, email, and notifications
10. OpenClaw / ClawHub capability bridge
11. Dynamic plugin and skill discovery
12. External tool verification layer

### Phase 3: Make Rex Smart

1. Long-term memory
2. Planning engine
3. Skill system
4. Proactive routines
5. Multi-user support
6. Feedback learning
7. Advanced retrieval
8. Vision model support

### Phase 4: Make Rex Feel Commercial

1. Polished GUI
2. Mobile access
3. Voice interruption
4. Fast streaming responses
5. Dashboard
6. Permissions UI
7. One-click installer
8. Backup and restore
9. Plugin manager
10. Guided setup flow

---

## 24. Non-Negotiable Acceptance Standard

For any real-world action, Rex should not report success until it has either:

1. Verified the result, or
2. Clearly stated that it only attempted the action and could not verify it.

### Required wording pattern

When verified:

> "Done. I turned on the kitchen lights and confirmed they are on."

When attempted but not verified:

> "I sent the command to turn on the kitchen lights, but I could not verify the final state because Home Assistant did not return an updated status."

When failed:

> "I could not turn on the kitchen lights. Home Assistant returned a connection error. The most likely cause is that the API server is offline or the token is invalid."

This standard should apply everywhere:

- Home Assistant
- n8n
- ComfyUI
- WooCommerce
- File edits
- Code changes
- Service startup
- Scheduling
- Notifications
- Email or messaging
- Git operations
