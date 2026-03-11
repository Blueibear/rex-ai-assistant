# PRD: Rex AI Assistant Completion

IMPORTANT

Stories must remain atomic.

If a story becomes too large, it must be split into smaller stories before implementation.

A story is complete only when all acceptance criteria checkboxes are checked.

---

## Introduction

Rex is a locally-hosted AI assistant with voice interaction, multi-provider LLM support, OS automation, smart home integration, and a web-based dashboard. This PRD covers the work required to bring the Rex codebase from its current state to a fully functional, installable, and tested system — across repository stability, code quality, runtime validation, integrations, and UI.

---

## Goals

- Establish a stable, importable, installable codebase with clean tests and no runtime errors
- Provide reliable CLI access to all Rex capabilities
- Enable voice interaction via wake word detection, speech-to-text, and text-to-speech
- Support multiple LLM providers (OpenAI, Anthropic, local) with configurable routing
- Deliver a web-based dashboard with chat and voice interfaces
- Integrate with Home Assistant, Plex, GitHub, email, and calendar
- Support extensibility through a plugin architecture and workflow engine
- Enable OS-level automation including application launching and browser control
- Maintain queryable long-term memory and a local knowledge base
- Ship with CI, documentation, and enforced code quality standards

---

## Non-Goals

- No mobile app interface
- No multi-user account management or authentication beyond the dashboard login
- No cloud hosting, deployment pipelines, or container orchestration
- No third-party plugin marketplace or plugin distribution
- No real-time collaboration or shared sessions between users
- No billing, usage metering, or API rate limiting

---

# PHASE 1 — Repository Stability

### US-001: Restore test isolation

**Description:** As a developer, I want tests to use temporary directories so that running the test suite does not modify tracked repository files.

**Acceptance Criteria:**
- [x] tests use temporary directories
- [x] repository files remain unchanged after pytest
- [x] git status remains clean
- [x] Typecheck passes

---

### US-002: Validate package imports

**Description:** As a developer, I want all package imports to succeed so that the codebase loads cleanly without errors.

**Acceptance Criteria:**
- [x] `import rex` succeeds
- [x] no circular imports occur
- [x] CLI loads without runtime error
- [x] Typecheck passes

---

### US-003: Fix planner registry mismatch

**Description:** As a developer, I want the planner registry to match the router interface so that the planner can be invoked without runtime errors.

**Acceptance Criteria:**
- [x] planner imports successfully
- [x] registry interfaces match router usage
- [x] runtime errors removed
- [x] Typecheck passes

---

# PHASE 2 — Install Reliability

### US-004: Verify standard installation

**Description:** As a developer, I want `pip install .` to succeed so that users can install Rex without manual intervention.

**Acceptance Criteria:**
- [x] `pip install .` succeeds
- [x] CLI entrypoints available
- [x] dependency conflicts resolved
- [x] Typecheck passes

---

### US-005: Verify editable install

**Description:** As a developer, I want `pip install -e .` to succeed so that development workflows function correctly.

**Acceptance Criteria:**
- [x] `pip install -e .` succeeds
- [x] CLI entrypoints load
- [x] Typecheck passes

---

### US-006: Validate optional extras

**Description:** As a developer, I want optional extras to install cleanly so that users can opt into additional capabilities without breaking the base install.

**Acceptance Criteria:**
- [x] base install works without extras
- [x] extras install successfully
- [x] missing extras handled gracefully
- [x] Typecheck passes

---

# PHASE 3 — Code Quality

### US-007: Fix Ruff violations

**Description:** As a developer, I want all Ruff lint violations resolved so that the codebase meets code quality standards.

**Acceptance Criteria:**
- [x] `ruff check rex` returns zero errors
- [x] unnecessary ignores removed
- [x] Typecheck passes

---

### US-008: Apply Black formatting

**Description:** As a developer, I want the codebase formatted with Black so that code style is consistent across all files.

**Acceptance Criteria:**
- [x] code formatted with Black
- [x] `black --check .` passes
- [x] Typecheck passes

---

### US-009: Fix MyPy errors

**Description:** As a developer, I want all MyPy errors resolved so that the codebase is fully type-safe.

**Acceptance Criteria:**
- [x] `mypy rex` returns zero errors
- [x] missing type hints added
- [x] Typecheck passes

---

# PHASE 4 — CLI Runtime

### US-010: Validate CLI entrypoints

**Description:** As a user, I want all CLI entrypoints to launch successfully so that I can access Rex from the command line.

**Acceptance Criteria:**
- [x] `rex` CLI launches
- [x] `rex-config` launches
- [x] `rex-agent` launches
- [x] Typecheck passes

---

### US-011: Validate doctor command

**Description:** As a user, I want the doctor command to run diagnostics so that I can verify my installation is healthy.

**Acceptance Criteria:**
- [x] doctor command runs
- [x] diagnostics printed
- [x] failures handled safely
- [x] Typecheck passes

---

### US-012: Validate configuration loading

**Description:** As a user, I want configuration to load from file and environment variables so that I can customize Rex behavior without code changes.

**Acceptance Criteria:**
- [x] config loads from config file
- [x] environment overrides supported
- [x] missing config handled safely
- [x] Typecheck passes

---

# PHASE 5 — LLM Providers

### US-013: OpenAI provider

**Description:** As a developer, I want the OpenAI provider to execute prompts reliably so that Rex can generate responses using OpenAI models.

**Acceptance Criteria:**
- [x] provider initializes
- [x] prompt execution works
- [x] response returned
- [x] failure handled gracefully
- [x] Typecheck passes

---

### US-014: Anthropic provider

**Description:** As a developer, I want the Anthropic provider to execute prompts reliably so that Rex can generate responses using Claude models.

**Acceptance Criteria:**
- [x] provider initializes
- [x] prompt execution works
- [x] response returned
- [x] Typecheck passes

---

### US-015: Local LLM provider

**Description:** As a developer, I want the local LLM provider to execute prompts so that Rex can run without cloud API dependencies.

**Acceptance Criteria:**
- [x] local model reachable
- [x] prompt execution works
- [x] response returned
- [x] Typecheck passes

---

### US-016: Provider routing

**Description:** As a developer, I want provider routing to be configurable so that Rex can switch between LLM backends based on user preference.

**Acceptance Criteria:**
- [x] provider selection configurable
- [x] routing logic implemented
- [x] fallback behavior works
- [x] Typecheck passes

---

# PHASE 6 — Voice Assistant

### US-017: Wake word detection

**Description:** As a user, I want wake word detection to trigger listening so that I can activate Rex hands-free.

**Acceptance Criteria:**
- [x] wake word detection triggers listening
- [x] microphone stream initializes
- [x] wake word does not trigger on common conversational speech
- [x] Typecheck passes

---

### US-018: Speech to text pipeline

**Description:** As a user, I want speech captured and transcribed so that I can interact with Rex by voice.

**Acceptance Criteria:**
- [x] microphone audio captured
- [x] audio converted to transcript
- [x] transcript matches spoken test phrase on at least 3 consecutive attempts
- [x] Typecheck passes

---

### US-019: Text to speech pipeline

**Description:** As a user, I want Rex responses spoken aloud so that I receive audio feedback without looking at a screen.

**Acceptance Criteria:**
- [x] TTS engine loads
- [x] audio generated
- [x] audio plays automatically
- [x] Typecheck passes

---

### US-020: Full voice interaction loop

**Description:** As a user, I want to speak to Rex and receive a spoken response so that I can have a complete voice-driven interaction.

**Acceptance Criteria:**
- [x] wake word triggers listening
- [x] STT produces transcript
- [x] LLM response generated
- [x] response spoken aloud
- [x] Typecheck passes

---

# PHASE 7 — Tool and Capability Framework

### US-021: Tool registry

**Description:** As a developer, I want tools to register with the tool registry so that Rex can discover and invoke available capabilities.

**Acceptance Criteria:**
- [x] tools register correctly
- [x] tool metadata stored
- [x] duplicate tools prevented
- [x] Typecheck passes

---

### US-022: Tool router

**Description:** As a developer, I want tool routing to dispatch execution correctly so that user requests reach the appropriate tool.

**Acceptance Criteria:**
- [x] tools routed correctly
- [x] execution dispatched
- [x] errors handled safely
- [x] Typecheck passes

---

### US-023: Capability discovery

**Description:** As a user, I want Rex to discover and expose its capabilities so that I know what actions are available.

**Acceptance Criteria:**
- [x] capabilities enumerated
- [x] tools discoverable
- [x] capability metadata exposed
- [x] Typecheck passes

---

# PHASE 8 — Planner and Reasoning

### US-024: Planner initialization

**Description:** As a developer, I want the planner to initialize successfully so that multi-step reasoning is available.

**Acceptance Criteria:**
- [x] planner loads successfully
- [x] dependencies resolved
- [x] planner callable
- [x] Typecheck passes

---

### US-025: Planner task execution

**Description:** As a user, I want the planner to accept tasks and execute tool calls so that Rex can complete multi-step actions autonomously.

**Acceptance Criteria:**
- [x] tasks accepted
- [x] task plan generated
- [x] tool calls executed
- [x] Typecheck passes

---

# PHASE 9 — Workflow Engine

### US-026: Workflow definitions

**Description:** As a developer, I want workflow definitions stored and validated so that automated sequences can be reliably triggered.

**Acceptance Criteria:**
- [x] workflows defined
- [x] schema validated
- [x] workflows stored
- [x] Typecheck passes

---

### US-027: Workflow runner

**Description:** As a user, I want workflows to execute step by step so that Rex can complete multi-stage automations.

**Acceptance Criteria:**
- [x] workflows executed
- [x] step transitions work
- [x] errors handled
- [x] Typecheck passes

---

# PHASE 10 — Event System

### US-028: Event bus

**Description:** As a developer, I want an event bus to publish events and notify subscribers so that system components communicate without tight coupling.

**Acceptance Criteria:**
- [x] events published
- [x] subscribers receive events
- [x] event propagation works
- [x] Typecheck passes

---

### US-029: Event triggers

**Description:** As a developer, I want events to trigger workflows so that Rex can react to system events automatically.

**Acceptance Criteria:**
- [x] triggers registered
- [x] events trigger workflows
- [x] errors logged
- [x] Typecheck passes

---

# PHASE 11 — Notification System

### US-030: Notification routing

**Description:** As a developer, I want notifications routed to the correct destination so that users receive alerts through their preferred channel.

**Acceptance Criteria:**
- [x] notifications generated
- [x] routing rules applied
- [x] delivery attempted
- [x] Typecheck passes

---

### US-031: Dashboard notifications

**Description:** As a user, I want dashboard notifications streamed in real time so that I see alerts without refreshing the page.

**Acceptance Criteria:**
- [x] SSE endpoint works
- [x] notifications streamed
- [x] disconnect handled
- [x] Typecheck passes

---

# PHASE 12 — Memory System

### US-032: Memory storage

**Description:** As a user, I want Rex to persist memories so that context from previous interactions is available in future sessions.

**Acceptance Criteria:**
- [x] memory records saved
- [x] storage persistent
- [x] retrieval possible
- [x] Typecheck passes

---

### US-033: User profiles

**Description:** As a user, I want Rex to store and retrieve my preferences so that it adapts to my personal configuration.

**Acceptance Criteria:**
- [x] user profiles created
- [x] preferences stored
- [x] retrieval works
- [x] Typecheck passes

---

# PHASE 13 — Plugin Architecture

### US-034: Plugin discovery

**Description:** As a developer, I want plugins discovered automatically from the plugin folder so that capabilities can be added without modifying core code.

**Acceptance Criteria:**
- [x] plugin loader scans plugin folder
- [x] plugins detected
- [x] plugin metadata loaded
- [x] Typecheck passes

---

### US-035: Plugin execution

**Description:** As a developer, I want plugin tools to be callable and isolated so that plugin failures do not crash the assistant.

**Acceptance Criteria:**
- [x] plugin tools callable
- [x] failures isolated
- [x] plugins unload safely
- [x] Typecheck passes

---

# PHASE 14 — Automation Engine

### US-036: Scheduler

**Description:** As a user, I want tasks scheduled and executed automatically so that Rex acts without requiring manual triggers.

**Acceptance Criteria:**
- [x] scheduler initializes
- [x] tasks scheduled
- [x] tasks executed
- [x] Typecheck passes

---

### US-037: Automation registry

**Description:** As a developer, I want automations stored and retrievable so that scheduled tasks persist across restarts.

**Acceptance Criteria:**
- [x] automations stored
- [x] automations retrieved
- [x] persistence works
- [x] Typecheck passes

---

# PHASE 15 — OS Automation

### US-038: Application launching

**Description:** As a user, I want Rex to launch applications on my behalf so that I can control my desktop by voice or text.

**Acceptance Criteria:**
- [x] applications launch
- [x] execution verified
- [x] failures handled
- [x] Typecheck passes

---

### US-039: Browser automation

**Description:** As a user, I want Rex to automate browser actions so that I can delegate web tasks.

**Acceptance Criteria:**
- [x] browser launches
- [x] navigation works
- [x] page actions executed
- [x] Typecheck passes

---

# PHASE 16 — Knowledge Base

### US-040: Knowledge ingestion

**Description:** As a developer, I want documents ingested and indexed so that Rex can answer questions from local knowledge.

**Acceptance Criteria:**
- [x] documents ingested
- [x] data indexed
- [x] query containing a keyword from an indexed document returns that document in results
- [x] Typecheck passes

---

### US-041: Knowledge queries

**Description:** As a user, I want to query the knowledge base and receive results so that Rex can surface stored information.

**Acceptance Criteria:**
- [x] queries executed
- [x] query returns at least one result when indexed content contains the queried term
- [x] errors handled
- [x] Typecheck passes

---

# PHASE 17 — Home Assistant Integration

### US-042: Home Assistant API connection

**Description:** As a developer, I want Rex to connect to the Home Assistant API so that smart home devices are accessible.

**Acceptance Criteria:**
- [x] API reachable
- [x] authentication works
- [x] entities retrieved
- [x] Typecheck passes

---

### US-043: Device control

**Description:** As a user, I want Rex to control lights and switches so that I can manage my home by voice.

**Acceptance Criteria:**
- [x] lights controlled
- [x] switches controlled
- [x] responses returned
- [x] Typecheck passes

---

# PHASE 18 — Messaging

### US-044: Email integration

**Description:** As a user, I want Rex to send email on my behalf so that I can compose and send messages by voice or text.

**Acceptance Criteria:**
- [x] email backend connects
- [x] send works
- [x] errors handled
- [x] Typecheck passes

---

### US-045: Calendar integration

**Description:** As a user, I want Rex to retrieve and create calendar events so that I can manage my schedule through conversation.

**Acceptance Criteria:**
- [x] events retrieved
- [x] events created
- [x] errors handled
- [x] Typecheck passes

---

# PHASE 19 — Dashboard

### US-046: Dashboard server

**Description:** As a developer, I want the dashboard server to start and serve a health endpoint so that the UI has a reliable backend.

**Acceptance Criteria:**
- [x] server starts
- [x] API reachable
- [x] health endpoint works
- [x] Typecheck passes

---

### US-047: Dashboard authentication

**Description:** As a user, I want to log into the dashboard so that my data and configuration are protected.

**Acceptance Criteria:**
- [x] login works
- [x] sessions created
- [x] invalid logins rejected
- [x] Typecheck passes

---

# PHASE 20 — Plex Integration

### US-048: Plex API client

**Description:** As a developer, I want Rex to connect to the Plex API so that media library data is accessible.

**Acceptance Criteria:**
- [x] Plex reachable
- [x] libraries retrieved
- [x] authentication works
- [x] Typecheck passes

---

### US-049: Plex playback control

**Description:** As a user, I want Rex to control Plex playback so that I can manage media by voice.

**Acceptance Criteria:**
- [x] play command works
- [x] pause command works
- [x] stop command works
- [x] Typecheck passes

---

# PHASE 21 — Web UI

### US-050: Web UI server

**Description:** As a user, I want the web UI to load and render so that I can access Rex from a browser.

**Acceptance Criteria:**
- [x] UI server starts
- [x] UI accessible
- [x] interface renders
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-051: Chat interface

**Description:** As a user, I want to send messages and see responses in the chat interface so that I can interact with Rex without a terminal.

**Acceptance Criteria:**
- [x] messages sent
- [x] responses displayed
- [x] session maintained
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-052: Voice interface

**Description:** As a user, I want to speak through the web UI and hear responses so that I have a browser-based voice interface.

**Acceptance Criteria:**
- [x] microphone input works
- [x] audio sent to backend
- [x] response audio plays
- [x] Typecheck passes
- [x] Verify changes work in browser

---

# PHASE 22 — Security

### US-053: Secret management

**Description:** As a developer, I want secrets loaded from the environment so that credentials are never stored in the repository.

**Acceptance Criteria:**
- [x] secrets loaded from environment
- [x] secrets not stored in repo
- [x] missing secrets detected
- [x] Typecheck passes

---

### US-054: API key validation

**Description:** As a developer, I want API keys validated on each request so that unauthorized access is rejected.

**Acceptance Criteria:**
- [x] API keys validated
- [x] unauthorized rejected
- [x] failures logged
- [x] Typecheck passes

---

# PHASE 23 — GitHub Integration

### US-055: GitHub API client

**Description:** As a developer, I want Rex to connect to the GitHub API so that repository data is accessible.

**Acceptance Criteria:**
- [x] GitHub reachable
- [x] repos listed
- [x] authentication works
- [x] Typecheck passes

---

### US-056: GitHub actions

**Description:** As a user, I want Rex to retrieve issues and trigger commits so that I can interact with GitHub by voice or text.

**Acceptance Criteria:**
- [x] issues retrieved
- [x] commits triggered
- [x] errors handled
- [x] Typecheck passes

---

# PHASE 24 — CI and Documentation

### US-057: CI pipeline

**Description:** As a developer, I want CI to run lint, typecheck, and tests on every PR so that regressions are caught before merging.

**Acceptance Criteria:**
- [x] CI runs on PR
- [x] lint executed
- [x] typecheck executed
- [x] tests executed
- [x] Typecheck passes

---

### US-058: Documentation updates

**Description:** As a developer, I want documentation updated to match the current codebase so that developers can onboard without digging through source code.

**Acceptance Criteria:**
- [x] README updated
- [x] install docs updated
- [x] CLI usage documented
- [x] developer docs updated
- [x] Typecheck passes

---

# PHASE 25 — Tool Execution Validation

### US-059: Tool execution logging

**Description:** As a developer, I want tool executions logged with timestamps and parameters so that I can audit and debug tool behavior.

**Acceptance Criteria:**
- [x] tool executions logged
- [x] execution timestamps recorded
- [x] tool parameters stored
- [x] Typecheck passes

---

### US-060: Tool execution error handling

**Description:** As a developer, I want tool failures captured and recorded so that the assistant remains stable when a tool errors.

**Acceptance Criteria:**
- [x] tool failures captured
- [x] failure reason recorded
- [x] execution does not crash assistant
- [x] Typecheck passes

---

# PHASE 26 — Planner Improvements

### US-061: Planner prompt generation

**Description:** As a developer, I want the planner to build prompts that include the task description and available tools so that the LLM has full context for reasoning.

**Acceptance Criteria:**
- [x] planner builds prompts correctly
- [x] task description included
- [x] available tools listed
- [x] Typecheck passes

---

### US-062: Planner tool selection

**Description:** As a developer, I want the planner to select the most appropriate tool from multiple options so that tasks are executed efficiently.

**Acceptance Criteria:**
- [x] planner selects appropriate tools
- [x] multiple tool options supported
- [x] tool selection validated
- [x] Typecheck passes

---

### US-063: Planner fallback behavior

**Description:** As a developer, I want the planner to fall back gracefully when a tool fails so that multi-step tasks recover where possible.

**Acceptance Criteria:**
- [x] planner detects tool failure
- [x] alternate strategy attempted
- [x] errors logged
- [x] Typecheck passes

---

# PHASE 27 — Workflow Enhancements

### US-064: Workflow state persistence

**Description:** As a developer, I want workflow state saved and restored across restarts so that long-running workflows survive interruptions.

**Acceptance Criteria:**
- [x] workflow state saved
- [x] state restored after restart
- [x] step progress tracked
- [x] Typecheck passes

---

### US-065: Workflow step validation

**Description:** As a developer, I want workflow step inputs validated before execution so that invalid configurations are rejected early.

**Acceptance Criteria:**
- [x] step inputs validated
- [x] invalid steps rejected
- [x] workflow execution halted on failure
- [x] Typecheck passes

---

# PHASE 28 — Event System Reliability

### US-066: Event subscription validation

**Description:** As a developer, I want event subscribers managed safely so that duplicate subscriptions are prevented and removal is supported.

**Acceptance Criteria:**
- [x] subscribers registered
- [x] subscriber removal supported
- [x] duplicate subscriptions prevented
- [x] Typecheck passes

---

### US-067: Event queue stability

**Description:** As a developer, I want the event queue to handle load safely so that events are not lost and processing remains sequential.

**Acceptance Criteria:**
- [x] events queued safely
- [x] queue overflow prevented
- [x] events processed sequentially
- [x] Typecheck passes

---

# PHASE 29 — Notification Delivery

### US-068: Notification persistence

**Description:** As a developer, I want notifications persisted to the database with timestamps so that delivery history is auditable.

**Acceptance Criteria:**
- [x] notifications stored in database
- [x] notifications retrieved
- [x] timestamps recorded
- [x] Typecheck passes

---

### US-069: Notification retry logic

**Description:** As a developer, I want failed notifications retried with a limited attempt count so that transient failures recover without infinite loops.

**Acceptance Criteria:**
- [x] failed notifications retried
- [x] retry attempts limited
- [x] failures logged
- [x] Typecheck passes

---

# PHASE 30 — Memory Retrieval

### US-070: Memory search

**Description:** As a user, I want to search memory entries and receive results so that past context is retrievable on demand.

**Acceptance Criteria:**
- [x] memory entries searchable
- [x] search for a stored memory keyword returns that memory entry in results
- [x] query failures handled
- [x] Typecheck passes

---

### US-071: Memory cleanup

**Description:** As a developer, I want expired memories cleaned up on a schedule so that the memory store remains lean and performant.

**Acceptance Criteria:**
- [x] expired memories removed
- [x] cleanup scheduled
- [x] memory store compacted
- [x] Typecheck passes

---

# PHASE 31 — OS Automation Reliability

### US-072: Process monitoring

**Description:** As a developer, I want launched processes monitored for crashes so that Rex can detect and restart failed processes.

**Acceptance Criteria:**
- [x] launched processes monitored
- [x] crashes detected
- [x] process restart supported
- [x] Typecheck passes

---

### US-073: File system safety

**Description:** As a developer, I want file system operations validated and unsafe paths rejected so that OS automation cannot cause unintended damage.

**Acceptance Criteria:**
- [x] file operations validated
- [x] unsafe paths rejected
- [x] errors handled
- [x] Typecheck passes

---

# PHASE 32 — Knowledge Base Improvements

### US-074: Document indexing

**Description:** As a developer, I want documents indexed and the index stored persistently so that knowledge is available after restart.

**Acceptance Criteria:**
- [x] documents indexed
- [x] index stored
- [x] indexing failures logged
- [x] Typecheck passes

---

### US-075: Knowledge refresh

**Description:** As a developer, I want the knowledge index refreshed when documents change so that search results stay current.

**Acceptance Criteria:**
- [x] documents updated
- [x] index refreshed
- [x] stale entries removed
- [x] Typecheck passes

---

# PHASE 33 — Web UI Reliability

### US-076: UI error handling

**Description:** As a user, I want frontend errors displayed clearly and backend errors logged so that problems are diagnosable without digging through server output.

**Acceptance Criteria:**
- [ ] frontend errors detected
- [ ] error messages displayed
- [ ] backend errors logged
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-077: UI reconnect behavior

**Description:** As a user, I want the UI to reconnect automatically after a connection loss so that temporary network issues do not require a manual page refresh.

**Acceptance Criteria:**
- [ ] SSE reconnect supported
- [ ] reconnect attempts limited
- [ ] UI recovers after connection loss
- [ ] Typecheck passes
- [ ] Verify changes work in browser