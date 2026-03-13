# PRD: Rex AI Assistant Completion

IMPORTANT

Stories must remain atomic.

If a story becomes too large, it must be split into smaller stories before implementation.

A story is complete only when all acceptance criteria checkboxes are checked.

 IMPORTANT

 Stories must remain atomic.

 If a story becomes too large, it must be split into smaller stories before implementation.

 A story is complete only when all acceptance criteria checkboxes are checked.

+## Codex task selection rule
+
+- A "task" means one full User Story section (US-###), not an individual acceptance criteria checkbox line.
+- Choose the first US-### that contains any unchecked acceptance criteria ([ ]).
+- Complete the full story in one iteration. If it cannot be completed in one iteration, split the story first.
+- Only mark acceptance criteria as [x] when the full US is complete and tests pass.

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
- [x] frontend errors detected
- [x] error messages displayed
- [x] backend errors logged
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-077: UI reconnect behavior

**Description:** As a user, I want the UI to reconnect automatically after a connection loss so that temporary network issues do not require a manual page refresh.

**Acceptance Criteria:**
- [x] SSE reconnect supported
- [x] reconnect attempts limited
- [x] UI recovers after connection loss
- [x] Typecheck passes
- [x] Verify changes work in browser

---

# PHASE 34 — Email Triage & Scheduling (beta — stub/mock data only)

> **Beta scope:** All stories in this phase use stub implementations and mock data. No live email or calendar credentials are required. The stub interfaces must match the real backend interfaces so live credentials can be wired in without changing calling code.

### US-078: Email inbox stub and mock data

**Description:** As a developer, I want a stub email inbox that returns mock data so that triage features can be built and tested without live credentials.

**Acceptance Criteria:**
- [x] `EmailInboxStub` class (or equivalent) exists and returns a list of mock email objects
- [x] mock emails cover at least three categories: urgent, action_required, and fyi
- [x] stub implements the same interface as the real email backend (US-044)
- [x] tests can instantiate and query the stub without any live credentials or network calls
- [x] Typecheck passes

---

### US-079: Email triage categorization

**Description:** As a user, I want incoming emails automatically categorized (urgent, action_required, fyi, newsletter) so that I can see what needs attention without reading everything.

**Acceptance Criteria:**
- [x] triage assigns one of four categories: urgent, action_required, fyi, newsletter
- [x] categorization logic uses sender address, subject keywords, and body patterns
- [x] triage results are queryable (e.g., "show urgent emails" returns only urgent-tagged items)
- [x] test using mock inbox data confirms at least one email correctly categorized into each category
- [x] Typecheck passes

---

### US-080: Email triage rules engine

**Description:** As a developer, I want triage rules stored in config and evaluated in priority order so that categorization can be customized per user without code changes.

**Acceptance Criteria:**
- [x] triage rules stored in config file (JSON or YAML)
- [x] rules support matching on: sender address, subject pattern, body keyword
- [x] rules evaluated in declared priority order; first match wins
- [x] adding or modifying a rule takes effect without restarting or modifying source code
- [x] Typecheck passes

---

### US-081: Calendar free/busy stub

**Description:** As a developer, I want a stub calendar that returns mock free/busy data so that scheduling features can be built and tested without live credentials.

**Acceptance Criteria:**
- [x] `CalendarStub` class returns mock free/busy blocks for a configurable date range
- [x] stub implements the same interface as the real calendar backend (US-045)
- [x] tests can query availability without any live credentials or network calls
- [x] Typecheck passes

---

### US-082: Free time finder

**Description:** As a user, I want Rex to find available meeting slots from my calendar so that I can ask "when am I free?" and get usable suggestions.

**Acceptance Criteria:**
- [x] given a date range and meeting duration, returns a list of available time slots
- [x] overlapping calendar events are excluded from returned slots
- [x] returns at least three candidate slots when the calendar is not fully booked
- [x] works correctly against stub/mock calendar data in beta
- [x] Typecheck passes

---

### US-083: Meeting invite scaffold

**Description:** As a user, I want Rex to draft a meeting invite with attendees, time, and agenda from a natural language request so that I can schedule by describing what I want.

**Acceptance Criteria:**
- [x] `MeetingInvite` data structure contains: title, attendees list, start time, end time, agenda
- [x] Rex can populate all invite fields from a natural language description
- [x] completed invite is displayed to the user for review before any action
- [x] stub send logs the invite and returns success without calling any real calendar API
- [x] Typecheck passes

---

# PHASE 35 — SMS Multi-Channel Messaging (beta — stub scaffolding)

> **Beta scope:** All stories in this phase are stub scaffolding. No real SMS messages are sent. Real delivery requires Twilio credentials to be wired into the `TwilioAdapter` interface defined in US-086.

### US-084: SMS send stub

**Description:** As a developer, I want a stub SMS sender that logs outbound messages so that SMS-triggered workflows can be built and tested without Twilio credentials.

**Acceptance Criteria:**
- [x] `SmsSenderStub` class accepts a phone number and message body
- [x] sent messages written to a structured in-memory log accessible for test assertions
- [x] stub implements the same interface as the real Twilio adapter (US-086)
- [x] calling `send` on the stub makes no network calls
- [x] Typecheck passes

---

### US-085: SMS receive stub

**Description:** As a developer, I want a stub SMS receiver that can inject test inbound messages so that inbound SMS handling can be built and tested without a live Twilio webhook.

**Acceptance Criteria:**
- [x] `SmsReceiverStub` class exposes a method to inject a test inbound message
- [x] injected messages are routed through the same handler as real inbound SMS would be
- [x] handler produces a response or triggers the expected downstream action
- [x] Typecheck passes

---

### US-086: Twilio adapter interface

**Description:** As a developer, I want a well-defined Twilio adapter interface so that the stub and the real Twilio client are interchangeable without changing any calling code.

**Acceptance Criteria:**
- [x] `TwilioAdapter` abstract class or Protocol defined with at minimum `send_sms(to: str, body: str)` signature
- [x] `SmsSenderStub` fully implements `TwilioAdapter`
- [x] swapping stub for a real Twilio client requires no changes outside the adapter registration point
- [x] Typecheck passes

---

### US-087: Multi-channel message router

**Description:** As a developer, I want outbound messages routed to the correct channel (dashboard, email, SMS) based on configuration so that Rex can deliver messages where the user prefers without hardcoding a channel.

**Acceptance Criteria:**
- [x] router accepts a message payload and a target channel identifier
- [x] routes correctly to dashboard, email, and SMS backends based on channel value
- [x] unknown or unconfigured channel raises a handled error and does not crash the assistant
- [x] active channel configurable without code changes
- [x] Typecheck passes

---

# PHASE 36 — Smart Notifications (beta — stub scaffolding)

> **Beta scope:** Digest delivery, escalation, and quiet-hour release are stub scaffolding — they log output rather than making real deliveries. Core priority tagging and routing rules are fully implemented.

### US-088: Notification priority levels

**Description:** As a developer, I want notifications to carry a priority level so that routing and delivery decisions can be based on urgency rather than treating all notifications equally.

**Acceptance Criteria:**
- [x] `NotificationPriority` enum defined with values: critical, high, medium, low
- [x] all notification creation paths accept a `priority` parameter
- [x] priority stored alongside notification record in the database
- [x] existing notifications without a stored priority default to `medium` on read
- [x] Typecheck passes

---

### US-089: Priority routing rules

**Description:** As a developer, I want routing rules that deliver critical and high notifications immediately while queuing medium and low ones so that users are not interrupted by low-priority alerts.

**Acceptance Criteria:**
- [x] critical and high priority notifications dispatched to configured delivery channels immediately on creation
- [x] medium and low priority notifications placed in the digest queue instead of immediate delivery
- [x] routing rules configurable without code changes
- [x] unit test confirms a critical notification bypasses the digest queue
- [x] unit test confirms a low notification is placed in the digest queue
- [x] Typecheck passes

---

### US-090: Digest mode

**Description:** As a user, I want low-priority notifications batched into periodic digests so that I receive a single summary instead of many individual interruptions.

**Acceptance Criteria:**
- [x] digest job runs on a configurable interval (default: 60 minutes)
- [x] digest collects all queued medium and low notifications since the last digest run
- [x] digest payload delivered to the dashboard notification endpoint as a single grouped message
- [x] digest job logs output when no real delivery backend is configured (beta stub behavior)
- [x] digest queue is cleared after each successful run
- [x] Typecheck passes

---

### US-091: Quiet hours

**Description:** As a user, I want to configure quiet hours so that non-critical notifications are held until I'm available rather than interrupting me at night or during focus time.

**Acceptance Criteria:**
- [x] quiet hours configured as start time and end time in user config
- [x] non-critical (medium, low) notifications generated during quiet hours are held in queue
- [x] critical notifications bypass quiet hours and deliver immediately regardless of schedule
- [x] held notifications are released and delivered when quiet hours end
- [x] Typecheck passes

---

### US-092: Auto-escalation

**Description:** As a developer, I want unacknowledged high-priority notifications to escalate after a configurable timeout so that important alerts are not silently missed.

**Acceptance Criteria:**
- [x] escalation timeout configurable per priority level (default: 15 minutes for high)
- [x] escalation job checks for unacknowledged high-priority notifications past their timeout
- [x] each escalation attempt logged with timestamp, notification ID, and attempt number
- [x] escalation stops after a configurable maximum attempt count (default: 3)
- [x] escalation stub logs events without making real deliveries in beta
- [x] Typecheck passes

---

# PRD: Rex Production Readiness Review
---

## Introduction

This PRD covers the full production readiness review of the Rex codebase. The goal is a clean, audited, well-tested, observable, and deployable system with no open security issues, no missing operational documentation, and a passing production smoke test suite. Work in this PRD begins after the core feature buildout (PRD.md) and technical debt cleanup (PRD-repo-quality.md) are complete.

No new features are introduced here. Every story is purely a quality, security, reliability, or operational concern.

---

## Goals

- Identify and remediate all critical and high security vulnerabilities before deployment
- Establish and enforce a minimum test coverage threshold across all modules
- Ensure consistent, structured error handling and logging throughout the system
- Harden configuration and secrets management for production environments
- Validate database connection reliability, migration state, and query safety
- Produce a uniform, well-documented API surface with input validation and rate limiting
- Establish a performance baseline with no known blocking I/O in async paths
- Deliver complete operational documentation: deployment guide, config reference, runbook, API reference
- Verify the full system starts cleanly, accepts traffic, and passes a production smoke test suite
- Produce a signed-off production readiness checklist with no open items

---

## Non-Goals

- No new features or integrations
- No refactoring of working code purely for aesthetics
- No migration to a different framework, ORM, or runtime
- No performance optimization beyond identifying and fixing async blocking issues
- No multi-region or high-availability deployment architecture
- No automated rollback or blue/green deployment pipelines

---

## Technical Considerations

- All stories in this PRD operate on the existing codebase; no new dependencies should be introduced without justification.
- Security scan tools: `pip-audit` (preferred) or `safety`. Either is acceptable.
- Structured logging should use Python's stdlib `logging` module with a JSON formatter — do not introduce a new logging library unless `structlog` is already a dependency.
- Coverage tooling: `pytest-cov`. Threshold should be agreed before US-102 (suggested minimum: 80% overall, 70% per module).
- Rate limiting: use an existing middleware approach compatible with the dashboard server framework already in use; do not add a full API gateway.
- Smoke tests should be standalone scripts or a dedicated pytest marker (`@pytest.mark.smoke`) that can be run in isolation against a live local instance.

---

# PHASE 37 — Security Audit

### US-093: Dependency vulnerability scan and remediation

**Description:** As a developer, I want all dependencies scanned for known CVEs so that Rex does not ship with vulnerable packages.

**Acceptance Criteria:**
- [x] `pip-audit` (or `safety check`) runs against the current lock file / installed packages
- [x] all critical and high severity findings remediated or explicitly documented as accepted risk with justification
- [x] scan added as a CI step that fails on new critical/high findings
- [x] Typecheck passes

---

### US-094: Input validation audit and remediation for HTTP endpoints

**Description:** As a developer, I want all HTTP endpoint inputs validated and sanitized so that malformed or malicious payloads cannot crash the server or cause unexpected behavior.

**Acceptance Criteria:**
- [x] all POST and PUT endpoints validated to reject missing or malformed required fields with a 400 response
- [x] string inputs checked for length limits where unbounded input could cause resource exhaustion
- [x] no endpoint passes raw user input directly to a shell command, file path, or SQL query without sanitization
- [x] at least one test per endpoint confirms a malformed payload returns 400, not 500
- [x] Typecheck passes

---

### US-095: Authentication and session security review

**Description:** As a developer, I want authentication and session management reviewed against baseline security requirements so that sessions cannot be hijacked or forged.

**Acceptance Criteria:**
- [x] session tokens are cryptographically random (min 128 bits of entropy)
- [x] session tokens invalidated on logout
- [x] authentication endpoints have a failed-attempt rate limit or lockout
- [x] tokens are not logged in plaintext anywhere in the logging output
- [x] Typecheck passes

---

### US-096: Hardcoded credential and secret scan

**Description:** As a developer, I want the full codebase scanned for hardcoded credentials so that no secrets are committed to the repository.

**Acceptance Criteria:**
- [x] `trufflehog`, `gitleaks`, or equivalent tool run against the full git history
- [x] zero confirmed hardcoded secrets (API keys, passwords, tokens) found in source files or commit history
- [x] any historical findings documented and rotated if real credentials
- [x] pre-commit hook or CI step added to block future secret commits
- [x] Typecheck passes

---

### US-097: HTTP security headers

**Description:** As a developer, I want security headers set on all HTTP responses so that browsers and clients are protected from common web vulnerabilities.

**Acceptance Criteria:**
- [x] `Content-Security-Policy` header present on all HTML responses
- [x] `X-Frame-Options: DENY` or `SAMEORIGIN` set
- [x] `X-Content-Type-Options: nosniff` set
- [x] CORS policy restricts allowed origins to configured whitelist (not wildcard `*` in production)
- [x] `Strict-Transport-Security` header set if HTTPS is used
- [x] Typecheck passes

---

# PHASE 38 — Test Coverage

### US-098: Measure and document baseline test coverage

**Description:** As a developer, I want a coverage report generated for every module so that gaps are visible and a target threshold can be set.

**Acceptance Criteria:**
- [x] `pytest --cov=rex --cov-report=term-missing` runs without error
- [x] coverage report saved to `coverage.txt` or equivalent
- [x] modules with below-50% coverage listed explicitly in the report
- [x] agreed minimum coverage threshold documented in `pyproject.toml` or `setup.cfg`
- [x] Typecheck passes

---

### US-099: Fill unit test gaps — planner, tool registry, workflow engine

**Description:** As a developer, I want every public method in the planner, tool registry, and workflow engine modules covered by at least one unit test so that regressions in core reasoning paths are caught.

**Acceptance Criteria:**
- [x] all public methods of `rex/planner/` have at least one passing unit test
- [x] all public methods of the tool registry and tool router have at least one passing unit test
- [x] all public methods of the workflow engine have at least one passing unit test
- [x] no new tests rely on external services or live credentials
- [x] Typecheck passes

---

### US-100: Fill unit test gaps — memory, notifications, event system

**Description:** As a developer, I want memory storage, notification routing, and the event bus covered by unit tests so that regressions in stateful components are caught.

**Acceptance Criteria:**
- [x] memory store and memory search have at least one passing unit test each
- [x] notification routing rules have at least one passing unit test covering each priority level
- [x] event bus publish and subscriber notification have at least one passing unit test
- [x] no new tests rely on a running database; use in-memory or mock storage
- [x] Typecheck passes

---

### US-101: Fill unit test gaps — LLM providers, integrations, voice pipeline

**Description:** As a developer, I want LLM provider adapters, external integrations, and the voice pipeline covered by unit tests using mocks so that integration failures are caught without live credentials.

**Acceptance Criteria:**
- [x] each LLM provider (OpenAI, Anthropic, local) has at least one unit test using a mock HTTP client
- [x] Home Assistant, Plex, and GitHub adapter methods have at least one unit test using stub/mock data
- [x] STT and TTS pipeline components have at least one unit test using mock audio data
- [x] no new tests make real network calls
- [x] Typecheck passes

---

### US-102: Enforce coverage threshold in CI

**Description:** As a developer, I want CI to fail if coverage drops below the agreed threshold so that new code cannot land without tests.

**Acceptance Criteria:**
- [x] `pytest --cov=rex --cov-fail-under=<threshold>` runs in CI
- [x] CI job fails when overall coverage is below the configured threshold
- [x] threshold value stored in `pyproject.toml` and documented
- [x] Typecheck passes

---

# PHASE 39 — Error Handling and Resilience

### US-103: Global unhandled exception handler

**Description:** As a developer, I want unhandled exceptions caught at the application boundary and logged with full context so that crashes produce actionable error records rather than silent failures.

**Acceptance Criteria:**
- [x] a top-level exception handler wraps the main application entry points
- [x] unhandled exceptions logged with: exception type, message, full traceback, and timestamp
- [x] application exits with a non-zero exit code on fatal error
- [x] handler does not swallow exceptions silently
- [x] Typecheck passes

---

### US-104: Consistent error response envelope

**Description:** As a developer, I want all API error responses to use a consistent JSON envelope so that clients can reliably parse errors without special-casing each endpoint.

**Acceptance Criteria:**
- [x] all error responses return JSON with at minimum: `error.code` (string), `error.message` (string)
- [x] HTTP status codes are semantically correct (400 for bad input, 401 for auth failure, 500 for server error)
- [x] no endpoint returns a plain-text error or an unstructured exception traceback to the client
- [x] at least one test per error condition verifies the response shape
- [x] Typecheck passes

---

### US-105: Retry with exponential backoff for external service calls

**Description:** As a developer, I want transient failures in external service calls retried with exponential backoff so that brief network interruptions do not surface as user-visible errors.

**Acceptance Criteria:**
- [x] LLM provider calls retried up to a configurable max attempt count (default: 3) on transient errors (timeout, 429, 503)
- [x] retry delay doubles between attempts with configurable base delay (default: 1s)
- [x] non-retryable errors (400, 401, 403) are not retried
- [x] retry behavior covered by at least one unit test using a mock that fails N times then succeeds
- [x] Typecheck passes

---

### US-106: Graceful shutdown

**Description:** As a developer, I want the application to handle SIGTERM cleanly so that in-flight requests complete and resources are released before the process exits.

**Acceptance Criteria:**
- [x] SIGTERM signal registered and handled in the main process
- [x] on SIGTERM, no new requests accepted and in-flight requests given up to a configurable drain timeout (default: 10s) to complete
- [x] open database connections and background jobs closed cleanly on shutdown
- [x] process exits with code 0 after clean shutdown
- [x] Typecheck passes

---

# PHASE 40 — Logging and Observability

### US-107: Structured JSON logging

**Description:** As a developer, I want all log output formatted as structured JSON so that logs are machine-parseable and easily ingested by log aggregation tools.

**Acceptance Criteria:**
- [x] all log output from `rex/` emitted as JSON (one object per line)
- [x] each log entry includes at minimum: `timestamp` (ISO 8601), `level`, `logger` (module name), `message`
- [x] no log lines use bare `print()` statements
- [x] existing test output remains readable (JSON logging can be disabled in test mode via config)
- [x] Typecheck passes

---

### US-108: Log level configuration per environment

**Description:** As a developer, I want log verbosity configurable per environment so that production runs at INFO and development can run at DEBUG without code changes.

**Acceptance Criteria:**
- [x] log level configurable via environment variable (e.g., `LOG_LEVEL=DEBUG`)
- [x] default log level is `INFO` when `LOG_LEVEL` is not set
- [x] per-module log level overrides supported via config
- [x] DEBUG-level logs do not appear in output when `LOG_LEVEL=INFO`
- [x] Typecheck passes

---

### US-109: Request and response logging middleware

**Description:** As a developer, I want every inbound HTTP request and outgoing response logged with method, path, status code, and duration so that API traffic is traceable without a separate APM tool.

**Acceptance Criteria:**
- [x] middleware logs each request: method, path, client IP (anonymized or configurable), timestamp
- [x] middleware logs each response: status code, duration in milliseconds
- [x] request and response log entries share a common request ID for correlation
- [x] request body and response body are NOT logged by default (to avoid PII leakage)
- [x] Typecheck passes

---

### US-110: Liveness and readiness health check endpoints

**Description:** As a developer, I want separate liveness and readiness endpoints so that process supervisors and load balancers can distinguish between a starting process and a failed one.

**Acceptance Criteria:**
- [x] `GET /health/live` returns 200 when the process is running, regardless of dependency state
- [x] `GET /health/ready` returns 200 only when all critical dependencies (database, config) are available
- [x] `GET /health/ready` returns 503 with a JSON body describing which dependencies are unavailable
- [x] both endpoints respond in under 500ms under normal conditions
- [x] Typecheck passes

---

# PHASE 41 — Configuration Hardening

### US-111: Startup config validation with fail-fast

**Description:** As a developer, I want the application to validate all required configuration and environment variables at startup and exit immediately with a clear error if any are missing or invalid so that misconfigured deployments fail loudly rather than silently misbehaving.

**Acceptance Criteria:**
- [x] a config validation step runs before any other initialization
- [x] missing required environment variables produce a specific error message naming the missing variable and exit code 1
- [x] invalid values (e.g., non-numeric port, malformed URL) produce a descriptive error and exit code 1
- [x] optional variables with defaults do not cause startup failure
- [x] Typecheck passes

---

### US-112: .env.example and environment variable reference

**Description:** As a developer, I want a `.env.example` file in the repo root documenting every environment variable so that new developers can configure the application without reading source code.

**Acceptance Criteria:**
- [x] `.env.example` exists at the repo root
- [x] every environment variable consumed by the application is present in `.env.example` with a comment describing its purpose and acceptable values
- [x] required variables are clearly marked as required; optional variables show their default
- [x] `.env` is in `.gitignore` and not committed
- [x] Typecheck passes

---

### US-113: Production configuration defaults

**Description:** As a developer, I want the application to enforce safe production defaults so that debug features, verbose tracing, and development shortcuts are disabled when running in production mode.

**Acceptance Criteria:**
- [x] `DEBUG` mode disabled when `ENVIRONMENT=production` (or equivalent)
- [x] stack traces not returned to API clients in production
- [x] development-only endpoints or routes disabled or unreachable in production mode
- [x] production mode detectable from a single `ENVIRONMENT` environment variable
- [x] Typecheck passes

---

# PHASE 42 — Database Production Readiness

### US-114: Database connection pool configuration

**Description:** As a developer, I want the database connection pool size and timeout configured explicitly so that Rex does not exhaust database connections under load or hang indefinitely on unavailable connections.

**Acceptance Criteria:**
- [x] connection pool min/max size configurable via environment variables
- [x] connection acquisition timeout configured (default: 5s); acquisition failure raises a handled error
- [x] idle connection timeout configured to prevent stale connections
- [x] pool settings logged at startup at INFO level
- [x] Typecheck passes

---

### US-115: Migration state validation on startup

**Description:** As a developer, I want the application to check that all database migrations have been applied before accepting traffic so that schema mismatches are caught immediately rather than at runtime.

**Acceptance Criteria:**
- [x] on startup, the migration state is queried and compared against the expected schema version
- [x] if unapplied migrations exist, the application logs the pending migration names and exits with code 1
- [x] migration check runs before any request handler is registered
- [x] check can be disabled via a `SKIP_MIGRATION_CHECK` environment variable for emergency use
- [x] Typecheck passes

---

### US-116: Query timeout enforcement

**Description:** As a developer, I want all database queries to have a timeout so that a slow query cannot block a request indefinitely.

**Acceptance Criteria:**
- [x] a default query timeout applied to all database operations (default: 10s, configurable)
- [x] queries that exceed the timeout raise a handled exception, not a hang
- [x] timeout errors logged with query context (excluding any PII in query parameters)
- [x] at least one test verifies timeout behavior using a mock that delays beyond the threshold
- [x] Typecheck passes

---

# PHASE 43 — API Polish

### US-117: Consistent error response envelope enforcement

**Description:** As a developer, I want a middleware or base handler to enforce the standard error envelope on every error response so that individual endpoint authors cannot accidentally return unstructured errors.

**Acceptance Criteria:**
- [x] error formatting logic lives in one place (middleware or exception handler), not duplicated per endpoint
- [x] a test hitting each endpoint with a deliberately bad request confirms the standard envelope is returned
- [x] 500-level errors include an `error.request_id` field for log correlation
- [x] Typecheck passes

---

### US-118: Request payload schema validation on all POST and PUT endpoints

**Description:** As a developer, I want request payloads validated against a schema at the framework level so that handler logic can assume valid input and validation errors are returned consistently.

**Acceptance Criteria:**
- [x] every POST and PUT endpoint declares a required schema (Pydantic model, dataclass, or equivalent)
- [x] requests with missing required fields return 400 with the specific field name(s) missing
- [x] requests with incorrect field types return 400 with a descriptive message
- [x] validation runs before any business logic executes
- [x] Typecheck passes

---

### US-119: Rate limiting on public-facing API endpoints

**Description:** As a developer, I want rate limits applied to public-facing endpoints so that a single client cannot exhaust server resources through excessive requests.

**Acceptance Criteria:**
- [x] rate limiter applied to all unauthenticated or public endpoints
- [x] rate limit configurable (default: 60 requests/minute per IP)
- [x] requests exceeding the limit receive a 429 response with a `Retry-After` header
- [x] rate limiter does not apply to health check endpoints
- [x] Typecheck passes

---

# PHASE 44 — Performance Baseline

### US-120: Response time baseline for core API endpoints

**Description:** As a developer, I want a documented response time baseline for core endpoints so that performance regressions are detectable in future test runs.

**Acceptance Criteria:**
- [x] response times measured for at minimum: health check, chat message send, notification list, config load
- [x] measurements taken with a local warm instance (min 10 requests, median reported)
- [x] baseline documented in `docs/performance-baseline.md`
- [x] any endpoint with p50 > 500ms flagged for investigation
- [x] Typecheck passes

---

### US-121: Audit and fix blocking I/O in async handlers

**Description:** As a developer, I want all async request handlers free of blocking synchronous I/O calls so that the event loop is never stalled by a slow operation.

**Acceptance Criteria:**
- [x] all async handler functions audited for synchronous file I/O, `time.sleep()`, and synchronous HTTP calls
- [x] any blocking calls found replaced with async equivalents or offloaded to a thread executor
- [x] findings and changes documented in a comment or commit message
- [x] Typecheck passes

---

### US-122: Memory usage baseline and leak detection

**Description:** As a developer, I want memory usage profiled under a representative workload so that obvious leaks are caught before production deployment.

**Acceptance Criteria:**
- [x] `tracemalloc` or `memray` used to profile memory during a simulated workload (min 100 requests)
- [x] baseline RSS memory usage documented in `docs/performance-baseline.md`
- [x] any object type accumulating unboundedly across requests flagged and investigated
- [x] no confirmed memory leaks (unbounded growth) present at release
- [x] Typecheck passes

---

# PHASE 45 — Documentation and Runbook

### US-123: Production deployment guide

**Description:** As a developer, I want a step-by-step production deployment guide so that a new operator can deploy Rex without tribal knowledge.

**Acceptance Criteria:**
- [x] `docs/deployment.md` exists and covers: prerequisites, environment setup, installation steps, first-run verification
- [x] guide documents how to apply database migrations before starting the service
- [x] guide documents how to verify the service is healthy after deployment
- [x] guide tested by following steps on a clean environment and confirming successful startup
- [x] Typecheck passes

---

### US-124: Environment variable and configuration reference

**Description:** As a developer, I want a single reference document listing every configuration option so that operators can tune Rex for their environment without reading source code.

**Acceptance Criteria:**
- [x] `docs/configuration.md` exists and lists every environment variable with: name, description, default, required/optional
- [x] document organized into logical sections (server, database, LLM providers, integrations, logging)
- [x] document consistent with `.env.example` (no variables in one but not the other)
- [x] Typecheck passes

---

### US-125: Operations runbook

**Description:** As an operator, I want a runbook covering common operational tasks so that I can start, stop, restart, and diagnose Rex without escalating to a developer.

**Acceptance Criteria:**
- [x] `docs/runbook.md` exists and covers: start/stop/restart procedure, log access and filtering, health check verification, what to do if a service fails to start
- [x] runbook documents the expected process list and how to verify each component is running
- [x] at least five common error scenarios documented with diagnosis steps and resolution
- [x] Typecheck passes

---

### US-126: API reference documentation

**Description:** As a developer or integrator, I want all public API endpoints documented so that I can build integrations without reading source code.

**Acceptance Criteria:**
- [x] `docs/api.md` or equivalent documents every public endpoint: method, path, request schema, response schema, error codes
- [x] authentication requirements documented per endpoint
- [x] at least one example request and response shown per endpoint
- [x] document consistent with the actual running API (no phantom or missing endpoints)
- [x] Typecheck passes

---

# PHASE 46 — Deployment Readiness

### US-127: Service startup sequence and dependency ordering

**Description:** As a developer, I want the startup sequence to enforce dependency ordering so that services that depend on the database or config do not start before those dependencies are ready.

**Acceptance Criteria:**
- [x] startup sequence documented and enforced in code: config validation → database connection → migration check → service initialization → begin accepting traffic
- [x] if any step fails, subsequent steps do not run
- [x] startup sequence logged at INFO level so the log stream shows exactly where a failure occurred
- [x] Typecheck passes

---

### US-128: Process supervisor configuration

**Description:** As an operator, I want Rex services managed by a process supervisor so that crashed services restart automatically and startup on system boot is handled without manual intervention.

**Acceptance Criteria:**
- [x] a `systemd` unit file (or `supervisor` config, whichever matches the target deployment environment) provided for all long-running Rex processes
- [x] unit file configures automatic restart on failure (with a backoff limit to prevent restart loops)
- [x] unit file documented in `docs/deployment.md`
- [x] starting the unit file on a clean system results in the service coming up and passing the liveness check
- [x] Typecheck passes

---

### US-129: Production smoke test suite

**Description:** As a developer, I want a smoke test suite that verifies all critical paths against a running instance so that a deployment can be validated in minutes without a full regression run.

**Acceptance Criteria:**
- [x] smoke tests marked with `@pytest.mark.smoke` and runnable via `pytest -m smoke`
- [x] smoke tests cover at minimum: health check, authentication, chat message round-trip, notification creation, CLI entrypoints
- [x] smoke tests connect to a running local instance (not mocks)
- [x] all smoke tests pass against a freshly started local instance
- [x] smoke test run time under 2 minutes
- [x] Typecheck passes

---

# PHASE 47 — Final Production Sign-off

### US-130: Full test suite clean run

**Description:** As a developer, I want the complete test suite to pass with zero failures and zero errors so that there are no known regressions at release.

**Acceptance Criteria:**
- [x] `pytest` exits with code 0
- [x] zero test failures, zero test errors
- [x] zero tests marked `xfail` that are unexpectedly passing (review any xfail markers)
- [x] test run completes in under 10 minutes on the reference machine
- [x] Typecheck passes

---

### US-131: Final security scan

**Description:** As a developer, I want a final end-to-end security scan run immediately before release so that no vulnerabilities introduced during the production readiness phase have been missed.

**Acceptance Criteria:**
- [x] `pip-audit` (or equivalent) returns zero critical or high CVEs
- [x] secret scan (`gitleaks` or equivalent) returns zero confirmed findings against the full git history including all new commits
- [x] security headers verified present on a live local instance using `curl -I`
- [x] findings (if any) documented with remediation status
- [x] Typecheck passes

---

### US-132: Production readiness checklist sign-off

**Description:** As a developer, I want a completed production readiness checklist committed to the repository so that there is a permanent record that every gate was passed before the first production deployment.

**Acceptance Criteria:**
- [x] `docs/production-readiness-checklist.md` exists and contains a checklist item for every phase in this PRD
- [x] every checklist item marked complete with the US number that completed it
- [x] any items explicitly waived documented with justification
- [x] document committed to the repository and reviewed by at least one other person before deployment
- [x] Typecheck passes

---

## Introduction

This PRD addresses six confirmed issues with Rex: a broken voice TTS playback pipeline, excessive installation complexity, a poorly organized README, a developer-oriented user experience, the absence of a polished user-facing GUI, and slow perceived response time. Each issue has corresponding requested changes. Work in this PRD is independent of the core feature buildout PRD and can begin once the repository is stable (Phase 1–4 of PRD.md complete).

---

## Goals

- Fix the voice mode response pipeline so Rex speaks replies back to the user
- Reduce installation to a single clear method a non-technical user can follow
- Rewrite the README so a first-time user can be up and running in under 10 minutes
- Build a polished, modern desktop GUI that serves as the primary way users interact with Rex
- Make the GUI the home for chat, voice, scheduling, and visibility into what Rex is managing
- Reduce actual end-to-end voice response latency through pipeline optimization
- Improve perceived responsiveness so interactions feel fast even when hardware limits speed

---

## Non-Goals

- No mobile app
- No cloud sync or multi-device support
- No user accounts or authentication beyond the existing dashboard login
- No plugin marketplace or plugin distribution UI
- No automated deployment or packaging (installers, signed binaries, app stores)
- No changes to LLM provider integrations (covered in PRD.md)

---

## Technical Considerations

- **Voice pipeline:** The break point is somewhere between LLM response delivery and TTS audio playback. Instrumentation stories (US-133, US-134) must run before the fix stories (US-135–US-137) to identify the exact failure point.
- **GUI framework:** Rex already has a FastAPI backend and a web-based dashboard. The recommended path is to build the new GUI as a modern single-page application (React + Tailwind or similar) served by the existing backend, optionally wrapped in Electron or Tauri for a native desktop window. Stories are written framework-agnostically; the implementer should choose based on the existing stack.
- **Install script:** Target Windows (`install.ps1`) as the primary platform. A `install.sh` variant for Linux/macOS is a secondary deliverable within the same story.
- **Performance:** Streaming TTS (playing the first audio chunk before the full response is generated) is the highest-leverage latency improvement. Stories are ordered so profiling (US-167) precedes optimization.
- **Story numbering:** Continues from US-132 (end of PRD-production-readiness.md). Phases continue from Phase 47.

---

# PHASE 48 — Voice TTS Playback Diagnosis

> Fix the voice pipeline break between LLM response generation and audio playback. Diagnosis stories come first to locate the exact failure point before any fixes are attempted.

### US-133: Add tracing logs to the full voice response pipeline

**Description:** As a developer, I want structured log statements at every stage of the voice response pipeline so that I can identify exactly where audio delivery fails.

**Acceptance Criteria:**
- [x] log entry emitted at each stage: LLM response received, TTS input text prepared, TTS engine called, audio data returned, audio playback initiated, audio playback completed
- [x] each log entry includes stage name, timestamp, and any relevant payload size or status
- [x] logs visible at DEBUG level without modifying source code (controlled by `LOG_LEVEL`)
- [x] running a voice interaction with `LOG_LEVEL=DEBUG` produces a trace covering all stages or clearly shows which stage is missing
- [x] Typecheck passes

---

### US-134: Document and confirm the confirmed break point in voice playback

**Description:** As a developer, I want the exact failure point in the voice pipeline documented in `AGENTS.md` so that subsequent fix stories have a precise target.

**Acceptance Criteria:**
- [x] a test script or manual procedure exists that triggers the full voice pipeline and captures log output
- [x] the log output identifies which stage completes last before audio stops
- [x] finding documented in `AGENTS.md` under a "Voice Pipeline Break Point" heading
- [x] finding includes: last successful stage, first missing stage, relevant code path (file and function name)
- [x] Typecheck passes

---

# PHASE 49 — Voice TTS Playback Fix

### US-135: Fix TTS text input delivery from LLM response handler

**Description:** As a developer, I want the LLM response text reliably passed to the TTS engine so that every generated reply is queued for speech synthesis.

**Acceptance Criteria:**
- [x] LLM response text reaches the TTS input function on every successful generation
- [x] empty or whitespace-only responses do not trigger TTS
- [x] errors in the LLM handler do not silently discard the response before TTS is called
- [x] unit test confirms TTS input function is called with the correct text after a mock LLM response
- [x] Typecheck passes

---

### US-136: Fix audio playback and output device selection

**Description:** As a developer, I want synthesized audio played through the correct output device so that Rex's spoken response is audible to the user.

**Acceptance Criteria:**
- [x] TTS audio output plays through the system default audio device
- [x] output device configurable via `config` (device name or index)
- [x] playback does not block the voice loop from processing new input after audio ends
- [x] audio playback errors are caught and logged, not silently swallowed
- [x] Typecheck passes

---

### US-137: Fix voice loop re-arm after TTS playback completes

**Description:** As a developer, I want the voice loop to return to the wake-word listening state immediately after TTS playback finishes so that Rex is ready for the next interaction.

**Acceptance Criteria:**
- [x] after TTS audio finishes playing, the wake word detector resumes within 1 second
- [x] the microphone stream is not left open or blocked after playback
- [x] a second voice interaction triggered after the first completes successfully produces a spoken response
- [x] Typecheck passes

---

### US-138: End-to-end voice round-trip integration test

**Description:** As a developer, I want an automated test that exercises the full wake-word-to-spoken-response pipeline using mocks so that regressions in voice mode are caught by CI.

**Acceptance Criteria:**
- [x] test injects a mock wake word event, a mock STT transcript, a mock LLM response, and asserts TTS was called with the expected text
- [x] test asserts the voice loop re-arms after the mock playback completes
- [x] test passes without any real microphone, speaker, or network connection
- [x] test added to CI and passes on first run
- [x] Typecheck passes

---

# PHASE 50 — Unified Installation

### US-139: Create a single-command install script

**Description:** As a user, I want to run one command that installs everything Rex needs so that I do not have to understand Python packaging, virtual environments, or optional extras.

**Acceptance Criteria:**
- [x] `install.ps1` (Windows) and `install.sh` (Linux/macOS) exist at the repo root
- [x] the script creates a virtual environment, installs Rex with all required dependencies, and verifies the install
- [x] on success, the script prints a clear "Rex is installed. Run `rex` to start." message
- [x] on failure, the script prints a specific error and exits with a non-zero code
- [x] Typecheck passes

---

### US-140: Consolidate optional extras into a single `[full]` install target

**Description:** As a developer, I want all user-facing capabilities bundled into a single `pip install rex[full]` so that users do not need to choose between extras.

**Acceptance Criteria:**
- [x] `pyproject.toml` defines a `[full]` extra that includes all extras required for the complete Rex experience (voice, integrations, GUI)
- [x] `pip install rex[full]` succeeds and installs all required packages
- [x] existing extras remain available for advanced users who want minimal installs
- [x] install script updated to use `rex[full]`
- [x] Typecheck passes

---

### US-141: Remove or archive legacy install instructions from the main flow

**Description:** As a developer, I want legacy and advanced install options moved out of the primary user-facing documentation so that new users see only one install path.

**Acceptance Criteria:**
- [x] the main README references only the single install script as the primary install method
- [x] any legacy install steps (manual pip commands, multiple extras choices, etc.) moved to `docs/advanced-install.md`
- [x] `docs/advanced-install.md` linked from the README under a clearly labeled "Advanced / Developer Install" section
- [x] Typecheck passes

---

### US-142: Improve `rex doctor` to validate the full install

**Description:** As a user, I want `rex doctor` to confirm that all components needed for the full Rex experience are working so that I know my install is complete before I try to use it.

**Acceptance Criteria:**
- [x] `rex doctor` checks and reports status for: Python version, all required packages, audio input device, audio output device, LM Studio reachability (with timeout), config file presence
- [x] each check reports PASS or FAIL with a specific, actionable message on failure
- [x] overall result clearly indicates whether Rex is ready to use
- [x] Typecheck passes

---

# PHASE 51 — README Overhaul

### US-143: Restructure README with quick start first and a table of contents

**Description:** As a new user, I want the README to open with a quick start section and a navigable table of contents so that I can get Rex running without reading the entire document.

**Acceptance Criteria:**
- [x] README opens with a one-paragraph description of what Rex is and who it is for
- [x] table of contents appears within the first 30 lines of the README
- [x] "Quick Start" section is the first major section after the description and table of contents
- [x] Quick Start contains no more than 5 steps
- [x] Typecheck passes

---

### US-144: Write a clear Quick Start section (5 steps or fewer)

**Description:** As a new user, I want a Quick Start guide that gets Rex running in under 10 minutes so that I do not need to read the full documentation to try it.

**Acceptance Criteria:**
- [x] Quick Start section contains exactly the steps: clone repo, run install script, configure LM Studio, run Rex, verify it works
- [x] each step is a single clear action with the exact command to run
- [x] Quick Start tested on a clean machine and confirmed to produce a working Rex install
- [x] no step requires the user to read another section first
- [x] Typecheck passes

---

### US-145: Move deep technical content into secondary docs

**Description:** As a developer, I want advanced configuration, architecture details, and developer setup moved to `docs/` so that the README remains concise without losing any information.

**Acceptance Criteria:**
- [x] any README section longer than 20 lines that is not relevant to first-time setup moved to a dedicated file in `docs/`
- [x] each moved section replaced with a one-line summary and a link in the README
- [x] all moved content preserved verbatim (no information lost)
- [x] Typecheck passes

---

### US-146: Add visual structure to the README (badges, section dividers, call-outs)

**Description:** As a new user, I want the README to use visual structure so that it is easy to scan and navigate.

**Acceptance Criteria:**
- [x] repo status badges added (CI status, Python version, license)
- [x] each major section clearly headed with a level-2 heading
- [x] important warnings or prerequisites use a blockquote or note callout, not inline text
- [x] README renders correctly on GitHub (no broken markdown)
- [x] Typecheck passes

---

# PHASE 52 — Onboarding Improvements

### US-147: First-run setup detection and guided message

**Description:** As a new user, I want Rex to detect when it is being run for the first time and print a short guided setup message so that I know what to do next.

**Acceptance Criteria:**
- [x] Rex detects first run (no config file or empty config file)
- [x] on first run, prints a clear "Welcome to Rex. Let's get you set up." message with the 3 most important next steps
- [x] first-run message does not appear on subsequent runs once config is present
- [x] Typecheck passes

---

### US-148: Friendly, actionable error messages for missing dependencies

**Description:** As a user, I want missing dependency errors to tell me exactly how to fix them so that I am not left with a raw Python traceback.

**Acceptance Criteria:**
- [x] `ImportError` for any optional Rex dependency caught at the module level and re-raised with a human-readable message including the install command to fix it
- [x] missing LM Studio connection produces a message like "Rex can't reach LM Studio at [url]. Is LM Studio running?" rather than a connection refused traceback
- [x] no raw Python tracebacks are shown to the user in normal operation (tracebacks reserved for DEBUG mode)
- [x] Typecheck passes

---

# PHASE 53 — GUI Framework and Shell

### US-149: Scaffold GUI application shell

**Description:** As a developer, I want the GUI application scaffolded with a main window, navigation sidebar, and content area so that feature panels can be added incrementally.

**Acceptance Criteria:**
- [x] GUI application launches with a single command (e.g., `rex --gui` or `rex-gui`)
- [x] main window renders with a left sidebar for navigation and a main content area
- [x] sidebar contains placeholder navigation items for: Chat, Voice, Schedule, Overview
- [x] window is resizable and has a minimum usable size (e.g., 800x600)
- [x] application closes cleanly without errors
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-150: Apply base visual design system (colors, typography, spacing)

**Description:** As a user, I want the GUI to use a consistent, modern visual design so that it feels like a polished product rather than a developer utility.

**Acceptance Criteria:**
- [x] a design token file (CSS variables, theme object, or equivalent) defines: primary color, background color, surface color, text color, accent color, font family, base spacing unit
- [x] all GUI components use values from the design token file, not hardcoded colors or sizes
- [x] overall appearance is dark or neutral-dark themed (not a default browser/OS chrome look)
- [x] typography uses a clean sans-serif font (system font stack or a single loaded font)
- [x] Typecheck passes
- [x] Verify changes work in browser

---

### US-151: Implement active navigation state and panel switching

**Description:** As a user, I want clicking a sidebar item to load the correct panel so that I can navigate between sections of the GUI.

**Acceptance Criteria:**
- [x] clicking each sidebar item displays the corresponding panel in the content area
- [x] the active sidebar item is visually highlighted
- [x] navigation does not reload the page or lose state in already-loaded panels
- [x] back/forward browser navigation works correctly if the GUI is web-based
- [x] Typecheck passes
- [x] Verify changes work in browser

---

# PHASE 54 — GUI Chat Panel

### US-152: Chat message list component

**Description:** As a user, I want to see a scrollable list of conversation messages in the Chat panel so that I can read my full conversation history with Rex.

**Acceptance Criteria:**
- [ ] Chat panel displays messages in chronological order, oldest at top, newest at bottom
- [ ] user messages and Rex messages are visually distinct (different alignment, color, or label)
- [ ] message list auto-scrolls to the newest message when a new message arrives
- [ ] message list is scrollable and handles at least 100 messages without layout issues
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-153: Chat message input and send

**Description:** As a user, I want a text input at the bottom of the Chat panel so that I can type a message and send it to Rex.

**Acceptance Criteria:**
- [ ] text input field visible and focused by default when the Chat panel is open
- [ ] pressing Enter or clicking a Send button submits the message
- [ ] input clears after send
- [ ] sending an empty message is a no-op (no empty messages added to the list)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-154: Connect chat UI to Rex backend

**Description:** As a user, I want messages I send in the chat UI to reach Rex and display Rex's response in the conversation so that the GUI is a functional chat interface.

**Acceptance Criteria:**
- [ ] submitted message sent to the Rex backend chat endpoint
- [ ] Rex's response displayed in the message list when received
- [ ] a loading indicator appears between message send and response arrival
- [ ] network errors produce a visible error message in the chat, not a silent failure
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-155: Chat message streaming display

**Description:** As a user, I want Rex's response to appear word-by-word as it is generated so that I see immediate feedback instead of waiting for the full response.

**Acceptance Criteria:**
- [ ] backend streams the response token-by-token or chunk-by-chunk via SSE or WebSocket
- [ ] chat UI appends tokens to the current message bubble as they arrive
- [ ] the loading indicator is replaced by the streaming message (not shown simultaneously)
- [ ] streaming works correctly for responses of at least 500 tokens
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

# PHASE 55 — GUI Voice Panel

### US-156: Voice mode toggle with status indicator

**Description:** As a user, I want a button in the Voice panel that activates voice mode and shows the current state so that I can start and stop listening without using the CLI.

**Acceptance Criteria:**
- [ ] Voice panel has a prominent button labeled "Start Listening" / "Stop Listening"
- [ ] button toggles voice mode on and off via the Rex backend
- [ ] current voice state (Idle, Listening, Processing, Speaking) displayed as a text label or icon near the button
- [ ] state updates in real time without requiring a page refresh
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-157: Visual waveform or animation during active listening

**Description:** As a user, I want a visual animation when Rex is listening so that I have clear feedback that my voice is being captured.

**Acceptance Criteria:**
- [ ] when voice mode is in the Listening state, an animated waveform, pulsing ring, or equivalent visual is displayed
- [ ] animation stops when Rex transitions to Processing or Speaking state
- [ ] animation is purely CSS/SVG — no external animation library required
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-158: Voice transcript display in the Voice panel

**Description:** As a user, I want to see what Rex heard me say and what it is replying so that I can confirm it understood me correctly.

**Acceptance Criteria:**
- [ ] the most recent spoken input transcript displayed in the Voice panel after recognition completes
- [ ] Rex's most recent spoken response text displayed below the transcript
- [ ] both fields clear when a new interaction begins
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

# PHASE 56 — GUI Schedule Panel

### US-159: Scheduled items list view

**Description:** As a user, I want to see a list of all scheduled tasks and automations in the Schedule panel so that I know what Rex has planned.

**Acceptance Criteria:**
- [ ] Schedule panel fetches and displays all scheduled items from the Rex backend
- [ ] each item shows: name, schedule (human-readable), enabled/disabled status, next run time
- [ ] list refreshes automatically or on panel focus
- [ ] empty state message shown when no items are scheduled
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-160: Enable/disable scheduled items from the GUI

**Description:** As a user, I want to toggle a scheduled item on or off from the GUI so that I can pause automations without deleting them.

**Acceptance Criteria:**
- [ ] each scheduled item has a visible toggle (switch or checkbox) for enabled/disabled
- [ ] toggling calls the Rex backend and persists the change
- [ ] UI reflects the new state immediately after toggle
- [ ] toggle errors show a visible error message and revert the UI to the previous state
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-161: Upcoming events and due items panel section

**Description:** As a user, I want to see items due soon or upcoming events in the Schedule panel so that I have a glanceable view of what is coming up.

**Acceptance Criteria:**
- [ ] a "Coming Up" section in the Schedule panel shows items due in the next 24 hours
- [ ] items sorted by next run time, soonest first
- [ ] each item shows time until next run (e.g., "in 2 hours")
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

# PHASE 57 — GUI Overview Panel

### US-162: Overview panel with Rex status summary

**Description:** As a user, I want an Overview panel that shows Rex's current status at a glance so that I can quickly see whether everything is working.

**Acceptance Criteria:**
- [ ] Overview panel displayed by default when the GUI opens
- [ ] shows: Rex running status (online/offline), active voice mode (on/off), LM Studio connection status, count of scheduled items, count of recent notifications
- [ ] each status item has a green/red or similar visual indicator
- [ ] status data fetched from the Rex health endpoints
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-163: Quick action buttons on the Overview panel

**Description:** As a user, I want quick action buttons on the Overview panel for the most common Rex interactions so that I can trigger them without navigating to another panel.

**Acceptance Criteria:**
- [ ] at minimum three quick action buttons present: "Start Listening", "Open Chat", "View Schedule"
- [ ] each button navigates to the relevant panel or triggers the relevant action
- [ ] buttons are prominently placed and clearly labeled
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

# PHASE 58 — GUI Polish and Accessibility

### US-164: Consistent component styling and hover/focus states

**Description:** As a user, I want all interactive elements to have consistent hover and focus styles so that the GUI feels cohesive and is usable by keyboard.

**Acceptance Criteria:**
- [ ] all buttons have a visible hover state and a visible focus ring
- [ ] all inputs have a visible focus state
- [ ] hover and focus states use the design token colors, not browser defaults
- [ ] tabbing through the GUI reaches all interactive elements in logical order
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-165: Loading and error states for all data-fetching panels

**Description:** As a user, I want panels to show loading spinners while fetching data and clear error messages on failure so that I always know what the GUI is doing.

**Acceptance Criteria:**
- [ ] every panel that fetches data from the backend shows a loading indicator while the request is in flight
- [ ] every panel shows a specific error message (not a generic "something went wrong") if the request fails
- [ ] error state includes a "Retry" button that re-fetches the data
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-166: Responsive layout for common window sizes

**Description:** As a user, I want the GUI to remain usable when the window is resized so that it works on different screen sizes and configurations.

**Acceptance Criteria:**
- [ ] layout remains functional and readable at widths from 800px to 1920px
- [ ] sidebar collapses to icons or a hamburger menu below 1024px width
- [ ] no horizontal scrollbars appear at any standard window width
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

# PHASE 59 — Performance: Actual Latency Reduction

### US-167: Profile and document end-to-end voice response latency

**Description:** As a developer, I want the total time for each stage of the voice pipeline measured and documented so that optimization work targets the highest-latency stages.

**Acceptance Criteria:**
- [ ] timing instrumentation added to: STT processing, LLM first token, LLM full response, TTS synthesis start, TTS first audio chunk, audio playback start
- [ ] 10 sample interactions measured and results recorded in `docs/performance-baseline.md`
- [ ] stage responsible for the majority of total latency identified explicitly
- [ ] Typecheck passes

---

### US-168: Implement streaming TTS — play first audio chunk before full response is ready

**Description:** As a user, I want Rex to begin speaking before it has finished generating the full response so that I hear audio sooner and the interaction feels faster.

**Acceptance Criteria:**
- [ ] TTS engine receives response text in chunks as the LLM streams output
- [ ] first audio chunk begins playing within 2 seconds of the LLM producing its first sentence
- [ ] subsequent audio chunks play without gaps or interruption
- [ ] full response audio completes without truncation
- [ ] Typecheck passes

---

### US-169: Pre-warm TTS engine on startup

**Description:** As a developer, I want the TTS engine initialized and warmed up at application startup so that the first voice interaction does not pay a cold-start penalty.

**Acceptance Criteria:**
- [ ] TTS engine loads and synthesizes a silent or very short warmup phrase during Rex startup
- [ ] warmup runs in the background and does not delay the application becoming ready
- [ ] first user-triggered TTS call after warmup completes in under 1 second on reference hardware
- [ ] Typecheck passes

---

### US-170: Audit and reduce blocking operations in the voice pipeline

**Description:** As a developer, I want all synchronous blocking calls in the voice pipeline replaced with async equivalents so that the event loop is never stalled during voice interactions.

**Acceptance Criteria:**
- [ ] voice pipeline code audited for synchronous I/O, `time.sleep()`, and blocking network calls
- [ ] any blocking calls replaced with async equivalents or offloaded to a thread executor
- [ ] changes documented in a commit message or `AGENTS.md` entry
- [ ] no `time.sleep()` calls remain in the hot path of the voice interaction loop
- [ ] Typecheck passes

---

# PHASE 60 — Performance: Perceived Responsiveness

### US-171: Immediate audio acknowledgment on wake word (verify and enforce)

**Description:** As a user, I want to hear an acknowledgment tone within 200ms of saying the wake word so that I know Rex heard me before processing begins.

**Acceptance Criteria:**
- [ ] acknowledgment tone plays within 200ms of wake word detection on reference hardware
- [ ] acknowledgment tone does not block STT from beginning simultaneously
- [ ] tone playback failure does not prevent the voice pipeline from continuing
- [ ] timing verified and documented in `docs/performance-baseline.md`
- [ ] Typecheck passes

---

### US-172: Typing / thinking indicator in chat during LLM generation

**Description:** As a user, I want to see an animated thinking indicator while Rex is generating a response so that I know it is working and not frozen.

**Acceptance Criteria:**
- [ ] animated dots or equivalent indicator appears in the chat message list while the LLM is generating
- [ ] indicator appears within 100ms of message send
- [ ] indicator disappears and is replaced by the response text when generation begins
- [ ] indicator is removed if the request fails (error state shown instead)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-173: Voice pipeline state announcements via GUI status

**Description:** As a user, I want the GUI voice panel to update its state label in real time during a voice interaction so that I can see exactly what Rex is doing (Listening, Thinking, Speaking).

**Acceptance Criteria:**
- [ ] GUI Voice panel state label transitions through: Idle → Listening → Thinking → Speaking → Idle during a full voice interaction
- [ ] each transition occurs within 500ms of the underlying pipeline stage changing
- [ ] state updates delivered via SSE or WebSocket (not polling)
- [ ] Typecheck passes
- [ ] Verify changes work in browser

---

### US-174: Configurable LLM response length limit for voice mode

**Description:** As a user, I want Rex to give shorter replies in voice mode by default so that spoken responses feel natural and do not make me wait for a very long answer to finish playing.

**Acceptance Criteria:**
- [ ] a `voice_max_tokens` config value controls the maximum LLM output length in voice mode (default: 150 tokens)
- [ ] voice mode prompt includes an instruction to keep responses concise
- [ ] chat mode is not affected by the voice token limit
- [ ] config value documented in `.env.example`
- [ ] Typecheck passes

