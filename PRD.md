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
- [ ] middleware logs each request: method, path, client IP (anonymized or configurable), timestamp
- [ ] middleware logs each response: status code, duration in milliseconds
- [ ] request and response log entries share a common request ID for correlation
- [ ] request body and response body are NOT logged by default (to avoid PII leakage)
- [ ] Typecheck passes

---

### US-110: Liveness and readiness health check endpoints

**Description:** As a developer, I want separate liveness and readiness endpoints so that process supervisors and load balancers can distinguish between a starting process and a failed one.

**Acceptance Criteria:**
- [ ] `GET /health/live` returns 200 when the process is running, regardless of dependency state
- [ ] `GET /health/ready` returns 200 only when all critical dependencies (database, config) are available
- [ ] `GET /health/ready` returns 503 with a JSON body describing which dependencies are unavailable
- [ ] both endpoints respond in under 500ms under normal conditions
- [ ] Typecheck passes

---

# PHASE 41 — Configuration Hardening

### US-111: Startup config validation with fail-fast

**Description:** As a developer, I want the application to validate all required configuration and environment variables at startup and exit immediately with a clear error if any are missing or invalid so that misconfigured deployments fail loudly rather than silently misbehaving.

**Acceptance Criteria:**
- [ ] a config validation step runs before any other initialization
- [ ] missing required environment variables produce a specific error message naming the missing variable and exit code 1
- [ ] invalid values (e.g., non-numeric port, malformed URL) produce a descriptive error and exit code 1
- [ ] optional variables with defaults do not cause startup failure
- [ ] Typecheck passes

---

### US-112: .env.example and environment variable reference

**Description:** As a developer, I want a `.env.example` file in the repo root documenting every environment variable so that new developers can configure the application without reading source code.

**Acceptance Criteria:**
- [ ] `.env.example` exists at the repo root
- [ ] every environment variable consumed by the application is present in `.env.example` with a comment describing its purpose and acceptable values
- [ ] required variables are clearly marked as required; optional variables show their default
- [ ] `.env` is in `.gitignore` and not committed
- [ ] Typecheck passes

---

### US-113: Production configuration defaults

**Description:** As a developer, I want the application to enforce safe production defaults so that debug features, verbose tracing, and development shortcuts are disabled when running in production mode.

**Acceptance Criteria:**
- [ ] `DEBUG` mode disabled when `ENVIRONMENT=production` (or equivalent)
- [ ] stack traces not returned to API clients in production
- [ ] development-only endpoints or routes disabled or unreachable in production mode
- [ ] production mode detectable from a single `ENVIRONMENT` environment variable
- [ ] Typecheck passes

---

# PHASE 42 — Database Production Readiness

### US-114: Database connection pool configuration

**Description:** As a developer, I want the database connection pool size and timeout configured explicitly so that Rex does not exhaust database connections under load or hang indefinitely on unavailable connections.

**Acceptance Criteria:**
- [ ] connection pool min/max size configurable via environment variables
- [ ] connection acquisition timeout configured (default: 5s); acquisition failure raises a handled error
- [ ] idle connection timeout configured to prevent stale connections
- [ ] pool settings logged at startup at INFO level
- [ ] Typecheck passes

---

### US-115: Migration state validation on startup

**Description:** As a developer, I want the application to check that all database migrations have been applied before accepting traffic so that schema mismatches are caught immediately rather than at runtime.

**Acceptance Criteria:**
- [ ] on startup, the migration state is queried and compared against the expected schema version
- [ ] if unapplied migrations exist, the application logs the pending migration names and exits with code 1
- [ ] migration check runs before any request handler is registered
- [ ] check can be disabled via a `SKIP_MIGRATION_CHECK` environment variable for emergency use
- [ ] Typecheck passes

---

### US-116: Query timeout enforcement

**Description:** As a developer, I want all database queries to have a timeout so that a slow query cannot block a request indefinitely.

**Acceptance Criteria:**
- [ ] a default query timeout applied to all database operations (default: 10s, configurable)
- [ ] queries that exceed the timeout raise a handled exception, not a hang
- [ ] timeout errors logged with query context (excluding any PII in query parameters)
- [ ] at least one test verifies timeout behavior using a mock that delays beyond the threshold
- [ ] Typecheck passes

---

# PHASE 43 — API Polish

### US-117: Consistent error response envelope enforcement

**Description:** As a developer, I want a middleware or base handler to enforce the standard error envelope on every error response so that individual endpoint authors cannot accidentally return unstructured errors.

**Acceptance Criteria:**
- [ ] error formatting logic lives in one place (middleware or exception handler), not duplicated per endpoint
- [ ] a test hitting each endpoint with a deliberately bad request confirms the standard envelope is returned
- [ ] 500-level errors include an `error.request_id` field for log correlation
- [ ] Typecheck passes

---

### US-118: Request payload schema validation on all POST and PUT endpoints

**Description:** As a developer, I want request payloads validated against a schema at the framework level so that handler logic can assume valid input and validation errors are returned consistently.

**Acceptance Criteria:**
- [ ] every POST and PUT endpoint declares a required schema (Pydantic model, dataclass, or equivalent)
- [ ] requests with missing required fields return 400 with the specific field name(s) missing
- [ ] requests with incorrect field types return 400 with a descriptive message
- [ ] validation runs before any business logic executes
- [ ] Typecheck passes

---

### US-119: Rate limiting on public-facing API endpoints

**Description:** As a developer, I want rate limits applied to public-facing endpoints so that a single client cannot exhaust server resources through excessive requests.

**Acceptance Criteria:**
- [ ] rate limiter applied to all unauthenticated or public endpoints
- [ ] rate limit configurable (default: 60 requests/minute per IP)
- [ ] requests exceeding the limit receive a 429 response with a `Retry-After` header
- [ ] rate limiter does not apply to health check endpoints
- [ ] Typecheck passes

---

# PHASE 44 — Performance Baseline

### US-120: Response time baseline for core API endpoints

**Description:** As a developer, I want a documented response time baseline for core endpoints so that performance regressions are detectable in future test runs.

**Acceptance Criteria:**
- [ ] response times measured for at minimum: health check, chat message send, notification list, config load
- [ ] measurements taken with a local warm instance (min 10 requests, median reported)
- [ ] baseline documented in `docs/performance-baseline.md`
- [ ] any endpoint with p50 > 500ms flagged for investigation
- [ ] Typecheck passes

---

### US-121: Audit and fix blocking I/O in async handlers

**Description:** As a developer, I want all async request handlers free of blocking synchronous I/O calls so that the event loop is never stalled by a slow operation.

**Acceptance Criteria:**
- [ ] all async handler functions audited for synchronous file I/O, `time.sleep()`, and synchronous HTTP calls
- [ ] any blocking calls found replaced with async equivalents or offloaded to a thread executor
- [ ] findings and changes documented in a comment or commit message
- [ ] Typecheck passes

---

### US-122: Memory usage baseline and leak detection

**Description:** As a developer, I want memory usage profiled under a representative workload so that obvious leaks are caught before production deployment.

**Acceptance Criteria:**
- [ ] `tracemalloc` or `memray` used to profile memory during a simulated workload (min 100 requests)
- [ ] baseline RSS memory usage documented in `docs/performance-baseline.md`
- [ ] any object type accumulating unboundedly across requests flagged and investigated
- [ ] no confirmed memory leaks (unbounded growth) present at release
- [ ] Typecheck passes

---

# PHASE 45 — Documentation and Runbook

### US-123: Production deployment guide

**Description:** As a developer, I want a step-by-step production deployment guide so that a new operator can deploy Rex without tribal knowledge.

**Acceptance Criteria:**
- [ ] `docs/deployment.md` exists and covers: prerequisites, environment setup, installation steps, first-run verification
- [ ] guide documents how to apply database migrations before starting the service
- [ ] guide documents how to verify the service is healthy after deployment
- [ ] guide tested by following steps on a clean environment and confirming successful startup
- [ ] Typecheck passes

---

### US-124: Environment variable and configuration reference

**Description:** As a developer, I want a single reference document listing every configuration option so that operators can tune Rex for their environment without reading source code.

**Acceptance Criteria:**
- [ ] `docs/configuration.md` exists and lists every environment variable with: name, description, default, required/optional
- [ ] document organized into logical sections (server, database, LLM providers, integrations, logging)
- [ ] document consistent with `.env.example` (no variables in one but not the other)
- [ ] Typecheck passes

---

### US-125: Operations runbook

**Description:** As an operator, I want a runbook covering common operational tasks so that I can start, stop, restart, and diagnose Rex without escalating to a developer.

**Acceptance Criteria:**
- [ ] `docs/runbook.md` exists and covers: start/stop/restart procedure, log access and filtering, health check verification, what to do if a service fails to start
- [ ] runbook documents the expected process list and how to verify each component is running
- [ ] at least five common error scenarios documented with diagnosis steps and resolution
- [ ] Typecheck passes

---

### US-126: API reference documentation

**Description:** As a developer or integrator, I want all public API endpoints documented so that I can build integrations without reading source code.

**Acceptance Criteria:**
- [ ] `docs/api.md` or equivalent documents every public endpoint: method, path, request schema, response schema, error codes
- [ ] authentication requirements documented per endpoint
- [ ] at least one example request and response shown per endpoint
- [ ] document consistent with the actual running API (no phantom or missing endpoints)
- [ ] Typecheck passes

---

# PHASE 46 — Deployment Readiness

### US-127: Service startup sequence and dependency ordering

**Description:** As a developer, I want the startup sequence to enforce dependency ordering so that services that depend on the database or config do not start before those dependencies are ready.

**Acceptance Criteria:**
- [ ] startup sequence documented and enforced in code: config validation → database connection → migration check → service initialization → begin accepting traffic
- [ ] if any step fails, subsequent steps do not run
- [ ] startup sequence logged at INFO level so the log stream shows exactly where a failure occurred
- [ ] Typecheck passes

---

### US-128: Process supervisor configuration

**Description:** As an operator, I want Rex services managed by a process supervisor so that crashed services restart automatically and startup on system boot is handled without manual intervention.

**Acceptance Criteria:**
- [ ] a `systemd` unit file (or `supervisor` config, whichever matches the target deployment environment) provided for all long-running Rex processes
- [ ] unit file configures automatic restart on failure (with a backoff limit to prevent restart loops)
- [ ] unit file documented in `docs/deployment.md`
- [ ] starting the unit file on a clean system results in the service coming up and passing the liveness check
- [ ] Typecheck passes

---

### US-129: Production smoke test suite

**Description:** As a developer, I want a smoke test suite that verifies all critical paths against a running instance so that a deployment can be validated in minutes without a full regression run.

**Acceptance Criteria:**
- [ ] smoke tests marked with `@pytest.mark.smoke` and runnable via `pytest -m smoke`
- [ ] smoke tests cover at minimum: health check, authentication, chat message round-trip, notification creation, CLI entrypoints
- [ ] smoke tests connect to a running local instance (not mocks)
- [ ] all smoke tests pass against a freshly started local instance
- [ ] smoke test run time under 2 minutes
- [ ] Typecheck passes

---

# PHASE 47 — Final Production Sign-off

### US-130: Full test suite clean run

**Description:** As a developer, I want the complete test suite to pass with zero failures and zero errors so that there are no known regressions at release.

**Acceptance Criteria:**
- [ ] `pytest` exits with code 0
- [ ] zero test failures, zero test errors
- [ ] zero tests marked `xfail` that are unexpectedly passing (review any xfail markers)
- [ ] test run completes in under 10 minutes on the reference machine
- [ ] Typecheck passes

---

### US-131: Final security scan

**Description:** As a developer, I want a final end-to-end security scan run immediately before release so that no vulnerabilities introduced during the production readiness phase have been missed.

**Acceptance Criteria:**
- [ ] `pip-audit` (or equivalent) returns zero critical or high CVEs
- [ ] secret scan (`gitleaks` or equivalent) returns zero confirmed findings against the full git history including all new commits
- [ ] security headers verified present on a live local instance using `curl -I`
- [ ] findings (if any) documented with remediation status
- [ ] Typecheck passes

---

### US-132: Production readiness checklist sign-off

**Description:** As a developer, I want a completed production readiness checklist committed to the repository so that there is a permanent record that every gate was passed before the first production deployment.

**Acceptance Criteria:**
- [ ] `docs/production-readiness-checklist.md` exists and contains a checklist item for every phase in this PRD
- [ ] every checklist item marked complete with the US number that completed it
- [ ] any items explicitly waived documented with justification
- [ ] document committed to the repository and reviewed by at least one other person before deployment
- [ ] Typecheck passes
