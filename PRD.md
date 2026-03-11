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
- [ ] triage assigns one of four categories: urgent, action_required, fyi, newsletter
- [ ] categorization logic uses sender address, subject keywords, and body patterns
- [ ] triage results are queryable (e.g., "show urgent emails" returns only urgent-tagged items)
- [ ] test using mock inbox data confirms at least one email correctly categorized into each category
- [ ] Typecheck passes

---

### US-080: Email triage rules engine

**Description:** As a developer, I want triage rules stored in config and evaluated in priority order so that categorization can be customized per user without code changes.

**Acceptance Criteria:**
- [ ] triage rules stored in config file (JSON or YAML)
- [ ] rules support matching on: sender address, subject pattern, body keyword
- [ ] rules evaluated in declared priority order; first match wins
- [ ] adding or modifying a rule takes effect without restarting or modifying source code
- [ ] Typecheck passes

---

### US-081: Calendar free/busy stub

**Description:** As a developer, I want a stub calendar that returns mock free/busy data so that scheduling features can be built and tested without live credentials.

**Acceptance Criteria:**
- [ ] `CalendarStub` class returns mock free/busy blocks for a configurable date range
- [ ] stub implements the same interface as the real calendar backend (US-045)
- [ ] tests can query availability without any live credentials or network calls
- [ ] Typecheck passes

---

### US-082: Free time finder

**Description:** As a user, I want Rex to find available meeting slots from my calendar so that I can ask "when am I free?" and get usable suggestions.

**Acceptance Criteria:**
- [ ] given a date range and meeting duration, returns a list of available time slots
- [ ] overlapping calendar events are excluded from returned slots
- [ ] returns at least three candidate slots when the calendar is not fully booked
- [ ] works correctly against stub/mock calendar data in beta
- [ ] Typecheck passes

---

### US-083: Meeting invite scaffold

**Description:** As a user, I want Rex to draft a meeting invite with attendees, time, and agenda from a natural language request so that I can schedule by describing what I want.

**Acceptance Criteria:**
- [ ] `MeetingInvite` data structure contains: title, attendees list, start time, end time, agenda
- [ ] Rex can populate all invite fields from a natural language description
- [ ] completed invite is displayed to the user for review before any action
- [ ] stub send logs the invite and returns success without calling any real calendar API
- [ ] Typecheck passes

---

# PHASE 35 — SMS Multi-Channel Messaging (beta — stub scaffolding)

> **Beta scope:** All stories in this phase are stub scaffolding. No real SMS messages are sent. Real delivery requires Twilio credentials to be wired into the `TwilioAdapter` interface defined in US-086.

### US-084: SMS send stub

**Description:** As a developer, I want a stub SMS sender that logs outbound messages so that SMS-triggered workflows can be built and tested without Twilio credentials.

**Acceptance Criteria:**
- [ ] `SmsSenderStub` class accepts a phone number and message body
- [ ] sent messages written to a structured in-memory log accessible for test assertions
- [ ] stub implements the same interface as the real Twilio adapter (US-086)
- [ ] calling `send` on the stub makes no network calls
- [ ] Typecheck passes

---

### US-085: SMS receive stub

**Description:** As a developer, I want a stub SMS receiver that can inject test inbound messages so that inbound SMS handling can be built and tested without a live Twilio webhook.

**Acceptance Criteria:**
- [ ] `SmsReceiverStub` class exposes a method to inject a test inbound message
- [ ] injected messages are routed through the same handler as real inbound SMS would be
- [ ] handler produces a response or triggers the expected downstream action
- [ ] Typecheck passes

---

### US-086: Twilio adapter interface

**Description:** As a developer, I want a well-defined Twilio adapter interface so that the stub and the real Twilio client are interchangeable without changing any calling code.

**Acceptance Criteria:**
- [ ] `TwilioAdapter` abstract class or Protocol defined with at minimum `send_sms(to: str, body: str)` signature
- [ ] `SmsSenderStub` fully implements `TwilioAdapter`
- [ ] swapping stub for a real Twilio client requires no changes outside the adapter registration point
- [ ] Typecheck passes

---

### US-087: Multi-channel message router

**Description:** As a developer, I want outbound messages routed to the correct channel (dashboard, email, SMS) based on configuration so that Rex can deliver messages where the user prefers without hardcoding a channel.

**Acceptance Criteria:**
- [ ] router accepts a message payload and a target channel identifier
- [ ] routes correctly to dashboard, email, and SMS backends based on channel value
- [ ] unknown or unconfigured channel raises a handled error and does not crash the assistant
- [ ] active channel configurable without code changes
- [ ] Typecheck passes

---

# PHASE 36 — Smart Notifications (beta — stub scaffolding)

> **Beta scope:** Digest delivery, escalation, and quiet-hour release are stub scaffolding — they log output rather than making real deliveries. Core priority tagging and routing rules are fully implemented.

### US-088: Notification priority levels

**Description:** As a developer, I want notifications to carry a priority level so that routing and delivery decisions can be based on urgency rather than treating all notifications equally.

**Acceptance Criteria:**
- [ ] `NotificationPriority` enum defined with values: critical, high, medium, low
- [ ] all notification creation paths accept a `priority` parameter
- [ ] priority stored alongside notification record in the database
- [ ] existing notifications without a stored priority default to `medium` on read
- [ ] Typecheck passes

---

### US-089: Priority routing rules

**Description:** As a developer, I want routing rules that deliver critical and high notifications immediately while queuing medium and low ones so that users are not interrupted by low-priority alerts.

**Acceptance Criteria:**
- [ ] critical and high priority notifications dispatched to configured delivery channels immediately on creation
- [ ] medium and low priority notifications placed in the digest queue instead of immediate delivery
- [ ] routing rules configurable without code changes
- [ ] unit test confirms a critical notification bypasses the digest queue
- [ ] unit test confirms a low notification is placed in the digest queue
- [ ] Typecheck passes

---

### US-090: Digest mode

**Description:** As a user, I want low-priority notifications batched into periodic digests so that I receive a single summary instead of many individual interruptions.

**Acceptance Criteria:**
- [ ] digest job runs on a configurable interval (default: 60 minutes)
- [ ] digest collects all queued medium and low notifications since the last digest run
- [ ] digest payload delivered to the dashboard notification endpoint as a single grouped message
- [ ] digest job logs output when no real delivery backend is configured (beta stub behavior)
- [ ] digest queue is cleared after each successful run
- [ ] Typecheck passes

---

### US-091: Quiet hours

**Description:** As a user, I want to configure quiet hours so that non-critical notifications are held until I'm available rather than interrupting me at night or during focus time.

**Acceptance Criteria:**
- [ ] quiet hours configured as start time and end time in user config
- [ ] non-critical (medium, low) notifications generated during quiet hours are held in queue
- [ ] critical notifications bypass quiet hours and deliver immediately regardless of schedule
- [ ] held notifications are released and delivered when quiet hours end
- [ ] Typecheck passes

---

### US-092: Auto-escalation

**Description:** As a developer, I want unacknowledged high-priority notifications to escalate after a configurable timeout so that important alerts are not silently missed.

**Acceptance Criteria:**
- [ ] escalation timeout configurable per priority level (default: 15 minutes for high)
- [ ] escalation job checks for unacknowledged high-priority notifications past their timeout
- [ ] each escalation attempt logged with timestamp, notification ID, and attempt number
- [ ] escalation stops after a configurable maximum attempt count (default: 3)
- [ ] escalation stub logs events without making real deliveries in beta
- [ ] Typecheck passes
