# REX ACTIVE CHECKLIST

## OpenClaw Dynamic Skill and Plugin Integration

### Core OpenClaw Bridge
- [ ] Detect whether OpenClaw gateway/service is running
- [ ] Create Rex ↔ OpenClaw bridge adapter
- [ ] Add configurable OpenClaw endpoint settings
- [ ] Add connection health monitoring
- [ ] Add automatic reconnect handling
- [ ] Add OpenClaw status panel in GUI

### Dynamic Plugin Discovery
- [ ] Detect newly installed OpenClaw plugins automatically
- [ ] Detect newly installed OpenClaw skills automatically
- [ ] Build OpenClaw capability registry sync
- [ ] Refresh capabilities without restarting Rex
- [ ] Categorize imported tools by type and risk level

### Rex Tool Access Layer
- [ ] Allow Rex to call approved OpenClaw plugins
- [ ] Allow Rex to call approved OpenClaw skills
- [ ] Normalize OpenClaw tool responses into Rex tool format
- [ ] Add timeout handling for external plugin calls
- [ ] Add fallback behavior if OpenClaw tools fail

### Safety and Permissions
- [ ] Add allowlist for approved OpenClaw plugins
- [ ] Add denylist support for dangerous plugins
- [ ] Require confirmation for risky actions
- [ ] Add plugin permission profiles per user
- [ ] Add audit logging for all OpenClaw tool usage

### Verification Layer
- [ ] Verify OpenClaw actions before Rex reports success
- [ ] Distinguish between attempted, completed, and verified actions
- [ ] Add retry logic for transient failures
- [ ] Add user-visible error explanations

### GUI Integration
- [ ] Add OpenClaw integration page to Settings
- [ ] Display installed OpenClaw plugins
- [ ] Display installed OpenClaw skills
- [ ] Allow enabling/disabling plugins from GUI
- [ ] Show plugin health and status
- [ ] Add plugin permissions management UI

### Long-Term Architecture Goals
- [ ] Treat OpenClaw as expandable external tool ecosystem
- [ ] Keep Rex as orchestrator, verifier, and safety layer
- [ ] Avoid hard dependency on OpenClaw for core Rex functionality
- [ ] Support distributed voice/device systems through OpenClaw
- [ ] Support future smart speaker endpoints and room-aware devices

## Notes

Goal:
Use OpenClaw and ClawHub as the expandable marketplace/tool ecosystem for Rex so that new abilities can be added through OpenClaw plugins and skills without requiring major direct modifications to the Rex codebase.

Architecture Principle:
Rex remains the primary assistant brain responsible for:
- intent routing
- permissions
- safety
- memory
- verification
- final user responses

OpenClaw acts as a modular external tool and capability provider.
