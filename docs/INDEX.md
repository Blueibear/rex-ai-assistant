# AskRex Assistant — Documentation Index

All documentation files under `docs/`, organized by category.

---

## Architecture

| File | Description |
|------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level system architecture overview |
| [autonomy.md](autonomy.md) | Autonomy engine design and behavior |
| [event_bus.md](event_bus.md) | Event bus architecture and message routing |
| [scheduler_event_bus.md](scheduler_event_bus.md) | Scheduler and event bus integration patterns |
| [workflow-engine.md](workflow-engine.md) | Workflow engine design and execution model |
| [contracts.md](contracts.md) | Runtime contract definitions and protocol interfaces |
| [contracts/](contracts/) | JSON schema files for typed contracts (Action, Task, ToolCall, etc.) |

---

## Configuration

| File | Description |
|------|-------------|
| [configuration.md](configuration.md) | Application configuration reference (AppConfig, rex_config.json) |
| [environment-variables.md](environment-variables.md) | Complete environment variable reference |
| [credentials.md](credentials.md) | Credential management and secret storage guide |
| [policy.md](policy.md) | Policy configuration for access control |
| [hardening.md](hardening.md) | Security hardening checklist and recommendations |

---

## Integrations

| File | Description |
|------|-------------|
| [calendar.md](calendar.md) | Calendar integration (Google, ICS) setup and usage |
| [email.md](email.md) | Email backend setup and usage |
| [messaging.md](messaging.md) | Messaging backend (SMS, Twilio) setup |
| [home_assistant.md](home_assistant.md) | Home Assistant MQTT integration |
| [wordpress_woocommerce.md](wordpress_woocommerce.md) | WordPress/WooCommerce read/write integration |
| [browser_os.md](browser_os.md) | Browser and OS automation via OpenClaw |
| [computers.md](computers.md) | Desktop automation integration |
| [openclaw-agent-setup.md](openclaw-agent-setup.md) | OpenClaw agent server setup guide |
| [openclaw-migration-status.md](openclaw-migration-status.md) | OpenClaw HTTP migration progress and status |
| [scheduler.md](scheduler.md) | Task scheduler configuration and usage |
| [notifications.md](notifications.md) | Notification delivery (SSE, push) setup |
| [knowledge_base.md](knowledge_base.md) | Knowledge base integration |
| [memory.md](memory.md) | Per-user memory system design and usage |
| [followup_engine.md](followup_engine.md) | Follow-up engine for deferred tasks |
| [voice_identity.md](voice_identity.md) | Voice identity and speaker recognition |
| [tools.md](tools.md) | Built-in tool reference (search, weather, etc.) |

---

## API

| File | Description |
|------|-------------|
| [api.md](api.md) | REST API endpoints and authentication reference |
| [dashboard.md](dashboard.md) | Dashboard API and SSE event streaming |
| [deployment.md](deployment.md) | Production deployment guide (systemd, reverse proxy) |
| [docker.md](docker.md) | Docker container setup and usage |
| [runbook.md](runbook.md) | Operational runbook for production deployments |
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | Step-by-step deployment checklist |

---

## Development

| File | Description |
|------|-------------|
| [DEPENDENCIES.md](DEPENDENCIES.md) | Dependency overview and version constraints |
| [developer_tools.md](developer_tools.md) | Development tooling and local workflow |
| [distribution.md](distribution.md) | Packaging and distribution guide |
| [github.md](github.md) | GitHub Actions and CI/CD configuration |
| [performance-baseline.md](performance-baseline.md) | Performance benchmarks and optimization targets |
| [production-readiness-checklist.md](production-readiness-checklist.md) | Pre-production readiness checks |
| [troubleshooting.md](troubleshooting.md) | Common issues and solutions |
| [doctor.md](doctor.md) | `rex doctor` health check command reference |
| [audit.md](audit.md) | Code and security audit notes |
| [branch_sync.md](branch_sync.md) | Branch synchronization procedures |
| [codex_verification_audit_2026-02-16.md](codex_verification_audit_2026-02-16.md) | Codex verification audit (2026-02-16) |
| [security-scan.md](security-scan.md) | Security scanning procedures and results |
| [SECURITY_DEPENDENCIES.md](SECURITY_DEPENDENCIES.md) | Security-relevant dependency notes |
| [usage.md](usage.md) | End-user usage guide |
| [advanced-install.md](advanced-install.md) | Advanced installation scenarios |
| [INSTRUCTION_MANUAL.md](INSTRUCTION_MANUAL.md) | Comprehensive user instruction manual |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick reference card for common operations |
| [WORKFLOW.md](WORKFLOW.md) | Development workflow and contribution guide |
| [README_STABILIZATION.md](README_STABILIZATION.md) | Stabilization notes from October 2025 |
| [FINAL_DELIVERY_SUMMARY.txt](FINAL_DELIVERY_SUMMARY.txt) | Delivery summary from October 2025 stabilization |
| [VERIFICATION_REPORT_VOICE_IDENTITY_BL009_BL012.md](VERIFICATION_REPORT_VOICE_IDENTITY_BL009_BL012.md) | Voice identity verification (BL-009, BL-012) |
| [Rex_AI_Assistant_Blueprint.pdf](Rex_AI_Assistant_Blueprint.pdf) | System blueprint (PDF) |

### Claude Code Reference

| File | Description |
|------|-------------|
| [claude/COMMANDS_AND_ENTRYPOINTS.md](claude/COMMANDS_AND_ENTRYPOINTS.md) | CLI commands and entry point reference for Claude Code |
| [claude/CONFIG_AND_SECURITY.md](claude/CONFIG_AND_SECURITY.md) | Config and security patterns for Claude Code sessions |
| [claude/INTEGRATIONS_STATUS.md](claude/INTEGRATIONS_STATUS.md) | Integration completion status |
| [claude/TESTING_AND_QUALITY.md](claude/TESTING_AND_QUALITY.md) | Testing standards and quality requirements |

### Prompt Templates

| File | Description |
|------|-------------|
| [prompts/CLAUDE_BUILD_PROMPT.txt](prompts/CLAUDE_BUILD_PROMPT.txt) | Claude build prompt template |
| [prompts/CODEX_REVIEW_FIX_PROMPT.txt](prompts/CODEX_REVIEW_FIX_PROMPT.txt) | Codex review and fix prompt template |

---

## Security

| File | Description |
|------|-------------|
| [security/INDEX.md](security/INDEX.md) | Security documents index |
| [security/SECURITY_AUDIT_2026-01-08.md](security/SECURITY_AUDIT_2026-01-08.md) | Full security audit conducted 2026-01-08 |
| [security/SECURITY_FIX_SUMMARY.md](security/SECURITY_FIX_SUMMARY.md) | Security fixes applied after 2026-01-08 audit |
| [security/SECRET-SCAN.md](security/SECRET-SCAN.md) | Secret scanning results |
| [security/VULNERABILITY-SCAN.md](security/VULNERABILITY-SCAN.md) | Dependency vulnerability scan results |

---

## Archive

Historical files from completed development cycles.

| File | Description |
|------|-------------|
| [archive/prd/INDEX.md](archive/prd/INDEX.md) | Index of archived PRD files |
| [archive/verification/INDEX.md](archive/verification/INDEX.md) | Index of archived verification reports |
| [archive/housekeeping/](archive/housekeeping/) | Archived housekeeping and summary files |
