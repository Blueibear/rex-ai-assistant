# AskRex Assistant Documentation Index

This index covers active documentation under `docs/`. Files under `docs/archive/` are historical records from completed development cycles and may intentionally describe superseded plans.

## Start Here

| File | Description |
|---|---|
| [../README.md](../README.md) | Current project overview and quick start |
| [../INSTALL.md](../INSTALL.md) | Supported install paths |
| [../RUNNING.md](../RUNNING.md) | Runtime command guide |
| [../PRD-production-readiness.md](../PRD-production-readiness.md) | Authoritative current production-readiness tracker; Section 13 contains the post-RC self-maintenance roadmap |
| [usage.md](usage.md) | User-facing usage guide |
| [UI_SURFACES.md](UI_SURFACES.md) | Current CLI, GUI, and service surface inventory |
| [BRANDING.md](BRANDING.md) | Canonical naming rules |
| [troubleshooting.md](troubleshooting.md) | Common failures and fixes |

## Architecture and Runtime

| File | Description |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level system architecture |
| [architecture/end-user-installation-and-voice-runtime.md](architecture/end-user-installation-and-voice-runtime.md) | Final consumer installer, background Rex Core/Voice Agent, multi-room endpoint, privacy-control, and screenless release contract |
| [api.md](api.md) | Active HTTP services and endpoint reference |
| [configuration.md](configuration.md) | Runtime configuration reference |
| [environment-variables.md](environment-variables.md) | Environment variable and secret reference |
| [credentials.md](credentials.md) | Credential storage and secret lookup |
| [contracts.md](contracts.md) | Runtime contract definitions |
| [contracts/](contracts/) | JSON schemas for contract payloads |
| [event_bus.md](event_bus.md) | Event bus design |
| [workflow-engine.md](workflow-engine.md) | Workflow execution and approval model |
| [autonomy.md](autonomy.md) | Autonomy mode behavior |
| [policy.md](policy.md) | Policy engine configuration |
| [tools.md](tools.md) | Tool registry and execution reference |
| [SELF_MAINTENANCE.md](SELF_MAINTENANCE.md) | Controlled self-maintenance, GitHub maintainer, verification, privacy authority, and rollback architecture |

## Mobile API Gateway Planning

| File | Description |
|---|---|
| [mobile/MOBILE_API_MASTER_SPEC.md](mobile/MOBILE_API_MASTER_SPEC.md) | Canonical endpoint, authentication, WebSocket, chat, voice, TTS, and security contract |
| [mobile/MOBILE_CLIENT_CONTRACT_AUDIT.md](mobile/MOBILE_CLIENT_CONTRACT_AUDIT.md) | Cross-repository mobile/backend audit and resolved contract conflicts |
| [mobile/MOBILE_API_ARCHITECTURE.md](mobile/MOBILE_API_ARCHITECTURE.md) | Gateway components, session schema, identity propagation, idempotency, and deployment architecture |
| [mobile/MOBILE_API_IMPLEMENTATION_PLAN.md](mobile/MOBILE_API_IMPLEMENTATION_PLAN.md) | Two-session Fable implementation backlog and validation gates |
| [mobile/MOBILE_API_TEST_MATRIX.md](mobile/MOBILE_API_TEST_MATRIX.md) | Automated, integration, LAN, and physical-iPhone validation matrix |

## GUI and Operations

| File | Description |
|---|---|
| [e2e-gui-launch-test.md](e2e-gui-launch-test.md) | Electron GUI launch verification notes |
| [dashboard.md](dashboard.md) | Dashboard API and SSE notes |
| [runbook.md](runbook.md) | Day-to-day operations runbook |
| [deployment.md](deployment.md) | Production deployment guide |
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | Deployment checklist |
| [docker.md](docker.md) | Developer-only Docker smoke-test/operator setup |
| [advanced-install.md](advanced-install.md) | Advanced install, GPU, and developer setup |
| [doctor.md](doctor.md) | `rex doctor` reference |
| [developer_tools.md](developer_tools.md) | Development tooling |
| [distribution.md](distribution.md) | Packaging and distribution notes |
| [github.md](github.md) | GitHub integration and CI notes |
| [performance-baseline.md](performance-baseline.md) | Performance baseline |
| [production-readiness-checklist.md](production-readiness-checklist.md) | Production readiness checks |

## Integrations

| File | Description |
|---|---|
| [calendar.md](calendar.md) | Calendar backend setup |
| [email.md](email.md) | Email backend setup |
| [messaging.md](messaging.md) | SMS/messaging setup |
| [home_assistant.md](home_assistant.md) | Home Assistant integration |
| [wordpress_woocommerce.md](wordpress_woocommerce.md) | WordPress and WooCommerce integration |
| [browser_os.md](browser_os.md) | Browser and OS automation |
| [computers.md](computers.md) | Remote computer control |
| [notifications.md](notifications.md) | Notification delivery and routing |
| [scheduler.md](scheduler.md) | Scheduler usage |
| [scheduler_event_bus.md](scheduler_event_bus.md) | Scheduler/event bus integration |
| [followup_engine.md](followup_engine.md) | Follow-up cue engine |
| [knowledge_base.md](knowledge_base.md) | Knowledge base storage and search |
| [memory.md](memory.md) | Memory systems |
| [voice_pipeline.md](voice_pipeline.md) | Canonical structured voice-pipeline timing log contract |
| [voice/wakeword-report.md](voice/wakeword-report.md) | Controlled synthetic acoustic wake-word precision/recall/latency evidence |
| [voice_identity.md](voice_identity.md) | Voice identity and enrollment |
| [openclaw-agent-setup.md](openclaw-agent-setup.md) | OpenClaw gateway setup |
| [openclaw-migration-status.md](openclaw-migration-status.md) | OpenClaw migration history/status |

## Security

| File | Description |
|---|---|
| [hardening.md](hardening.md) | Hardening checklist |
| [security-scan.md](security-scan.md) | Security scan procedures |
| [SECURITY_DEPENDENCIES.md](SECURITY_DEPENDENCIES.md) | Security-sensitive dependency notes |
| [security/SECURITY_ADVISORY.md](security/SECURITY_ADVISORY.md) | Security advisory |
| [security/SECURITY_AUDIT_2026-01-08.md](security/SECURITY_AUDIT_2026-01-08.md) | Security audit report |
| [security/SECRET-SCAN.md](security/SECRET-SCAN.md) | Secret scan report |
| [security/VULNERABILITY-SCAN.md](security/VULNERABILITY-SCAN.md) | Vulnerability scan report |

## Claude/Codex Reference

| File | Description |
|---|---|
| [claude/COMMANDS_AND_ENTRYPOINTS.md](claude/COMMANDS_AND_ENTRYPOINTS.md) | CLI commands and entry points reference |
| [claude/CONFIG_AND_SECURITY.md](claude/CONFIG_AND_SECURITY.md) | Config and security patterns |
| [claude/INTEGRATIONS_STATUS.md](claude/INTEGRATIONS_STATUS.md) | Current integration readiness snapshot |
| [claude/TESTING_AND_QUALITY.md](claude/TESTING_AND_QUALITY.md) | Testing and quality expectations |
| [prompts/CLAUDE_BUILD_PROMPT.txt](prompts/CLAUDE_BUILD_PROMPT.txt) | Claude build prompt template |
| [prompts/CODEX_REVIEW_FIX_PROMPT.txt](prompts/CODEX_REVIEW_FIX_PROMPT.txt) | Codex review/fix prompt template |
| [codex_verification_audit_2026-02-16.md](codex_verification_audit_2026-02-16.md) | Codex verification audit |

## Reference and Historical Active Files

| File | Description |
|---|---|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Current quick reference |
| [INSTRUCTION_MANUAL.md](INSTRUCTION_MANUAL.md) | Long-form instruction manual, may lag the top-level quick start |
| [WORKFLOW.md](WORKFLOW.md) | Development workflow notes |
| [README_STABILIZATION.md](README_STABILIZATION.md) | Historical stabilization notes |
| [FINAL_DELIVERY_SUMMARY.txt](FINAL_DELIVERY_SUMMARY.txt) | Historical delivery summary |
| [VERIFICATION_REPORT_VOICE_IDENTITY_BL009_BL012.md](VERIFICATION_REPORT_VOICE_IDENTITY_BL009_BL012.md) | Voice identity verification report |
| [Rex_AI_Assistant_Blueprint.pdf](Rex_AI_Assistant_Blueprint.pdf) | Blueprint PDF |

## Archive

| Path | Description |
|---|---|
| [archive/prd/INDEX.md](archive/prd/INDEX.md) | Archived PRDs |
| [archive/verification/INDEX.md](archive/verification/INDEX.md) | Archived verification reports |
| [archive/housekeeping/](archive/housekeeping/) | Archived housekeeping notes and patches |
| [archive/progress/](archive/progress/) | Archived progress logs |
