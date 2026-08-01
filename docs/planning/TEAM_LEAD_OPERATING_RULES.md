# AskRex Delivery Operating Rules

## Authority and goal

The delivery lead may create branches and worktrees, commit, push, open pull requests, merge, and perform non-destructive repository operations without waiting for approval.

The goal is a shippable AskRex Assistant desktop application and privately distributed mobile application that are secure, reliable, fast or perceptually responsive, visually aligned, intuitive, and truthful about capability availability.

## Product sources of truth

1. `docs/planning/source-of-truth/REX_Unified_Build_Spec_UPDATED.md`
2. `docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md`

All other PRDs and roadmaps are supporting inputs. Review them for valuable capabilities, but they cannot override the two authoritative documents or reintroduce superseded architecture without evidence.

## Agent usage

- The delivery lead supervises Claude Code and Codex as implementation and review agents.
- Give each agent focused, file-specific assignments in isolated worktrees.
- Review and independently validate every claimed change before integration.
- Avoid redundant broad audits unless independent verification is justified.
- Use existing Claude account allowance and credit balance only.
- Never buy additional Anthropic credits or any other paid service.
- Prefer local tests and targeted context to conserve model usage.

## Delivery requirements

- Every real desktop capability must be discoverable, usable, and configurable in the Electron app.
- Mobile must expose mobile-appropriate capabilities and securely request authorized desktop-native actions through a paired desktop session.
- Desktop and mobile must share branding, logo assets, design tokens, terminology, and interaction patterns.
- Features that are incomplete must be completed or truthfully disabled/hidden; no misleading controls or success messages.
- Preserve AskRex as the orchestrator, identity, memory, policy, verification, and response layer. OpenClaw remains optional and must not be required for core functionality.
- Improve intelligence through context retrieval, memory, model routing and fallback, tool selection, planning, verification, recovery, and user-specific preferences.
- Improve actual and perceived latency with streaming, immediate acknowledgements, persistent processes, warmup, progressive status, cancellation, and background execution.

## Security baseline

- Least privilege and deny-by-default for privileged operations.
- Device-bound desktop/mobile pairing with short-lived one-time codes or QR enrollment.
- Per-user and per-device capability scopes, revocation, expiry, and audit history.
- Strong reauthentication for high-risk mobile requests.
- Encrypted transport and secure OS credential storage.
- Replay protection, rate limiting, strict validation, secret redaction, and verified updates.
- Risk-classified confirmations and independent post-action verification.
- No action is reported as successful unless verified, or explicitly described as attempted but unverified.

## External dependencies

Complete software work first and maintain one consolidated checklist for credentials, hardware, and real external services. Do not interrupt the user for each dependency and do not incur charges.