# Mobile API Gateway Agent Rules

The root `CLAUDE.md` remains authoritative. For issue #323 work, also read every file in this directory before editing.

Mandatory rules:

- Follow `MOBILE_API_MASTER_SPEC.md` as the canonical wire and security contract.
- Credentials establish an immutable server-side principal.
- Never trust client `user_id`, role, permissions, risk level, approval state, or biometric state.
- Validate the canonical user ID before private state, path, cache, credential, database, event, memory, history, or tool access.
- Pass the validated identity explicitly into the canonical `Assistant`; never mutate a process-global current user.
- Never put access tokens, refresh tokens, or TTS text in URLs.
- All mobile wire fields use the documented `snake_case` contract.
- HTTP and WebSocket retries share server-side idempotency keyed by `(user_id, message_id)` before acknowledgement or tool execution.
- Reuse the existing users database, permissions, policy, approvals, Assistant, STT, and TTS systems.
- Unsupported surfaces return explicit 501 errors and false capabilities; never fake data or success.
- Normal conversational output is `completed`. `verified` requires real completion evidence or state readback.
- Default network binding is localhost. LAN binding remains authenticated, rate-limited, and development-only without TLS.
- Update the root `CLAUDE.md` in implementation PRs when commands, configuration, dependencies, environment variables, integrations, or runtime package structure change.
