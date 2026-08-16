# US-113 OpenClaw/ClawHub Capability Sync Implementation Plan

**Goal:** Discover the configured OpenClaw gateway's tool/skill inventory with least-privilege authenticated WebSocket RPC, normalize untrusted metadata into Rex's canonical capability model, and atomically refresh the remote snapshot without allowing remote metadata to widen local authority.

## Implementation sequence

1. **Gateway read-only RPC client**
   - Add `websocket-client>=1.9.0,<2` as a lightweight direct dependency.
   - Add `rex/openclaw/gateway_rpc.py` with HTTP(S)->WS(S) URL conversion, challenge/connect handshake, `operator.read` scope only, bounded timeout, request/response correlation, and sanitized errors/logs.
   - Keep token out of command lines, exceptions, and logs.

2. **Capability sync model + persistence**
   - Add `rex/openclaw/capability_sync.py`.
   - Fetch `tools.catalog`, optional `tools.effective` when a configured/default session key is resolvable, and `skills.status`.
   - Validate bounded IDs/descriptions/schemas/status records before normalization; reject malformed/duplicate-conflicting snapshots before mutation.
   - New remote capabilities use conservative OpenClaw security defaults. Existing local canonical security metadata is never replaced.
   - Persist the last validated remote snapshot to household runtime data, never to tracked config.

3. **Atomic registry refresh**
   - Add a bounded atomic remote-snapshot apply helper to `CapabilityRegistry` under a lock.
   - Stage the whole desired remote set before apply.
   - Removed remote cards remain present but disabled/unavailable so stale capabilities never linger executable.
   - Sync failure preserves the last safe snapshot while marking its remote runtime state unhealthy/unavailable; local cards remain untouched.

4. **Lifecycle hooks**
   - Expose startup/manual/hot refresh entry points without adding a polling thread.
   - Initialize once in the service bootstrap when OpenClaw tools are explicitly enabled and gateway config is complete.
   - Keep reconnect-triggered refresh for US-114; US-113 exposes the refresh function/event seam only.

5. **Verification and docs**
   - TDD `tests/rex2/test_openclaw_capability_sync.py` for add/update/remove/malformed/duplicate/malicious metadata/failure snapshot preservation.
   - Extend `tests/test_openclaw_http_client.py` or a focused RPC test for auth/scopes/handshake/timeouts/token redaction.
   - Update `CLAUDE.md`, `PRD-production-readiness.md`, and progress history in the same implementation commit; leave GitHub criterion unchecked until exact-head CI passes.

## Security invariants
- Remote descriptions/schemas are data, never executable instructions.
- Remote metadata may not weaken operation/risk/permissions/identity/verification already owned locally.
- Catalog presence alone does not prove execution liveness; only current session-effective tool IDs receive executable bindings. ClawHub skill status remains informational unless OpenClaw exposes a real effective tool ID.
- Unknown effective OpenClaw tools require validated identity, Rex `openclaw_execute` permission (or `admin`), and sensitive-action confirmation before gateway dispatch.
- No OpenClaw failure may block core local Rex startup.
- No write/admin gateway scope is requested by discovery.
