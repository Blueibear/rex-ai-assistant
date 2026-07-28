# Tool execution and verification contract

All canonical local and OpenClaw tool dispatch passes through
`rex.tools.execution.ToolExecutionLifecycle`. The lifecycle records these stages in order:

1. Capability availability
2. Argument validation
3. User identity validation
4. Permission evaluation
5. Risk classification
6. Confirmation when required
7. Execution
8. Normalized result
9. Independent post-action verification
10. Truthful response generation
11. Redacted audit recording

Read-only tools may return `completed` after a successful call. Mutating tools may return
`verified` only when their registered verifier succeeds or when a specialized policy service (such
as Home Assistant) supplies a verified normalized result. A successful HTTP response, accepted
request, returned object, or explicit user request is not independent proof. A mutation without a
verifier returns `attempted_unverified`; a timeout after a possible write uses the same status.

The normalized statuses are `completed` (read-only only), `verified`, `attempted_unverified`,
`confirmation_required`, `denied`, `failed`, and `unavailable`. Results include the request ID,
risk classification, executed stages, detail, output, and any error. Duplicate mutation request IDs
are deduplicated per user and tool; reuse with different arguments is denied.

Registered mutations require a canonical user identity. Missing arguments, invalid identities,
permission failures, prohibited operations, and unavailable capabilities fail before the handler is
called. Gateway-only OpenClaw discovery entries raise an explicit unavailable error if mistakenly
invoked locally; no empty no-op result is registered as success.

Lifecycle audit entries store argument names plus a deterministic argument hash, not argument
values, and then pass through the existing recursive sensitive-key redaction. Tool implementations
must still avoid putting message bodies, credentials, confirmation tokens, or personal content in
free-form error strings.
