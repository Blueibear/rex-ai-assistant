# Provider reliability and routing evaluation

US-111 adds a privacy-safe in-memory provider reliability ledger to ModelRouter 2.0.
It records only bounded operational evidence: provider ID, latest bounded latency,
success/failure counters, rate-limit count, consecutive failures, failure category,
and remaining cooldown. It never accepts or stores prompts, responses, credentials,
user identity, exception text, memory content, or tool results.

Provider candidates are explicit and ordered. The router filters that supplied list
by current health and selects the first available candidate. It does not discover,
auto-configure, or silently enable a cloud or paid provider. Existing legacy
`cloud_limit_hit()` behavior remains supported and feeds the same reliability ledger.

## Deterministic evaluation

Run the required offline evaluation with:

```powershell
python scripts/rexbench.py --profile routing-eval --iterations 8
```

The default corpus is checked in at `tests/fixtures/rexbench/routing-eval.json`.
It covers a healthy primary provider, a rate-limit fallback, an outage fallback,
and the fail-closed case where every candidate is unavailable. Results are labeled
`deterministic_local` and report selection accuracy plus bounded routing timings.

## Optional live probe

A live provider probe is deliberately separate and opt-in:

```powershell
python scripts/rexbench.py --profile routing-eval --iterations 8 --live-provider-eval
```

That flag may use the currently configured provider and can therefore incur network
or provider usage. It must never become a required CI dependency. Its output is
labeled `live_provider` separately from deterministic evidence and does not retain
the generated content or raw provider exception message.

## Runtime diagnostics

`ModelRouter.provider_diagnostics()` exposes content-free health metadata suitable
for logs or future status surfaces. Routing evidence uses bounded reason labels such
as `provider_openai_cooldown`, `provider_fallback_selected`, and
`all_providers_unavailable`; these labels explain degradation without exposing a
credential, request, response, or private user value.
