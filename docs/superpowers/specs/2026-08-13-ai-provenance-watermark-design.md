# Rex 2.0 AI Provenance & Watermark Capability Design

Date: 2026-08-13
Status: Approved design; implementation intentionally deferred until Rex 2.0 prerequisites are stable.

## Purpose

Add an `AI Provenance & Watermark` capability to Rex 2.0 that can inspect and clean AI provenance markers from user-owned text and files, optionally reduce statistical text-watermark signals through rewriting, and optionally detect/remove pixel-domain image watermarks through external heavy backends.

The capability must use the Rex 2.0 canonical runtime, policy, action lifecycle, cancellation, audit, and verification paths. It must not become a legacy direct-execution skill that can mutate files outside those controls.

## Upstream baseline

Primary upstream project: `guillaumemeyer/watermarks-remover`.

Initial integration baseline:
- Release: `v0.4.0`
- Commit: `c267d785e8c09856bb2f9316ea8de3064651fe79`
- Primary license: MIT

Rex must pin the exact reviewed revision. Updating the pin requires compatibility/security review and passing integration tests. Rex must never execute an unreviewed moving upstream `main` branch as the production dependency.

## Architecture

### Rex ownership boundary

Rex remains responsible for:
- natural-language intent and capability selection;
- validated user/device identity;
- authorization and confirmation;
- path and output policy;
- action lifecycle and cancellation;
- audit logging;
- independent post-action verification where technically possible;
- truthful user-facing reporting.

The upstream toolkit remains an implementation backend. Optional reverse-SynthID and CtrlRegen/noai-watermark components remain external runtimes rather than Rex-owned code.

### Proposed modules

```text
plugins/skills/ai_provenance_skill.py      # discovery/intent metadata only
rex/provenance/service.py                  # capability service/orchestration
rex/provenance/models.py                   # typed requests/findings/results
rex/provenance/upstream.py                 # pinned upstream adapter
rex/provenance/inspector.py                # read-only inspection
rex/provenance/cleaner.py                  # deterministic cleaning
rex/provenance/rewrite.py                  # statistical rewrite integration
rex/provenance/pixel.py                    # optional pixel backends
rex/openclaw/tools/ai_provenance.py        # canonical Rex tool/action surface if retained by current 2.0 tool architecture
```

Exact file placement may change to match the final Rex 2.0 capability contract. The invariant is that file mutations must go through the canonical action lifecycle rather than the legacy `SkillRouter.execute()` direct handler path.

## Tier 1: deterministic inspection and cleaning

Tier 1 is the default installed capability.

Supported target classes:
- plain text and Markdown;
- HTML;
- PNG/JPEG;
- SVG;
- PDF;
- DOCX;
- ODT;
- directories;
- websites/sitemaps for audit-only remote inspection.

Supported functions include:
- invisible Unicode and exotic-space hygiene;
- C2PA/Content Credentials inspection/removal where hard-bound metadata is accessible;
- EXIF/XMP and container/document metadata inspection/removal;
- aggregate directory and website audits;
- confidence-classified findings.

### Mutation workflow

1. Inspect the source.
2. Resolve the requested cleaning operation.
3. Authorize the write through Rex policy.
4. Preserve the original by default and write `*.cleaned.*`.
5. Execute the deterministic clean.
6. Re-inspect the output.
7. Report only removals supported by before/after evidence as verified.
8. Report unsupported or best-effort surfaces separately.

An explicit in-place operation may be supported only when Rex policy allows it and the user has intentionally requested it.

## Tier 2: statistical text-watermark reduction

Tier 2 is optional per request and uses Rex's existing request-local LLM routing.

Default workflow:
1. Layer A deterministic text hygiene.
2. Rewrite with substantial lexical and syntactic churn while preserving facts, numbers, names, and technical identifiers.
3. Prefer a rewrite model different from the suspected source model when practical.
4. Run deterministic hygiene again.
5. Report Layer A results as verifiable and Layer B as best-effort.

Rex must not claim that text is human-written, undetectable, or guaranteed to fail an official vendor detector unless an authoritative detector provides such evidence.

## Tier 3: pixel-domain image support

Tier 3 is optional and heavy. It is not part of the default install.

### reverse-SynthID

Purpose: optional scoring/detection signal.

Requirements:
- external checkout/runtime only;
- never vendored into AskRex;
- retain upstream non-commercial research licensing boundary;
- describe the score as unofficial/best-effort rather than an official Google detector result.

### CtrlRegen / noai-watermark

Purpose: optional regenerating removal backend for pixel-domain watermark classes.

Requirements:
- external checkout/runtime only;
- never vendored while upstream lacks a permissive license;
- approximately 10 GB of model downloads expected;
- GPU strongly recommended;
- model/runtime setup is explicit opt-in;
- default strength `0.25`;
- documented presets may expose `0.15`, `0.25`, `0.35`, `0.5`, and `0.7` with a clear warning that stronger settings regenerate more image content.

A missing GPU, model, gated-model token, or external checkout is an optional-component availability state, not a failure of core Rex.

## Security and supply chain

- Pin reviewed upstream revisions.
- Verify expected upstream revision before execution.
- Keep secrets in Rex's credential/vault path or environment-only handoff; never place secrets in argv or logs.
- Enforce Rex path traversal, symlink, output-root, overwrite, and user-scope protections on all writes.
- Treat unsupported/unknown health as unavailable or degraded rather than healthy.
- Preserve cancellation semantics and truthful action state.
- Never allow a skill wrapper to bypass confirmation or verification for writes.

## Licensing

- `watermarks-remover`: MIT at the reviewed baseline.
- `reverse-SynthID`: external only; retain upstream non-commercial research terms.
- `mertizci/noai-watermark` / CtrlRegen backend: external only; do not vendor or redistribute while no permissive license is present.

Optional component setup must expose these licensing distinctions before installation/enabling.

## Testing

Required normal-CI coverage:
- capability/intent routing;
- unsupported file types;
- binary-vs-text refusal;
- original preservation;
- Unicode cleaning;
- C2PA/EXIF/XMP/container metadata inspection and cleaning;
- PDF/DOCX/ODT/HTML/Markdown paths;
- before/after verification;
- false-success prevention;
- rewrite degradation/failure behavior;
- semantic-preservation constraints;
- path traversal and symlink refusal;
- permission denial;
- timeout/cancellation;
- missing optional components;
- Tier 3 strength mapping.

Heavy Tier 3 model execution must be mocked in normal GitHub Actions. A separate documented integration procedure may exercise the real models on an approved GPU-equipped development machine.

## Prerequisites for implementation

Implementation begins only after the relevant Rex 2.0 production path is stable enough that:
- capability retrieval/routing is canonical;
- file-writing tools execute through the canonical action lifecycle;
- mutation verification and cancellation semantics are stable;
- failure/recovery states can represent unavailable/degraded optional backends truthfully.

OpenClaw may expose or distribute capabilities in the future, but this feature must not require OpenClaw to function. Rex remains the orchestrator and authority.

## Success criteria

The capability is complete when Rex can accept natural-language requests to inspect or clean supported user-owned content, route the work through Rex 2.0 policy/action infrastructure, preserve originals by default, verify deterministic results where possible, truthfully distinguish best-effort statistical/pixel operations, and optionally use the external Tier 3 backends without making them core Rex dependencies.
