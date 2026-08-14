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

## Rex 2.0 Capability Backlog

### AI Provenance & Watermark

**Status:** Planned; implement after the canonical Rex 2.0 capability/action path is stable. Do not implement as a legacy direct-execution `plugins/skills/*.py` mutation path.

**Design:** `docs/superpowers/specs/2026-08-13-ai-provenance-watermark-design.md`

**Upstream baseline:** `guillaumemeyer/watermarks-remover` v0.4.0, commit `c267d785e8c09856bb2f9316ea8de3064651fe79`. Pin the exact upstream commit; do not execute an unreviewed moving `main`/`latest`.

**Prerequisites:**
- [ ] Canonical capability retrieval/routing is the production path for discoverable Rex abilities.
- [ ] File-writing operations execute through the canonical action lifecycle, including authorization, cancellation, audit, and verification.
- [ ] Request-level failure/recovery semantics are stable enough to classify unavailable optional backends without false success.
- [ ] External capability/dependency health can be represented truthfully as unavailable, degraded, ready, or optional rather than assumed healthy.

**Tier 1 - deterministic inspection and cleaning (default):**
- [ ] Add `AI Provenance & Watermark` as a Rex 2.0 capability with natural-language routing.
- [ ] Support unified inspect/clean for PNG, JPEG, SVG, PDF, DOCX, ODT, HTML, Markdown, and text.
- [ ] Support invisible Unicode hygiene, C2PA/Content Credentials, EXIF, XMP, document/container metadata, directory audits, and website/sitemap audits.
- [ ] Inspect before mutation and re-inspect the output after mutation.
- [ ] Preserve originals by default and write `*.cleaned.*` outputs unless the user explicitly requests an in-place operation and policy allows it.
- [ ] Keep deterministic findings separate from probable/informational/false-positive findings.
- [ ] Never claim removal was verified unless the post-clean inspection provides evidence for that specific marker/container.

**Tier 2 - statistical text-watermark reduction (optional per request):**
- [ ] Add Layer B rewrite support through Rex's existing request-local LLM routing rather than a mandatory new paid provider.
- [ ] Prefer a rewrite model different from the suspected source model when practical.
- [ ] Preserve facts, numbers, names, technical identifiers, and user-requested meaning constraints.
- [ ] Run deterministic text hygiene again after rewriting.
- [ ] Report statistical watermark reduction as best-effort; never claim official vendor-detector success without an official detector/key.

**Tier 3 - pixel-domain image watermark support (optional/heavy):**
- [ ] Support optional reverse-SynthID scoring as an external, non-bundled detection backend.
- [ ] Support optional CtrlRegen/noai-watermark removal as an external, non-bundled regeneration backend.
- [ ] Keep reverse-SynthID and CtrlRegen outside the AskRex repository and outside the default Rex install because of their separate licensing and heavy dependencies.
- [ ] Default CtrlRegen strength to `0.25`; expose documented presets only with clear regeneration/quality tradeoffs.
- [ ] Report before/after scoring when available, while clearly identifying reverse-SynthID as unofficial and best-effort.
- [ ] Treat GPU/model/Hugging Face/backend absence as an optional-component state, not a Rex failure.
- [ ] Do not download the ~10 GB Tier 3 model payload until the user installs/enables that optional component.

**Security, supply-chain, and licensing:**
- [ ] Pin and review upstream revisions before updating the integration.
- [ ] Add compatibility tests before advancing the upstream pin.
- [ ] Keep API/Hugging Face tokens in Rex's credential/vault path or environment-only handoff; never put secrets on command-line arguments or in logs.
- [ ] Preserve the licensing boundary: watermarks-remover is MIT; reverse-SynthID remains external under its upstream non-commercial research terms; CtrlRegen/noai-watermark remains external and must not be vendored while it lacks a permissive license.
- [ ] Apply Rex path/symlink/output-boundary protections to all file mutations.

**Verification and tests:**
- [ ] Test skill/capability routing and unsupported formats.
- [ ] Test binary-vs-text refusal and original-file preservation.
- [ ] Test Unicode, C2PA, EXIF/XMP, PDF, DOCX, ODT, HTML/Markdown inspection/cleaning paths.
- [ ] Test post-clean verification and no-false-success behavior.
- [ ] Test rewrite backend failure/degradation and semantic-preservation constraints.
- [ ] Mock Tier 3 heavy model execution in normal CI; do not download multi-gigabyte models in standard GitHub Actions.
- [ ] Test missing GPU, missing external checkout, model-download failure, timeout, cancellation, symlink/path traversal, and permission denial.
- [ ] Add an explicit optional-component integration test procedure for a real GPU-equipped development machine.

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

Rex 2.0 capability principle:
New capabilities such as AI Provenance & Watermark must plug into the canonical Rex runtime/action lifecycle. A skill may provide discovery and natural-language intent metadata, but mutations must not bypass Rex authorization, verification, audit, cancellation, or user-scoped policy.
