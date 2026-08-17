# AskRex Assistant Controlled Self-Maintenance

## Status

Planned Rex 2.0 capability. The architecture is approved as a direction, but bounded autonomous self-maintenance is not part of the current production-readiness release-candidate definition.

## Purpose

Rex should be able to maintain and extend AskRex Assistant without becoming an unconstrained self-modifying process. The design treats self-maintenance as a normal software-development workflow with independent source-control protections, policy enforcement, validation, owner-gated authority changes, and rollback.

## Core principle

Rex may maintain Rex. Rex may not remove the independent constraints that govern Rex.

GitHub rulesets, required CI, Rex policy, credential boundaries, and owner approval remain separate enforcement points.

## Capability acquisition order

1. Existing Rex tool/capability.
2. Existing enabled local skill.
3. Approved OpenClaw/ClawHub capability.
4. Generated or extended local skill.
5. Core Rex source modification only when required.

The least invasive safe option wins.

## Existing foundations
The current repository already contains foundations that this feature must reuse:

- `rex/skills/registry.py`, `router.py`, `trainer.py`, and `loader.py`;
- `rex/vscode_service.py` for code patching and pytest execution;
- `rex/github_service.py` for issues, PRs, commits, and branch patch workflows;
- policy, audit, and tool-execution lifecycle components;
- `rex/openclaw/` HTTP and tool adapters;
- existing CI, security, typecheck, packaging, and smoke-test gates.

These are building blocks, not proof that autonomous self-maintenance is already complete.

## Self-maintenance workflow

```text
request / defect / capability gap
        |
        v
classify risk + resolve capability path
        |
        v
create isolated branch + worktree when code changes are required
        |
        v
reproduce / diagnose / root-cause
        |
        v
minimal patch + tests + required docs
        |
        v
targeted validation -> broader local gates
        |
        v
open/update PR
        |
        v
independent GitHub required checks
        |
        v
policy/owner approval if required
        |
        v
merge -> activate candidate -> post-activation verification
        |
        +--> pass: record verified running version
        |
        +--> fail: automatic rollback -> verify rollback
```

## GitHub maintainer identity

Use a dedicated GitHub App or equivalent machine identity installed only on explicitly approved repositories. Initial scope should be `Blueibear/AskRex-Assistant`.

Typical permissions may include issues, repository contents/branches, pull requests, and check/status visibility. Workflow-file mutation should be separately permissioned and owner-gated when it can weaken or replace required controls.

Rex must not be able to:
- delete the repository;
- increase its own permissions;
- install itself on additional repositories;
- remove or bypass protected-branch/ruleset requirements;
- force-push protected branches;
- mark independent CI checks as successful;
- silently weaken security gates to make its own PR pass.

## Approval classes

### Routine and potentially pre-approved

- read-only diagnosis;
- issue creation/triage;
- isolated maintenance branch/worktree creation;
- bounded code/doc/test changes;
- branch push and PR creation;
- responding to CI failures;
- merge after all gates pass when policy explicitly permits it.

### Owner approval required

- GitHub App permission or installation-scope changes;
- branch protection/ruleset changes;
- authentication and credential-vault changes that alter authority;
- self-maintenance policy changes;
- update/rollback or verification changes that weaken safeguards;
- removal or weakening of required tests, CI, secret scanning, or security gates;
- any change whose primary effect is to increase Rex's own authority.

## Constitutional privacy boundaries

Per-user privacy/context authority is part of the protected self-maintenance constitution, not an ordinary preference that Rex may rewrite for convenience.

Rex, generated skills, OpenClaw/ClawHub capabilities, and developer/self-repair agents may read and enforce the current policy state, but they must never autonomously widen:

- whether a source is eligible for broad/background contextual use;
- private-versus-household uploaded-document scope or disclosure audience;
- another user's private memory/context disclosure boundary;
- `location_assist`;
- person-specific `location_share`;
- any equivalent future permission whose effect is to expose or reuse one user's private information for another user or broader context.

Changes to these authority boundaries require the appropriate user/data-owner authorization at the real mutation boundary. Rex cannot approve its own proposal to widen them. Household or administrative status does not override another user's `location_assist` or `location_share` choices. A generated skill, OpenClaw capability, or maintenance agent inherits these same limits and cannot obtain broader authority through installation, code generation, configuration, or self-update.

Revocation must fail closed and invalidate affected active context/caches where the underlying subsystem supports revisions. A maintenance change that cannot prove it preserves these boundaries must remain blocked rather than deploy.

## Verification and truthfulness

Self-maintenance must use the same status discipline as tool execution:

- `proposed`
- `attempted`
- `completed`
- `verified`
- `failed`
- `blocked`
- `rolled_back`

A patch that applies is not a verified fix. A PR that merges is not a verified deployment. A restarted process is not healthy until its health and functional smoke checks pass.

## Rollout

1. Read-only diagnosis and recommendations.
2. Issue/PR creation with owner-reviewed changes.
3. Supervised isolated code changes.
4. Bounded autonomous routine fixes after repeated successful trials.
5. Optional bounded auto-merge only with independent required checks and proven rollback.

Authority-changing operations remain owner-gated at every stage.

## Source-of-truth links

- `PRD-production-readiness.md`, Section 13: implementation backlog after RC.
- `CLAUDE.md`: agent and maintainer safety rules.
- `docs/security/`: threat model and security controls when implemented.
- `REX_Unified_Build_Spec_UPDATED.md` and `REX_ACTIVE_CHECKLIST.md` in the project planning set: product-level architecture and checklist.
