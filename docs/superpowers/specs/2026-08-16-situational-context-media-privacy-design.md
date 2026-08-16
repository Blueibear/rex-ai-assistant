# Situational Context, Media Identity, and Privacy Design

**Date:** 2026-08-16
**Status:** Approved conversational requirements captured for implementation planning
**Scope:** US-086 amendments, US-121, US-122, and new situational-context/proactivity work

## Goal

Rex should feel aware of the current situation without making users operate separate technical subsystems. It should preserve conversational references, select the correct user/provider account, route output naturally, use connected information when useful, and make high-signal proactive suggestions.

The design must preserve hard privacy boundaries. Access to data, permission to include data in broad context, permission to disclose data to another user, and permission to perform an action are separate decisions.

## Design choice

Use one canonical contextual-state and source-policy layer with adapters from media, calendar, memory/uploads, Home Assistant, weather/traffic/search, and future providers.

Rejected alternatives:
- Feature-specific context flags: simpler initially, but duplicates identity/privacy logic and causes inconsistent behavior.
- One global "personalization on/off" switch: easy to explain, but too coarse for uploads, location, and cross-user disclosure.

The canonical layer is the recommended approach because it keeps authority decisions centralized while allowing each capability to expose only the state it owns.

## Core invariants

- Context access never grants action authority.
- A source may be available to Rex without being eligible for broad/background context.
- User-private information never becomes household context implicitly.
- Household administration never overrides another user's personal location/privacy decision.
- Ambiguous identity, target, or active-reference resolution must not silently widen authority.
- Rex should explain uncertainty or ask a short clarification when evidence is insufficient.

## Context-source policy

Each contextual source has explicit policy metadata rather than relying on prompt convention. At minimum the policy records:

- source type and stable source ID
- owner user ID when private
- audience scope: private user or household
- `context_enabled`: whether the source may participate in broad/background reasoning
- disclosure policy: who may receive information derived from the source
- source revision for safe context-cache invalidation
- optional retention/expiry metadata where appropriate

For ordinary connected integrations, connecting the source is treated as permission for Rex to use it contextually for that user unless the user disables contextual use. This applies to sources such as calendar data that the user deliberately connects to Rex.

Two source classes are exceptions and require additional explicit decisions: uploaded information and location.

### Uploaded information

Every upload must ask two independent questions at upload time:

1. Should Rex include this source in broader/background context when relevant?
2. Is the source private to the uploading user or shared household context?

The uploader can later change these settings. Another user cannot promote a private upload to household scope. If `context_enabled` is false, authorized users may still ask Rex to use the file directly, but it must not silently influence unrelated turns, proactive suggestions, or situational reasoning.

Retrieval filters owner/audience scope before ranking. Derived facts retain source provenance so private uploaded information cannot leak through summaries, proactive suggestions, or another user's prompt.

### Location policy

Location is opt-in per user. Rex must not enable, infer, or activate location tracking because a household administrator requested it.

Location permissions are split into separate grants:

- `location_assist`: Rex may use the user's location privately when needed to help that user, for example commute traffic, nearby results, weather, travel time, or arrival-aware assistance.
- `location_share`: Rex may disclose that user's current/recent location to explicitly named other users. This is a separate, person-specific grant.

Enabling `location_assist` never implies `location_share`. If Cole enables location-assisted commute help but has not granted location sharing to James, a request such as "Where is Cole?" must not reveal Cole's location or confirm whether Rex currently has location data.

A household administrator cannot override another user's location grants. Location should be accessed only when it materially improves the current task or an enabled proactive rule, not continuously merely because permission exists.

## Situational context model

Rex should maintain a bounded, user-scoped view of relevant current state rather than injecting every connected datum into every prompt. Capability adapters publish typed, expiring context references such as:

- current/active media session
- most recently addressed device or room
- active timers/alarms
- upcoming calendar commitments
- current conversational entities and references
- recent verified actions and their targets
- relevant environmental/current-information observations

TurnEngine resolves phrases such as "it", "that one", "move it to the living room", or "turn it up" against those typed references. Capabilities own their domain state; the conversation layer owns generic reference resolution. No capability gets its own separate conversation engine.

## Media identity and account selection

Media source identity and playback target are separate decisions.

Each Rex user may link provider accounts under that user's profile. Credentials/tokens remain in the credential vault and are bound to the owning user/provider/account slot. Provider adapters expose capabilities without exposing credentials to prompts.

For Apple Music, the architecture must support a first-class MusicKit/Apple Music provider when Apple developer credentials are available. US-121 does not require purchasing Apple membership or claiming live Apple Music verification before those credentials exist.

Account resolution for a media request is:

1. High-confidence identified speaker: use that user's linked/default provider account.
2. Explicit account/user override: use it only if current authorization permits it.
3. Unresolved speaker: use the configured household primary playback account for ordinary playback if policy allows.
4. Account-mutating actions still use canonical identity/authorization/confirmation policy and may not silently borrow another user's private library authority.

This prevents one user's listening history, favorites, or playlists from being changed through another user's account merely because they share speakers.

## Media target resolution

For an interactive media request with no explicit output target, the default target is the trusted request-origin/listening endpoint when that endpoint is a playable audio target.

Examples include the Rex speaker/microphone endpoint that heard the command, the current phone endpoint, or the current desktop audio endpoint. Device/session proximity never grants control authority; the origin is only a routing preference after authorization.

An explicit natural-language target such as "living room", "bedroom speaker", or a persistent speaker group overrides the request-origin default. Unsupported or ambiguous targets fail truthfully rather than guessing.

### Active media session

Successful playback creates or refreshes a short-lived active media-session reference containing only the state needed for conversational continuity: owner/account reference, source/provider, playback target, media item/session ID where available, capability state, and expiry/revision metadata.

Follow-ups such as "move it to the living room", "turn it up", "pause it", "add this to favorites", or "play something like this" resolve against that session when unambiguous. If multiple plausible sessions exist, Rex asks one short clarification rather than selecting silently.

## Proactive assistance

Rex should proactively identify useful next actions from context rather than merely report isolated facts. Examples include:

- after a calendar event such as an interview, naturally asking how it went on a later interaction
- offering to save a birthday mentioned in conversation when no birthday record exists
- warning about a major commute delay and suggesting a better route or departure time
- combining weather with a known commute or appointment when conditions materially affect the plan
- offering a follow-up reminder or action when the next step is obvious from the conversation

Proactivity uses a canonical opportunity evaluator, not independent feature-specific nags. Candidate opportunities include provenance, affected user, freshness, confidence, urgency, expected benefit, prior dismissals/preferences, and the action Rex could offer.

The evaluator should surface only high-signal opportunities. It should prefer a concise, personable "by the way" suggestion during a natural interaction unless the issue is urgent enough for an enabled notification channel.

Declined suggestion patterns should reduce future frequency. Accepted patterns may become explicit automations/preferences, but a suggestion itself never grants execution authority.

## Natural response behavior

Situational reasoning is an internal mechanism, not user-facing jargon. Responses should describe the useful outcome in ordinary language and should not expose policy IDs, confidence scores, context-cache terms, or provider plumbing unless the user explicitly asks for technical detail.

Rex may explain why it made a suggestion when asked, using source-level reasoning such as "Your calendar shows the interview ended an hour ago" or "Traffic on your usual route is unusually slow." It must not invent unseen evidence.

## Error and uncertainty behavior

- Missing/stale current information: fetch or refresh from the authoritative provider when the request requires current truth.
- Ambiguous identity: use only configured safe fallbacks; never broaden private authority.
- Ambiguous contextual reference: ask a concise clarification.
- Offline/unsupported media target: explain the limitation and offer an authorized alternative if one is obvious.
- Provider mutation with uncertain outcome: retain canonical attempted/unverified semantics until independently verified.
- Revoked source/context/location permission: invalidate affected contextual state and cached artifacts immediately through revision changes.

## Testing requirements

Tests must cover at minimum:

- per-user provider-account isolation and unresolved-speaker primary-account fallback
- request-origin media default plus explicit room/group override
- active media-session follow-ups and ambiguous-session clarification
- upload `context_enabled` on/off independently from private/household scope
- prevention of uploader-private content leaking into another user's context
- explicit location-assist opt-in and separate person-specific location-sharing grants
- proof that household admin cannot override another user's location policy
- location non-disclosure responses that do not confirm whether tracking data exists
- contextual reference expiry/revocation and context-cache revision invalidation
- proactive opportunity ranking, dismissal suppression, and cross-user isolation
- action authorization remaining independent from contextual access

## Story mapping and implementation order

1. **US-086 amendment:** add independent per-upload context inclusion plus private/household audience policy to the existing scoped upload/indexing story.
2. **US-121:** implement provider-neutral audio targets, media provider/account abstraction, playback controls, verified target state, active media-session state, and request-origin default routing behavior needed by media.
3. **US-122:** implement per-user output/account routing policy and Settings surfaces, including household primary playback-account fallback and explicit target overrides.
4. **US-123 (new active story):** implement canonical situational-context/source-policy and proactive-opportunity layers, including location privacy rules and generalized cross-capability conversational references.
5. **US-118 release gate:** add deterministic privacy/context/media/proactivity coverage before feature-complete claims.

US-123 intentionally composes existing calendar, context builder/cache, suggestion engine, identity, permissions, current-info tools, and capability state. It must not replace those subsystems or build a second memory store.

## Non-goals for the first implementation slices

- Continuous GPS collection when no enabled feature needs it.
- Inferring location permission from IP addresses, device presence, calendar locations, or household-admin status.
- Automatically sharing one user's private provider account, uploaded content, or location with another user.
- Building provider-specific conversational command grammars.
- Claiming Apple Music live integration until Apple developer credentials, user authorization, and real-device verification exist.
- Making proactive suggestions execute sensitive actions without normal action authorization/confirmation.

## Acceptance definition

The design is successful when a user can speak naturally, Rex can use the right authorized context without cross-user leakage, media follows the speaker/account/output rules above, follow-up references work without needless repetition, and Rex can offer timely assistance from connected information while preserving explicit location and upload privacy controls.
