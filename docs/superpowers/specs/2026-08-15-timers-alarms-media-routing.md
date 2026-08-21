# AskRex Timers, Alarms, Speaker Routing, and Media Orchestration

**Date:** 2026-08-15
**Status:** Mandatory production-readiness addendum
**Authority:** This addendum is part of the active AskRex production-readiness workstream and must be completed before the final release gate. It supplements `PRD-production-readiness.md` and `docs/superpowers/plans/2026-08-08-rex2-production-readiness-integration.md`. The 2026-08-16 situational-context/media/privacy design further refines US-121/US-122 and adds the US-123 context/proactivity boundary.

## Goal

AskRex must provide the everyday time and audio capabilities expected from a household assistant. Timers, alarms, spoken responses, notifications, and music must be routable to specific speakers, rooms, devices, or user-defined speaker groups. Routing must be explicit, inspectable, per-user aware, and verification-first.

Rex must not treat a reminder as a timer. Timers need accurate countdown semantics. Alarms need clock-time and recurrence semantics. Media routing needs a provider-neutral target model so the same user command can work with local speakers, Home Assistant `media_player` entities, and future OpenClaw or direct smart-speaker adapters without changing the user-facing command model.

## Required user experiences

Examples that must work after implementation:

- "Rex, set a 10-minute timer."
- "Set a 20-minute laundry timer and a 5-minute pasta timer."
- "How much time is left on my timers?"
- "Add five minutes to the pasta timer."
- "Cancel the laundry timer."
- "Set an alarm for 7:00 tomorrow morning."
- "Wake me at 7:00 every weekday."
- "Snooze that alarm for 10 minutes."
- "Set my morning alarm for 7:00 and only play it on the bedroom speaker."
- "Set Cole's kitchen timer for 15 minutes and ring it on the kitchen speaker."
- "Play music in the living room."
- "Play this on the downstairs speakers."
- "Create a speaker group called downstairs with the living room and kitchen speakers."
- "Add the office speaker to downstairs."
- "Set the downstairs group to 35 percent."
- "Move the music from the kitchen to downstairs."
- "Pause the bedroom speaker."
- "What is playing in the living room?"

## Design principles

1. **One canonical audio-target model.** A target may be a local audio device, smart speaker, room, Home Assistant media player, or user-defined group.
2. **Provider-neutral orchestration.** Rex owns intent, identity, permissions, routing, policy, and verification. Home Assistant, OpenClaw, Sonos/Bose adapters, or other providers supply device-specific execution.
3. **Per-user ownership.** Timers, alarms, routing defaults, and preferences must distinguish James and Cole.
4. **Explicit override beats defaults.** "Only on the bedroom speaker" must override a user's normal alarm route for that alarm.
5. **Verification before success.** Rex may say an alarm, route, group, or media action is complete only after the canonical action lifecycle has verified the resulting state where verification is technically possible.
6. **Fail clearly.** If a requested target is offline or unavailable, Rex must say so and follow the configured fallback policy instead of silently choosing another speaker.
7. **No minute-granularity timer polling.** Timer expiration must not depend on the existing reminder service's roughly minute-level polling behavior.

---

## US-120: First-class concurrent timers and alarms

**Priority:** P1 release requirement
**Workstream:** Scheduling / Household Assistant

**Description:** As a user, I want Rex to create, manage, persist, and accurately fire multiple timers and alarms so basic household timekeeping works without relying on reminders.

### Acceptance criteria

- [ ] A canonical timer/alarm service exists with separate timer and alarm semantics.
- [ ] Multiple concurrent timers are supported per user with unique IDs and optional human-readable names.
- [ ] Natural-language timer creation supports durations in seconds, minutes, and hours.
- [ ] Users can list, query remaining time, cancel, pause, resume, rename, and add/subtract time from timers.
- [ ] Alarm creation supports absolute local clock times, dates, and recurring schedules such as weekdays or selected days.
- [ ] Users can list, enable/disable, edit, cancel, snooze, and dismiss alarms.
- [ ] Timezone and daylight-saving transitions are handled using the user's canonical timezone context.
- [ ] Timers and alarms persist across Rex restarts. On restart, elapsed timers/alarms are reconciled truthfully rather than silently discarded.
- [ ] Timer expiration does not depend on reminder-service minute polling. Expiration scheduling is precise enough for ordinary household use and has deterministic timing tests with an explicit tolerance.
- [ ] Timer/alarm ownership is isolated per user. James and Cole can have timers or alarms with the same display name without collisions.
- [ ] Timer/alarm actions are exposed through the canonical Capability Registry and TurnEngine/action lifecycle rather than a one-off bypass.
- [ ] Voice, typed chat, Electron, and mobile/PWA surfaces can create and manage the same underlying timers/alarms.
- [ ] Unit and integration tests cover concurrent timers, restart recovery, recurring alarms, snooze, cancellation, identity isolation, and timezone/DST cases.
- [ ] Documentation and capability discovery accurately report timer/alarm support only after the feature is verified.

---

## US-121: Canonical speaker, room, group, and media orchestration

**Priority:** P1 release requirement
**Workstream:** Audio / Media / Home Assistant / External Capabilities

**Description:** As a user, I want Rex to understand speakers, rooms, and speaker groups and to route music and audio commands to the target I name.

### Acceptance criteria

- [x] A canonical audio target registry exists with stable IDs, display names, provider, room, capabilities, online/health state, and user-visible aliases.
- [x] The registry can represent local playback devices, Home Assistant `media_player` entities, and future OpenClaw/direct smart-speaker providers through adapters rather than provider-specific user commands.
- [x] Rex has a user-bound media-provider/account abstraction whose credentials stay in the vault; Apple Music/MusicKit can plug into that abstraction when Apple developer credentials and per-user authorization are available.
- [x] Rex can resolve natural-language targets such as "bedroom speaker", "kitchen", "downstairs", or a named device without unsafe fuzzy ambiguity.
- [x] When no media target is named, the trusted request-origin/listening endpoint is the preferred authorized playback target.
- [x] Users can create, rename, inspect, modify, and delete persistent speaker groups.
- [x] Groups can contain multiple compatible speakers and can be addressed as one target.
- [x] Rex supports provider-appropriate media actions including play, pause, resume, stop, next/previous where available, volume, mute/unmute, and playback-state query.
- [x] Rex can start or route requested music/media to one speaker, one room, or one group when the configured provider supports the requested source.
- [x] Successful playback establishes bounded active-media context so conversational follow-ups such as "pause it", "turn it up", and "move it to the living room" work when unambiguous.
- [x] Rex can move or re-target active playback when the provider supports transfer; otherwise it explains the limitation instead of claiming success.
- [x] Capability differences are explicit. Unsupported actions on a target return a truthful, user-actionable result.
- [x] Group and playback mutations use the canonical action lifecycle and verify resulting device/group state where technically possible.
- [x] Per-user permissions can restrict which devices/groups a user may control and which media provider account may be used.
- [x] Audio target discovery and health refresh without requiring a full Rex restart where the underlying provider supports dynamic discovery.
- [x] Tests cover target resolution, ambiguous names, request-origin defaulting, active-session follow-ups, provider-account isolation, group CRUD, offline devices, mixed provider capability differences, playback control, permissions, and verified outcomes.

**US-121 local evidence:** persistent groups are CRUD/addressable targets and their mutations are lifecycle-verified, but they intentionally advertise no direct playback controls until a reviewed group execution adapter exists. Home Assistant is the verified control adapter; LAN Sonos/Bose targets are discovery-visible only. Apple Music is provider/account metadata only. Physical-speaker playback remains a release-gate validation item below.

---

## US-122: Output routing policies and Settings UI

**Priority:** P1 release requirement
**Workstream:** Settings / Voice / Timers / Media

**Description:** As a user, I want to decide where Rex sends alarms, timers, spoken responses, and media by default, with per-event and time-of-day rules that can be overridden in a command.

**Implementation status (2026-08-21):** The US-122 product behavior is implemented and locally validated: shared per-user policy persistence/resolution, Electron and authenticated-mobile settings surfaces, timer/alarm explicit targets and due-event routing, request-origin media routing, media-account isolation/fallback policy, conditional rules, quiet hours, target volume/fallback handling, route explanation, privacy-safe decision audit, and Settings speaker-group/test-playback controls. Media-account linking itself is the canonical US-121 `MediaAccountStore.put()` registration seam backed by a per-user credential-vault reference; US-122 lists and selects those already-linked accounts and intentionally does not invent a fake Apple Music token/OAuth flow before real provider credentials exist. Physical-speaker/provider production verification remains governed by the release-gate requirement below.

### Acceptance criteria

- [x] Settings exposes audio outputs, rooms, and speaker groups using the canonical audio target registry.
- [x] Users can set a default spoken-response target, default timer target, default alarm target, and default media target.
- [x] Each profile can link/select its own media-provider account(s) and default provider/account without exposing another user's credentials or library authority.
- [x] High-confidence voice identity selects that user's linked/default media account; unresolved identity may use a configured household primary playback account for ordinary playback only under policy.
- [x] For interactive media, the authorized request-origin/listening endpoint is the normal default target when no room/device/group is named.
- [x] Defaults can be configured per user rather than globally only.
- [x] A timer or alarm may store an explicit output target that overrides defaults, for example "only sound this alarm on the bedroom speaker."
- [x] Routing policies can include time/day conditions, for example morning alarms on the bedroom speaker and evening timers on the kitchen speaker.
- [x] Users can configure fallback behavior for unavailable targets: no fallback, named fallback target/group, or ask before rerouting.
- [x] Quiet-hours behavior is configurable and does not silently suppress safety-critical or explicitly overridden events.
- [x] Timer/alarm routes can specify target volume where supported without permanently changing a device's normal volume unless configured to do so.
- [x] Speaker-group management is available from Settings with create, rename, membership edit, test playback, and delete operations.
- [x] A one-off natural-language target always overrides a stored default for that action.
- [x] Rex can answer "where will my morning alarm play?" and explain the resolved route and policy that produced it.
- [x] Routing decisions and fallbacks are recorded in structured logs without exposing sensitive user content.
- [x] Electron and mobile/PWA surfaces use the same routing/policy backend and do not maintain separate conflicting settings.
- [x] Tests cover per-user provider-account isolation, unresolved-speaker primary-account fallback, request-origin media defaulting, per-user outputs, time-of-day routing, explicit overrides, target outage fallback, quiet hours, group routing, and concurrent James/Cole policies.

---

## Release-gate requirement

US-120, US-121, and US-122 are mandatory before AskRex is considered feature-complete for household-assistant use. The final production RexBench/release gate must include deterministic coverage for timer accuracy and concurrency, alarm recurrence/snooze/restart recovery, audio-target resolution, group routing, per-user routing isolation, and unavailable-target behavior. Physical-device verification is required for at least one real speaker/media path before claiming that path production-verified.
