# AskRex End-User Installation and Always-On Voice Runtime

## Status and purpose

This document defines the final consumer installation and household voice-runtime contract for AskRex Assistant.

It describes the product experience that implementation and release-readiness work must converge on. It does not claim that every item below is implemented today. Current implementation status remains governed by `PRD-production-readiness.md` and verified code/tests.

The final product is not merely a desktop chat application with a voice button. AskRex is intended to function as an always-available household assistant that can be summoned by wake word without the desktop or mobile app being open.

## Non-negotiable product contract

A normal end user installs one AskRex application package. The user must not need to install Python, Node.js, Git, create a virtual environment, clone the repository, run terminal commands, or edit JSON to get the supported consumer experience.

Closing the Electron window must not stop Rex Core or the local voice-listening runtime. The desktop and mobile apps are interaction and control surfaces, not lifecycle owners for the assistant.

When always-on voice is enabled and healthy, Rex remains available after GUI close and normal reboot/sign-in so a user can say the configured wake word and complete a full spoken interaction without opening an app.

OpenClaw and ClawHub remain optional external capability providers. They are never required for core AskRex startup, wake-word listening, local voice interaction, identity, permissions, memory, Home Assistant control, or truthful result verification.

## Runtime roles

### Rex Core

Rex Core is the authoritative assistant runtime on the primary household machine. It owns orchestration, identity, permissions, memory, context, model routing, tools, integrations, scheduling, verification, and final response generation.

Rex Core must be able to run independently of any visible Electron window. On Windows, installation should use the appropriate persistent background mechanism so Core survives reboot without requiring a terminal session.

### Local Voice Agent

The local Voice Agent is the per-user background audio component for the machine on which AskRex is installed. It owns access to the interactive user's microphone and speaker, wake-word detection, capture handoff, local voice status, and playback coordination.

The Voice Agent must start automatically when enabled and the user signs in. It must remain available while the Electron window is closed.

The implementation may use separate processes or service boundaries, but the end user experiences one AskRex installation and one coherent status/control surface.

### Rex Room endpoint

A Rex Room endpoint is a lightweight trusted satellite used in an additional room. It is not a second independent Rex brain.

A room endpoint may support:

- input and output: full voice endpoint;
- output only: Rex can speak/play there but cannot listen there;
- input only: capture endpoint with response routed elsewhere when explicitly configured.

## Consumer installation flow

The supported Windows consumer path is a packaged installer such as `AskRex-Setup.exe`.

The installer must bundle or provision the managed runtime required by the supported AskRex experience. A clean supported Windows machine must not require a preinstalled system Python or Node.js runtime.

The installer/setup flow should offer these user-facing choices without exposing developer mechanics:

1. Install AskRex.
2. Optionally start AskRex automatically with Windows.
3. Launch first-run setup.
4. Configure the primary user and supported AI provider.
5. Choose and preview Rex's TTS voice.
6. Test the microphone and speaker.
7. Choose the wake word and calibrate it from normal speaking positions.
8. Assign the local listening endpoint to a room.
9. Explain and enable background voice availability.
10. Optionally connect Home Assistant and other integrations.
11. Optionally add household users and additional room endpoints.
12. Perform a real screenless voice test before declaring voice setup complete.

The wizard should not require integrations that are not needed for basic Rex conversation. Optional integrations must degrade gracefully when absent.

## Setup verification rule

Writing settings successfully is not proof that voice setup works.

Voice setup is verified only after the configured path completes wake detection, audio capture, STT, the canonical Assistant/TurnEngine response path, TTS generation, and audible playback through the selected output.

If any stage cannot be verified, the wizard reports the specific degraded or blocked stage and keeps text/app interaction available where possible.

## Background lifecycle and user controls

When always-on voice is enabled, the GUI may be minimized or closed without stopping Rex Core or the Voice Agent.

The product must provide clear controls for:

- Start Rex automatically with Windows.
- Enable or disable wake-word listening for the signed-in user.
- Pause Listening immediately from the tray/control surface.
- Resume Listening explicitly.
- Show the current microphone, speaker, wake-word, Core, and endpoint health state.
- Select replacement audio devices when configured hardware disappears.

The system must never hide an audio failure behind a healthy-looking status. A missing microphone may degrade voice input while leaving text/mobile interaction available. A disconnected Home Assistant instance may degrade smart-home control while leaving ordinary conversation available.

## Multi-room discovery and pairing

Additional room endpoints must pair to one authoritative Rex Core through an authenticated, revocable local trust relationship. Pairing must not grant broader user or household permissions than the authenticated user already has.

Setup must capability-test each discovered endpoint rather than infer capability from branding or the physical presence of a microphone.

For example, a smart speaker may be a valid output target while its microphone is inaccessible to third-party software. In that case AskRex must label it output-only rather than presenting it as a full Rex Room endpoint.

Each paired endpoint records a stable device identity, room assignment, available input/output capabilities, current health, and authorization state.

OpenClaw may contribute optional device/tool capabilities, but pairing, identity, permission, room authority, and result verification remain Rex responsibilities.

## Room context and Home Assistant

The endpoint that hears a request contributes trusted request-origin context. Room/device origin does not itself grant permission, but it may resolve ordinary natural-language references when the mapping is unambiguous.

Example:

- Input endpoint: `Bedroom Rex`
- Assigned room: `Bedroom`
- User request: `turn the light off`
- Candidate Home Assistant target: authorized bedroom light entity

Rex must still use the canonical Home Assistant/action lifecycle and verify the resulting device state before reporting success.

The setup experience should help the user review Home Assistant areas, rooms, aliases, and device mappings instead of forcing manual entity-ID editing. Existing canonical media/output-routing/context services must be reused rather than creating a second room or speaker registry.

By default, an interactive spoken response should return through the authorized endpoint that heard the request when that endpoint supports output, unless the user explicitly names another destination or a current per-user routing rule applies.

## Household identity

Speaker identity and room identity are separate context dimensions.

A voice turn should resolve, when available:

- the authenticated or recognized Rex user;
- the trusted listening endpoint;
- the assigned room;
- the user's permissions and privacy/context policy;
- the requested response/output mode.

Unknown or ambiguous speaker identity must fail closed for private or permission-sensitive operations. Household/admin status does not silently substitute for another person's identity or private authority.

## Privacy and always-listening behavior

Always-on voice must be understandable and controllable by the user.

Wake-word detection should remain local to the endpoint when the supported local detector can do so. Audio capture for a request begins only after the configured activation path is satisfied, subject to implementation-specific buffering needed for reliable wake-word recognition.

The product must provide a clear visible status for listening state and an immediate Pause Listening control. The user must be able to disable wake-word startup without uninstalling AskRex.

Logs, telemetry, and endpoint health records must not contain raw microphone audio, transcripts, private memory, credentials, or user identity unless a separate documented feature explicitly requires and authorizes that data.

## Failure and recovery behavior

A single component failure must not unnecessarily terminate the rest of AskRex.

Examples:

- Microphone unavailable: voice input is degraded; text/mobile remain available.
- Speaker unavailable: Rex may show the response on an available screen but must not claim it was spoken.
- Room endpoint offline: that room becomes unavailable; Core and other endpoints continue.
- Home Assistant unavailable: smart-home actions fail closed with a useful explanation; general assistant use continues.
- OpenClaw unavailable: optional external tools are unavailable; native Rex capabilities continue.
- Model/provider failure: use only configured, permitted fallback providers and report failure honestly if no valid fallback exists.

Watchdog/restart behavior must not create restart loops that hide an underlying configuration or hardware problem.

## Final consumer acceptance gate

AskRex is not finished as a household voice assistant if normal voice use requires the Electron window or mobile app to remain open.

Before final consumer release, a clean supported Windows machine must prove all of the following with documented evidence:

1. Install AskRex using the packaged installer with no preinstalled Python, Node.js, Git, or repository checkout.
2. Complete the supported first-run wizard without terminal or manual JSON steps.
3. Verify microphone, speaker, TTS voice, wake word, and the local room assignment.
4. Enable background voice operation and automatic startup.
5. Close the Electron window completely.
6. Say the wake word and complete a spoken question/answer round trip.
7. Reboot/sign in without manually starting developer commands.
8. Repeat the wake-word spoken interaction successfully.
9. If Home Assistant is configured, issue a low-risk room-context command such as turning off a room light and independently verify the final entity state.
10. Demonstrate Pause Listening and Resume Listening.
11. Demonstrate truthful degraded behavior when a voice component is unavailable.
12. Confirm the same Core continues operating when optional OpenClaw connectivity is unavailable.

Multi-room release readiness additionally requires at least one securely paired non-Core room endpoint to pass its declared capability tests and complete a screenless wake-word interaction from its assigned room.

## Current-status warning

This document defines the final target. It does not promote today's beta wake-word path to production by documentation alone.

Until the physical-audio reliability, background-lifecycle, clean-install, reboot, privacy-control, and screenless acceptance gates in `PRD-production-readiness.md` pass, current user-facing surfaces must continue to describe wake-word/always-on behavior with their verified status.
