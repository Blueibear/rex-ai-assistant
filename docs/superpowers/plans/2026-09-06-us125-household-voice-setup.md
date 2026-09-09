# US-125 Consumer Household Voice Setup Plan

## Goal
Extend the existing Electron first-run wizard into the canonical consumer household-voice setup flow without creating parallel identity, audio, wake-word, room, or runtime authority.

## Required flow
1. Primary user/profile and supported AI provider.
2. Rex voice selection plus preview.
3. Microphone selection plus functional test.
4. Speaker/audio target selection plus functional test.
5. Wake-word selection/calibration.
6. Local room assignment.
7. Explicit background-voice choice with privacy explanation before enablement.
8. Optional Home Assistant/additional household extensions.
9. Screenless voice verification state that is distinct from configuration persistence.

## Reuse boundaries
- Keep `SetupWizardPage.tsx` as the first-run surface and extend `setupWizardModel.ts`/typed IPC rather than adding another wizard.
- Reuse preload APIs for voice preview/test, audio targets, wake-word inventory/status/sample/training, profile, and existing background runtime controls.
- Reuse the canonical voice bridge/TurnEngine path for verification; do not create a direct LLM or parallel voice pipeline.
- Keep Home Assistant and additional users/rooms optional.

## Truthful verification
Saving configuration is `configured`, not `verified`. Voice can become `verified` only after the full canonical wake detection -> capture -> STT -> Assistant/TurnEngine -> TTS -> audible playback path succeeds. Automated CI may prove orchestration and failure-state behavior with controlled fakes; physical audible verification remains a US-130 release-evidence boundary.

## TDD sequence
1. Add RED model/UI contracts for required stages, persisted choices, privacy-before-enable, optional extensions, and saved-vs-verified truth.
2. Extend typed setup state/payload/bridge persistence minimally until those contracts are GREEN.
3. Add RED orchestration tests for per-stage success/failure, cancellation/resume, and unaffected text usability.
4. Implement a bounded setup-verification state machine that composes existing IPC/voice/runtime primitives.
5. Add packaged first-run smoke coverage without pretending simulated audio is physical acceptance.
6. Run GUI typecheck/Vitest/ESLint/build, Python focused/full tests, security gates, and Windows packaged-artifact smoke; update `CLAUDE.md`, PRD, and progress evidence before merge.