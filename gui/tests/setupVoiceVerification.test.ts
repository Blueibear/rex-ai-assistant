import { describe, expect, it } from 'vitest'
import * as setupWizardModel from '../src/pages/setupWizardModel'
import type { TurnStatusUpdate, VoiceTranscriptEntry } from '../src/types/ipc'

type VerificationEvent =
  | { type: 'started' }
  | { type: 'voice_state'; state: string }
  | { type: 'voice_status'; status: string }
  | { type: 'transcript'; entry: VoiceTranscriptEntry }
  | { type: 'turn_status'; update: TurnStatusUpdate }
  | { type: 'voice_error'; error: string }
  | { type: 'cancelled' }
  | { type: 'playback_confirmed' }
  | { type: 'playback_rejected' }

type VerificationState = {
  status: string
  currentStage: string | null
  passedStages: string[]
  error?: string
}

type VerificationApi = {
  createVoiceVerificationState?: () => VerificationState
  reduceVoiceVerification?: (
    state: VerificationState,
    event: VerificationEvent
  ) => VerificationState
}

function getVerificationApi(): Required<VerificationApi> {
  const api = setupWizardModel as unknown as VerificationApi
  expect(api.createVoiceVerificationState).toBeTypeOf('function')
  expect(api.reduceVoiceVerification).toBeTypeOf('function')
  return api as Required<VerificationApi>
}

function advanceToTurn(
  reduce: Required<VerificationApi>['reduceVoiceVerification'],
  initialState: VerificationState
): VerificationState {
  let state = reduce(initialState, { type: 'started' })
  state = reduce(state, { type: 'voice_state', state: 'listening' })
  state = reduce(state, { type: 'voice_state', state: 'processing' })
  return reduce(state, {
    type: 'transcript',
    entry: { text: 'Rex, say setup verified', role: 'user', timestamp: 1 }
  })
}

describe('US-125 canonical voice verification state machine', () => {
  it('verifies only after canonical wake, capture, STT, turn, TTS, and audible-playback evidence', () => {
    const api = getVerificationApi()
    const reduce = api.reduceVoiceVerification
    let state = api.createVoiceVerificationState()

    state = reduce(state, { type: 'started' })
    expect(state).toMatchObject({ status: 'running', currentStage: 'wake' })

    state = reduce(state, { type: 'voice_state', state: 'listening' })
    expect(state).toMatchObject({ currentStage: 'capture' })

    state = reduce(state, { type: 'voice_state', state: 'processing' })
    expect(state).toMatchObject({ currentStage: 'stt' })

    state = reduce(state, {
      type: 'transcript',
      entry: { text: 'Rex, say setup verified', role: 'user', timestamp: 1 }
    })
    expect(state).toMatchObject({ currentStage: 'turn' })

    state = reduce(state, {
      type: 'turn_status',
      update: { turnId: 'turn-1', sequence: 3, status: 'done', terminal: true }
    })
    expect(state).toMatchObject({ currentStage: 'tts' })

    state = reduce(state, { type: 'voice_status', status: 'voice_playback_complete' })
    expect(state).toMatchObject({ status: 'running', currentStage: 'playback' })

    state = reduce(state, { type: 'playback_confirmed' })
    expect(state).toMatchObject({ status: 'verified', currentStage: null })
    expect(state.passedStages).toEqual(['wake', 'capture', 'stt', 'turn', 'tts', 'playback'])
  })

  it('fails at the active stage with the voice bridge error intact', () => {
    const api = getVerificationApi()
    let state = api.createVoiceVerificationState()
    state = api.reduceVoiceVerification(state, { type: 'started' })
    state = api.reduceVoiceVerification(state, { type: 'voice_state', state: 'listening' })
    state = api.reduceVoiceVerification(state, { type: 'voice_state', state: 'processing' })

    state = api.reduceVoiceVerification(state, {
      type: 'voice_error',
      error: 'Speech recognition failed.'
    })

    expect(state).toMatchObject({
      status: 'failed',
      currentStage: 'stt',
      error: 'Speech recognition failed.'
    })
  })

  it('fails specifically when the canonical Rex turn terminates unsuccessfully', () => {
    const api = getVerificationApi()
    let state = advanceToTurn(api.reduceVoiceVerification, api.createVoiceVerificationState())

    state = api.reduceVoiceVerification(state, {
      type: 'turn_status',
      update: { turnId: 'turn-1', sequence: 4, status: 'error', terminal: true }
    })

    expect(state).toMatchObject({
      status: 'failed',
      currentStage: 'turn',
      error: 'Rex turn failed.'
    })
  })

  it('requires explicit audible-playback confirmation and reports rejection as playback failure', () => {
    const api = getVerificationApi()
    let state = advanceToTurn(api.reduceVoiceVerification, api.createVoiceVerificationState())
    state = api.reduceVoiceVerification(state, {
      type: 'turn_status',
      update: { turnId: 'turn-1', sequence: 3, status: 'done', terminal: true }
    })
    state = api.reduceVoiceVerification(state, {
      type: 'voice_status',
      status: 'voice_playback_complete'
    })

    expect(state).toMatchObject({ status: 'running', currentStage: 'playback' })
    state = api.reduceVoiceVerification(state, { type: 'playback_rejected' })
    expect(state).toMatchObject({
      status: 'failed',
      currentStage: 'playback',
      error: 'Audible playback was not confirmed.'
    })
  })

  it('can be cancelled and retried without carrying stale progress forward', () => {
    const api = getVerificationApi()
    let state = api.createVoiceVerificationState()
    state = api.reduceVoiceVerification(state, { type: 'started' })
    state = api.reduceVoiceVerification(state, { type: 'voice_state', state: 'listening' })
    state = api.reduceVoiceVerification(state, { type: 'cancelled' })

    expect(state).toMatchObject({
      status: 'cancelled',
      currentStage: 'capture',
      passedStages: ['wake']
    })

    state = api.reduceVoiceVerification(state, { type: 'started' })
    expect(state).toMatchObject({
      status: 'running',
      currentStage: 'wake',
      passedStages: []
    })
    expect(state.error).toBeUndefined()
  })

  it('ignores out-of-order evidence so saved configuration cannot impersonate verification', () => {
    const api = getVerificationApi()
    let state = api.createVoiceVerificationState()
    state = api.reduceVoiceVerification(state, { type: 'started' })
    state = api.reduceVoiceVerification(state, { type: 'playback_confirmed' })
    state = api.reduceVoiceVerification(state, {
      type: 'voice_status',
      status: 'voice_playback_complete'
    })
    state = api.reduceVoiceVerification(state, {
      type: 'turn_status',
      update: { turnId: 'turn-1', sequence: 3, status: 'done', terminal: true }
    })

    expect(state).toMatchObject({ status: 'running', currentStage: 'wake', passedStages: [] })
  })
})
