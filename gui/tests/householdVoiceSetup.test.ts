import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'
import { buildSetupSubmission, type SetupFormData } from '../src/pages/setupWizardModel'

const pageSource = readFileSync(
  fileURLToPath(new URL('../src/pages/SetupWizardPage.tsx', import.meta.url)),
  'utf8'
)
const ipcSource = readFileSync(
  fileURLToPath(new URL('../src/types/ipc.ts', import.meta.url)),
  'utf8'
)

describe('US-125 household voice first-run contracts', () => {
  it('persists canonical PortAudio and local-room choices with setup submission', () => {
    const extended = {
      username: 'alice',
      password: 'pass1234', // pragma: allowlist secret
      llmProvider: 'local',
      llmApiKey: '',
      ttsProvider: 'edge',
      haBaseUrl: '',
      haToken: '',
      ttsVoiceId: 'en-US-AriaNeural',
      microphoneDeviceIndex: 2,
      speakerDeviceIndex: 4,
      localDeviceId: 'local_voice',
      wakeWordId: 'hey_rex',
      roomName: 'Office',
      backgroundVoiceEnabled: true
    } as unknown as SetupFormData

    const payload = buildSetupSubmission(extended) as unknown as Record<string, unknown>

    expect(payload.tts_voice_id).toBe('en-US-AriaNeural')
    expect(payload.microphone_device_index).toBe(2)
    expect(payload.speaker_device_index).toBe(4)
    expect(payload.local_device_id).toBe('local_voice')
    expect(payload.wake_word_id).toBe('hey_rex')
    expect(payload.room_name).toBe('Office')
    expect(payload.background_voice_enabled).toBe(true)
  })

  it('exposes every required core household voice stage in the existing wizard', () => {
    for (const requiredLabel of [
      'Choose Rex voice',
      'Test microphone',
      'Test speaker',
      'Wake word',
      'Room',
      'Background voice',
      'Verify voice'
    ]) {
      expect(pageSource).toContain(requiredLabel)
    }
  })

  it('explains persistent background listening before offering enablement', () => {
    const privacyCopy = pageSource.indexOf('continues listening when the AskRex window is closed')
    const backgroundChoice = pageSource.indexOf('Enable background voice')

    expect(privacyCopy).toBeGreaterThanOrEqual(0)
    expect(backgroundChoice).toBeGreaterThan(privacyCopy)
  })

  it('keeps configuration saved separate from truthful voice verification', () => {
    expect(pageSource).not.toContain('Rex is ready!')
    expect(pageSource).toContain('Voice setup saved')
    expect(pageSource).toContain('Voice not yet verified')
  })

  it('uses canonical typed PortAudio fields rather than browser device IDs', () => {
    for (const field of [
      'tts_voice_id',
      'microphone_device_index',
      'speaker_device_index',
      'local_device_id',
      'wake_word_id',
      'room_name',
      'background_voice_enabled'
    ]) {
      expect(ipcSource).toContain(field)
    }
    expect(ipcSource).not.toContain('microphone_device_id: string')
    expect(ipcSource).not.toContain('speaker_target_id: string')
  })

  it('keeps Home Assistant explicitly optional', () => {
    expect(pageSource).toContain('Home Assistant')
    expect(pageSource).toContain('Do this later')
  })
})
