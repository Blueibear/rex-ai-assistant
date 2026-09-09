import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'

const pageSource = readFileSync(
  fileURLToPath(new URL('../src/pages/SetupWizardPage.tsx', import.meta.url)),
  'utf8'
)

describe('US-125 setup voice verification UX', () => {
  it('drives verification through the canonical voice and turn-status IPC surfaces', () => {
    expect(pageSource).toContain('createVoiceVerificationState')
    expect(pageSource).toContain('reduceVoiceVerification')
    expect(pageSource).toContain('window.rex.startVoice(')
    expect(pageSource).toContain('window.rex.onTurnStatus(')
    expect(pageSource).toContain("type: 'voice_state'")
    expect(pageSource).toContain("type: 'transcript'")
    expect(pageSource).toContain("type: 'turn_status'")
    expect(pageSource).toContain("type: 'voice_status'")
  })

  it('shows stage-specific progress and failure without conflating saved config with verification', () => {
    expect(pageSource).toContain('Wake word detection')
    expect(pageSource).toContain('Microphone capture')
    expect(pageSource).toContain('Speech recognition')
    expect(pageSource).toContain('Canonical Rex turn')
    expect(pageSource).toContain('Speech synthesis')
    expect(pageSource).toContain('Audible playback')
    expect(pageSource).toContain('Voice verification failed')
    expect(pageSource).toContain('Voice setup saved')
  })

  it('requires the user to confirm that Rex was actually audible', () => {
    expect(pageSource).toContain("Did you hear Rex's reply?")
    expect(pageSource).toContain('Yes, I heard Rex')
    expect(pageSource).toContain("No, I didn't hear it")
    expect(pageSource).toContain("type: 'playback_confirmed'")
    expect(pageSource).toContain("type: 'playback_rejected'")
    expect(pageSource).toContain('Voice verified')
  })

  it('supports retry, cancellation, and finishing setup without a false verified state', () => {
    expect(pageSource).toContain('Retry voice verification')
    expect(pageSource).toContain('Cancel verification')
    expect(pageSource).toContain('Continue without voice')
    expect(pageSource).toContain("type: 'cancelled'")
    expect(pageSource).toContain('window.rex.stopVoice()')
  })
})
