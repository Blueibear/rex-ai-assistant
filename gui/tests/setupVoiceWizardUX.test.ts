import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'

const pageSource = readFileSync(
  fileURLToPath(new URL('../src/pages/SetupWizardPage.tsx', import.meta.url)),
  'utf8'
)

describe('US-125 setup Rex voice UX', () => {
  it('loads real voice inventory for the selected setup provider', () => {
    expect(pageSource).toContain('voiceApiProvider')
    expect(pageSource).toContain("return 'edge-tts'")
    expect(pageSource).toContain('window.rex.listVoices(voiceApiProvider(data.ttsProvider))')
  })

  it('uses a named voice selector instead of a raw voice id field', () => {
    expect(pageSource).toContain('Choose a Rex voice')
    expect(pageSource).toContain('voice.name')
    expect(pageSource).not.toContain('placeholder="Select or enter a Rex voice"')
  })

  it('previews through the real voice sample IPC and renderer audio playback', () => {
    expect(pageSource).toContain(
      'window.rex.previewVoice(voiceApiProvider(data.ttsProvider), data.ttsVoiceId)'
    )
    expect(pageSource).toContain('new AudioContext()')
    expect(pageSource).toContain('decodeAudioData')
    expect(pageSource).not.toContain('window.rex.testVoice')
  })

  it('reports preview success/failure without treating preview as full voice verification', () => {
    expect(pageSource).toContain('Voice preview played')
    expect(pageSource).toContain('voicePreviewError')
    expect(pageSource).toContain('Voice not yet verified')
  })
})
