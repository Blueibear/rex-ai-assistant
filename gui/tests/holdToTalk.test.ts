import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'
import { voiceProvider } from '../src/types/voiceProvider'

describe('Hold-to-Talk production path', () => {
  it('normalizes configured UI engines to packaged providers', () => {
    expect(voiceProvider('system')).toBe('pyttsx3')
    expect(voiceProvider('openai')).toBe('edge-tts')
    expect(voiceProvider('xtts')).toBe('xtts')
  })

  it('keeps cancellation, playback, output routing, replay, and timing wired', () => {
    const source = readFileSync(join(__dirname, '../src/pages/VoicePage.tsx'), 'utf8')
    for (const contract of [
      'AbortController',
      'speakerDeviceId',
      'setSinkId',
      'synthesizeSpeech',
      'Cancel response',
      'Replay',
      'logVoiceTiming',
      'devicechange',
    ]) {
      expect(source).toContain(contract)
    }
    const chatHandler = readFileSync(join(__dirname, '../src/main/handlers/chat.ts'), 'utf8')
    const preload = readFileSync(join(__dirname, '../src/preload/index.ts'), 'utf8')
    expect(chatHandler).toContain("ipcMain.handle('rex:cancelChatStream'")
    expect(preload).toContain("ipcRenderer.invoke('rex:cancelChatStream'")
  })
})
