import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'

describe('voice audio diagnostics', () => {
  it('surfaces actionable bridge diagnostics through Electron to an alert', () => {
    const bridge = readFileSync(join(__dirname, '../../bridge/rex_voice_bridge.py'), 'utf8')
    const handler = readFileSync(join(__dirname, '../src/main/handlers/voice.ts'), 'utf8')
    const preload = readFileSync(join(__dirname, '../src/preload/index.ts'), 'utf8')
    const page = readFileSync(join(__dirname, '../src/pages/VoicePage.tsx'), 'utf8')

    expect(bridge).toContain('diagnostic_callback=emit_audio_diagnostic')
    expect(bridge).toContain('"error": user_message')
    expect(handler).toContain("broadcastVoiceEvent('rex:voiceError', { error })")
    expect(handler).toContain('context.failStartup(error)')
    expect(preload).toContain("ipcRenderer.on('rex:voiceError'")
    expect(page).toContain('role="alert"')
    expect(page).toContain('subtext={error}')
  })
})
