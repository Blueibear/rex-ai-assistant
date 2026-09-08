import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const source = readFileSync(resolve(__dirname, '../src/main/handlers/setupPreview.ts'), 'utf8')

describe('US-125 pre-auth setup preview boundary', () => {
  it('does not let an unauthenticated renderer override wake-word asset paths', () => {
    expect(source).not.toContain("'rex:getWakeWordStatus', (_event, values?: Settings)")
    expect(source).not.toContain('const source = values ??')
    expect(source).toContain("const source = (stored.voice ?? {}) as Settings")
  })

  it('only previews voice IDs that AskRex enumerated for setup', () => {
    expect(source).toContain('const setupVoiceInventory = new Map<string, Set<string>>()')
    expect(source).toContain('setupVoiceInventory.set(')
    expect(source).toContain('allowedVoiceIds?.has(voiceId)')
    expect(source).toContain('Choose a voice from the available setup list before previewing it.')
  })
})
