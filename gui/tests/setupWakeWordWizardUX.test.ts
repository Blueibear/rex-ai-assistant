import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'

const pageSource = readFileSync(
  fileURLToPath(new URL('../src/pages/SetupWizardPage.tsx', import.meta.url)),
  'utf8'
)

describe('US-125 setup wake-word UX', () => {
  it('loads the canonical wake-word inventory', () => {
    expect(pageSource).toContain('window.rex.listWakeWords()')
    expect(pageSource).toContain('wakeWords')
    expect(pageSource).toContain('wake_words')
  })

  it('uses a named wake-word selector instead of a raw id field', () => {
    expect(pageSource).toContain('Choose a wake word')
    expect(pageSource).toContain('wakeWords.map((wakeWord)')
    expect(pageSource).toContain('wakeWord.name')
    expect(pageSource).not.toContain('placeholder="hey_rex"')
  })

  it('previews a real wake-word sample without claiming detection', () => {
    expect(pageSource).toContain('window.rex.previewWakeWordSample(data.wakeWordId)')
    expect(pageSource).toContain('Wake-word sample played')
    expect(pageSource).toContain('Actual wake detection is still required')
  })

  it('reports canonical wake-word asset readiness separately from full verification', () => {
    expect(pageSource).toContain('window.rex.getWakeWordStatus')
    expect(pageSource).toContain('Wake-word asset ready')
    expect(pageSource).toContain('wakeWordStatusError')
    expect(pageSource).toContain('Voice not yet verified')
  })
})
