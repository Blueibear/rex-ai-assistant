import { existsSync, readFileSync, readdirSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const root = join(__dirname, '..')
const settingsPage = join(root, 'src/pages/SettingsPage.tsx')
const settingsDir = join(root, 'src/pages/settings')

const expectedImports = [
  'GeneralSettingsSection',
  'VoiceSettingsSection',
  'AiSettingsSection',
  'IntegrationsSettingsSection',
  'NotificationsSettingsSection',
  'UsersSettingsSection',
  'AudioOutputSettingsSection',
  'SystemSettingsSection',
  'AboutSettingsSection'
]

function lineCount(path: string): number {
  return readFileSync(path, 'utf8').split(/\r?\n/).length
}

describe('SettingsPage decomposition', () => {
  it('keeps the facade small and delegates every category', () => {
    expect(lineCount(settingsPage)).toBeLessThan(700)
    const source = readFileSync(settingsPage, 'utf8')
    for (const name of expectedImports) {
      expect(source).toContain(name)
    }
    for (const category of [
      'general', 'voice', 'ai', 'integrations', 'notifications', 'users', 'audio', 'system', 'about'
    ]) {
      expect(source).toContain(`case '${category}'`)
    }
  })

  it('keeps every extracted settings module under 1000 lines', () => {
    expect(existsSync(settingsDir)).toBe(true)
    const files = readdirSync(settingsDir, { recursive: true })
      .map(String)
      .filter((name) => /\.(ts|tsx)$/.test(name))
    expect(files.length).toBeGreaterThanOrEqual(9)
    for (const name of files) {
      expect(lineCount(join(settingsDir, name)), name).toBeLessThan(1000)
    }
  })
})

describe('AI settings navigation persistence (US-071)', () => {
  it('reloads AI settings whenever navigation remounts the AI panel', () => {
    const pageSource = readFileSync(settingsPage, 'utf8')
    const aiSource = readFileSync(join(settingsDir, 'AiSettingsSection.tsx'), 'utf8')

    expect(pageSource).toContain("case 'ai'")
    expect(pageSource).toContain('<AiSettingsSection />')
    expect(pageSource).toContain('renderPanel(activeCategory)')
    expect(aiSource).toContain("getSettings('ai')")
  })
})
