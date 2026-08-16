import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const root = join(__dirname, '..')
const aiSection = readFileSync(join(root, 'src/pages/settings/AiSettingsSection.tsx'), 'utf8')
const systemSection = readFileSync(join(root, 'src/pages/settings/SystemSettingsSection.tsx'), 'utf8')
const defaults = readFileSync(join(root, 'src/main/settingsDefaults.ts'), 'utf8')
const mirror = readFileSync(join(root, 'src/main/settingsMirror.ts'), 'utf8')

describe('autonomy settings UI authority (US-073)', () => {
  it('keeps the only Autonomy Mode control under AI settings', () => {
    expect(aiSection).toContain('Autonomy Mode')
    expect(aiSection).toContain('id="autonomyMode"')
    expect(systemSection).not.toContain('Autonomy Mode')
    expect(systemSection).not.toContain('autonomyMode')
  })

  it('does not define a duplicate System autonomy default or mirror path', () => {
    const systemDefaults = defaults.slice(defaults.indexOf('system: {'))
    expect(systemDefaults).not.toContain('autonomyMode')
    const systemMirror = mirror.slice(mirror.indexOf("if (section === 'system')"))
    expect(systemMirror).not.toContain('models.autonomy_mode')
  })
})
