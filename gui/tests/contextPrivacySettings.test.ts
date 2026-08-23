import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const section = readFileSync(
  join(process.cwd(), 'src/pages/settings/ContextPrivacySettingsSection.tsx'),
  'utf8'
)
const settingsPage = readFileSync(join(process.cwd(), 'src/pages/SettingsPage.tsx'), 'utf8')
const preload = readFileSync(join(process.cwd(), 'src/preload/index.ts'), 'utf8')
const handler = readFileSync(join(process.cwd(), 'src/main/handlers/contextPolicy.ts'), 'utf8')

describe('Context & Privacy settings (US-123)', () => {
  it('exposes context/privacy as its own settings section', () => {
    expect(settingsPage).toContain("id: 'privacy'")
    expect(settingsPage).toContain("label: 'Context & Privacy'")
    expect(settingsPage).toContain('<ContextPrivacySettingsSection />')
  })

  it('uses clear separate controls for context, location, sharing, and proactivity', () => {
    expect(section).toContain('Use this in future conversations')
    expect(section).toContain('Private to me')
    expect(section).toContain('Shared household')
    expect(section).toContain('Use my location to help me')
    expect(section).toContain('Share my location with')
    expect(section).toContain('Proactive assistance')
  })

  it('uses the dedicated privacy bridge instead of generic persisted settings', () => {
    expect(preload).toContain("ipcRenderer.invoke('rex:getContextPrivacy')")
    expect(preload).toContain("ipcRenderer.invoke('rex:updateContextPrivacy', command, payload)")
    expect(handler).toContain("resolveBridgePath('rex_context_policy_bridge.py')")
    expect(handler).toContain('privateSessionPayload(session, payload)')
  })

  it('renders only safe source/upload metadata, never document content or source paths', () => {
    expect(section).toContain('source.source_type')
    expect(section).toContain('upload.title')
    expect(section).not.toContain('upload.content')
    expect(section).not.toContain('source_path')
  })

  it('sends each privacy mutation as a distinct owner-bound command', () => {
    expect(section).toContain("'set_source_context'")
    expect(section).toContain("'update_upload_policy'")
    expect(section).toContain("'set_location_assist'")
    expect(section).toContain("'set_location_share'")
    expect(section).toContain("'set_proactive_assistance'")
  })
})
