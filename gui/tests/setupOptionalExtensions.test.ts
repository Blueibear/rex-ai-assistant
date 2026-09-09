import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'

const extensionSource = readFileSync(
  fileURLToPath(new URL('../src/pages/SetupOptionalExtensionsPage.tsx', import.meta.url)),
  'utf8'
)
const appSource = readFileSync(
  fileURLToPath(new URL('../src/renderer/src/App.tsx', import.meta.url)),
  'utf8'
)

describe('US-125 optional household extensions', () => {
  it('offers Home Assistant, additional household voice enrollment, and another room as optional follow-ups', () => {
    expect(extensionSource).toContain('Optional household setup')
    expect(extensionSource).toContain('Home Assistant')
    expect(extensionSource).toContain('Additional household voice')
    expect(extensionSource).toContain('Additional room endpoint')
    expect(extensionSource).toContain('Not now, open dashboard')
  })

  it('routes optional follow-ups through existing settings and pairing surfaces', () => {
    expect(extensionSource).toContain("'/settings/home-assistant'")
    expect(extensionSource).toContain("'/settings?section=users'")
    expect(extensionSource).toContain("'/pairing'")
  })

  it('shows the optional offer only after first-run setup completes', () => {
    expect(appSource).toContain('showSetupExtensions')
    expect(appSource).toContain('setShowSetupExtensions(true)')
    expect(appSource).toContain('<SetupOptionalExtensionsPage')
  })
})
