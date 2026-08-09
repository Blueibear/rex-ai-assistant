import { describe, expect, it } from 'vitest'
import { settingsCategoryFromSearch } from '../src/types/settingsRouting'

describe('Settings query routing', () => {
  it('opens a valid requested settings section', () => {
    expect(settingsCategoryFromSearch(new URLSearchParams('section=integrations'))).toBe('integrations')
    expect(settingsCategoryFromSearch(new URLSearchParams('section=voice'))).toBe('voice')
  })

  it('fails safely to General for missing or invalid sections', () => {
    expect(settingsCategoryFromSearch(new URLSearchParams())).toBe('general')
    expect(settingsCategoryFromSearch(new URLSearchParams('section=not-real'))).toBe('general')
  })
})
