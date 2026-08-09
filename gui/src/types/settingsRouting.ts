export const SETTINGS_CATEGORY_IDS = [
  'general',
  'voice',
  'ai',
  'integrations',
  'notifications',
  'users',
  'audio',
  'system',
  'about'
] as const

export type SettingsCategoryId = (typeof SETTINGS_CATEGORY_IDS)[number]

export function settingsCategoryFromSearch(search: URLSearchParams): SettingsCategoryId {
  const requested = search.get('section')
  return SETTINGS_CATEGORY_IDS.find((category) => category === requested) ?? 'general'
}

export function settingsCategoryFromHash(hash: string): SettingsCategoryId {
  const query = hash.includes('?') ? hash.slice(hash.indexOf('?') + 1) : ''
  return settingsCategoryFromSearch(new URLSearchParams(query))
}
