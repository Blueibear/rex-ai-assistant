// Secret redaction for persisted GUI settings (US-027).
//
// Secrets belong in .env (the canonical secret store, written via
// writeEnvKey in configStore.ts). config/gui_settings.json must never
// contain a credential, so writeGuiSettings strips any key matching a
// secret pattern at any nesting depth before persisting.

const SECRET_KEY_FRAGMENTS = [
  'password',
  'passwd',
  'secret',
  'credential',
  'apikey',
  'api_key',
  'api-key',
  'privatekey',
  'private_key',
  'authkey',
  'auth_key'
]

export function isSecretSettingKey(key: string): boolean {
  const k = key.toLowerCase()
  // Keys that *name* an environment variable (e.g. api_key_env) hold a
  // variable name, not a secret value.
  if (k.endsWith('_env')) return false
  // ha_token, authToken, refresh_token… — but not max_tokens / tokens.
  if (k === 'token' || k.endsWith('token')) return true
  return SECRET_KEY_FRAGMENTS.some((fragment) => k.includes(fragment))
}

export function redactSecretSettings<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((item) => redactSecretSettings(item)) as unknown as T
  }
  if (value !== null && typeof value === 'object') {
    const result: Record<string, unknown> = {}
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      if (isSecretSettingKey(key)) continue
      result[key] = redactSecretSettings(entry)
    }
    return result as unknown as T
  }
  return value
}
