// Secret redaction for persisted GUI settings (US-027).
//
// Secrets belong in the OS credential vault. config/gui_settings.json must never
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
  'auth_key',
  'accountsid',
  'account_sid',
  'fromnumber',
  'from_number',
  'phonenumber',
  'phone_number',
  'transfernumber',
  'transfer_number'
]

export function isSecretSettingKey(key: string): boolean {
  const k = key.toLowerCase()
  // Opaque vault references and boolean presence metadata are safe to persist.
  if (k.endsWith('ref') || k.endsWith('refs') || (k.startsWith('has') && k.endsWith('credential'))) return false
  // Keys that *name* an environment variable (e.g. api_key_env) hold a
  // variable name, not a secret value.
  if (k.endsWith('_env')) return false
  // ha_token, authToken, refresh_token… — but not max_tokens / tokens.
  if (k === 'token' || k.endsWith('token')) return true
  if (k === 'sid' || k.endsWith('sid')) return true
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
