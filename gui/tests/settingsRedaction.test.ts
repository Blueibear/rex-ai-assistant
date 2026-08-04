import { describe, expect, it } from 'vitest'
import { isSecretSettingKey, redactSecretSettings } from '../src/main/settingsRedaction'

describe('GUI settings secret redaction (US-027)', () => {
  it('classifies secret-pattern keys', () => {
    for (const key of [
      'ha_token',
      'haToken',
      'token',
      'authToken',
      'refresh_token',
      'llm_api_key',
      'apiKey',
      'openai_api_key',
      'openrouter_api_key',
      'password',
      'smtp_passwd',
      'client_secret',
      'emailClientSecret',
      'calendarClientSecret',
      'smsSid',
      'smsAuthToken',
      'smsFromNumber',
      'phoneSid',
      'phoneAuthToken',
      'phoneNumber',
      'phoneTransferNumber',
      'telegramBotToken',
      'twilio_credential',
      'private_key'
    ]) {
      expect(isSecretSettingKey(key), key).toBe(true)
    }
  })

  it('keeps legitimate non-secret keys', () => {
    for (const key of [
      'max_tokens',
      'maxTokens',
      'tokens',
      'api_key_env',
      'tts_provider',
      'llm_provider',
      'ha_base_url',
      'theme',
      'username',
      'credentialRef',
      'hasCredential'
    ]) {
      expect(isSecretSettingKey(key), key).toBe(false)
    }
  })

  it('strips secrets at any nesting depth before persisting', () => {
    // Redaction is key-based; values are inert fixture placeholders.
    const fixturePlaceholder = 'synthetic-fixture-value' // pragma: allowlist secret
    const input = {
      james: {
        theme: 'dark',
        ha_token: fixturePlaceholder,
        ai: { llm_provider: 'openai', llm_api_key: fixturePlaceholder, max_tokens: 512 },
        devices: [{ name: 'lamp', token: fixturePlaceholder }]
      }
    }
    expect(redactSecretSettings(input)).toEqual({
      james: {
        theme: 'dark',
        ai: { llm_provider: 'openai', max_tokens: 512 },
        devices: [{ name: 'lamp' }]
      }
    })
  })

  it('passes primitives and null through unchanged', () => {
    expect(redactSecretSettings('value')).toBe('value')
    expect(redactSecretSettings(42)).toBe(42)
    expect(redactSecretSettings(null)).toBe(null)
    expect(redactSecretSettings(true)).toBe(true)
  })
})
