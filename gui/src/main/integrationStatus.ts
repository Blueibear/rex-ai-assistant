import { createHash } from 'crypto'
import type { IntegrationConnectionStatus, Settings } from '../types/ipc'
import { readEnvFile, readGuiSettings, readRexConfig, writeGuiSettings } from './configStore'
import { readSavedHomeAssistantCredentials, testHomeAssistantConnection } from './homeAssistant'
import { defaultSettingsMap } from './settingsDefaults'

export type TestableIntegration = 'email' | 'calendar' | 'sms' | 'homeassistant' | 'phone'

export type IntegrationTestResult = {
  ok: boolean
  state: IntegrationConnectionStatus
  error?: string
}

export interface StoredIntegrationStatus {
  state: IntegrationConnectionStatus
  testedAt?: string
  error?: string
  fingerprint?: string
}

export function hasText(value: unknown): boolean {
  return typeof value === 'string' && value.trim() !== ''
}

export function integrationSettingsFrom(stored: Record<string, Settings>): Record<string, unknown> {
  return {
    ...defaultSettingsMap.integrations,
    ...((stored.integrations ?? {}) as Record<string, unknown>)
  }
}

export const OUTLOOK_EMAIL_UNSUPPORTED =
  'Outlook email sync is not implemented yet. The current Outlook settings only store app credentials; Rex cannot read Outlook mail until Microsoft Graph OAuth token support is added.'

export const OUTLOOK_CALENDAR_UNSUPPORTED =
  'Outlook calendar sync is not implemented yet. The current Outlook settings only store app credentials; Rex cannot read or write Outlook events until Microsoft Graph OAuth token support is added.'

function hasConfiguredOutlookEmail(integrations: Record<string, unknown>): boolean {
  if (
    integrations.emailProvider === 'outlook' &&
    hasText(integrations.emailClientId) &&
    hasText(integrations.emailClientSecret) // pragma: allowlist secret
  ) {
    return true
  }
  const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
  return accounts.some((raw) => {
    if (!raw || typeof raw !== 'object') return false
    const account = raw as Record<string, unknown>
    return account.backend === 'outlook' && hasText(account.clientId) && hasText(account.clientSecret) // pragma: allowlist secret
  })
}

function hasConfiguredOutlookCalendar(integrations: Record<string, unknown>): boolean {
  return (
    integrations.calendarProvider === 'outlook' &&
    hasText(integrations.calendarClientId) &&
    hasText(integrations.calendarClientSecret) // pragma: allowlist secret
  )
}

function unsupportedOutlookStatus(
  type: string,
  stored: Record<string, Settings>
): StoredIntegrationStatus | null {
  const integrations = integrationSettingsFrom(stored)
  if (type === 'email' && hasConfiguredOutlookEmail(integrations)) {
    return { state: 'unavailable', error: OUTLOOK_EMAIL_UNSUPPORTED }
  }
  if (type === 'calendar' && hasConfiguredOutlookCalendar(integrations)) {
    return { state: 'unavailable', error: OUTLOOK_CALENDAR_UNSUPPORTED }
  }
  return null
}

function readIntegrationStatuses(stored: Record<string, Settings>): Record<string, StoredIntegrationStatus> {
  const raw = stored.integrationStatuses
  if (!raw || typeof raw !== 'object') return {}
  const statuses: Record<string, StoredIntegrationStatus> = {}
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!value || typeof value !== 'object') continue
    const entry = value as Record<string, unknown>
    const legacyStatus = entry.status
    const state = entry.state ?? (
      legacyStatus === 'connected' ? 'configured' :
        legacyStatus === 'error' ? 'degraded' : 'unconfigured'
    )
    if (![
      'unavailable', 'unconfigured', 'configured', 'reachable', 'authenticated',
      'degraded', 'read_only', 'write_capable', 'write_tested', 'verified'
    ].includes(String(state))) continue
    statuses[key] = {
      state: state as IntegrationConnectionStatus,
      testedAt: typeof entry.testedAt === 'string' ? entry.testedAt : undefined,
      error: typeof entry.error === 'string' ? entry.error : undefined,
      fingerprint: typeof entry.fingerprint === 'string' ? entry.fingerprint : undefined
    }
  }
  return statuses
}

function integrationFingerprint(
  type: string,
  stored: Record<string, Settings> = readGuiSettings()
): string {
  const integrations = integrationSettingsFrom(stored)
  const rexConfig = readRexConfig()
  const env = readEnvFile()
  const ha = readSavedHomeAssistantCredentials()
  const payload: Record<string, unknown> = {}

  if (type === 'email') {
    payload.emailProvider = integrations.emailProvider
    payload.emailClientId = integrations.emailClientId
    payload.emailClientSecret = integrations.emailClientSecret // pragma: allowlist secret
    payload.emailAccounts = integrations.emailAccounts
  } else if (type === 'calendar') {
    payload.calendarProvider = integrations.calendarProvider
    payload.calendarClientId = integrations.calendarClientId
    payload.calendarClientSecret = integrations.calendarClientSecret // pragma: allowlist secret
  } else if (type === 'sms') {
    payload.smsSid = integrations.smsSid
    payload.smsAuthToken = integrations.smsAuthToken
    payload.smsFromNumber = integrations.smsFromNumber
    payload.twilioEnvSid = env.TWILIO_ACCOUNT_SID
    payload.twilioEnvToken = env.TWILIO_AUTH_TOKEN
    payload.twilioEnvFromNumber = env.TWILIO_FROM_NUMBER
  } else if (type === 'homeassistant') {
    payload.haUrl = ha.baseUrl
    payload.haToken = ha.token
  } else if (type === 'phone') {
    payload.phoneSid = integrations.phoneSid
    payload.phoneAuthToken = integrations.phoneAuthToken
    payload.phoneNumber = integrations.phoneNumber
    payload.twilioEnvSid = env.TWILIO_ACCOUNT_SID
    payload.twilioEnvToken = env.TWILIO_AUTH_TOKEN
    payload.twilioEnvPhoneNumber = env.TWILIO_PHONE_NUMBER
  } else if (type === 'telegram') {
    payload.telegramBotToken = integrations.telegramBotToken || env.TELEGRAM_BOT_TOKEN
    payload.telegramChatId = integrations.telegramChatId
  } else if (type === 'search') {
    payload.serpapi = env.SERPAPI_API_KEY
    payload.brave = env.BRAVE_API_KEY
    payload.google = env.GOOGLE_CSE_ID
  } else if (type === 'mqtt') {
    payload.mqtt = env.MQTT_BROKER_HOST
  } else if (type === 'openai') {
    const openai = rexConfig.openai && typeof rexConfig.openai === 'object'
      ? (rexConfig.openai as Record<string, unknown>)
      : {}
    payload.openai = env.OPENAI_API_KEY || openai.api_key // pragma: allowlist secret
  } else if (type === 'ollama') {
    const ollama = rexConfig.ollama && typeof rexConfig.ollama === 'object'
      ? (rexConfig.ollama as Record<string, unknown>)
      : {}
    payload.ollamaBaseUrl = ollama.base_url
  } else if (type === 'push') {
    payload.pushProvider = integrations.pushProvider
    payload.pushToken = integrations.pushToken
  } else if (type === 'openclaw') {
    const openclaw = rexConfig.openclaw && typeof rexConfig.openclaw === 'object'
      ? (rexConfig.openclaw as Record<string, unknown>)
      : {}
    payload.openclawGatewayUrl = openclaw.gateway_url
  }

  return createHash('sha256').update(JSON.stringify(payload)).digest('hex')
}

export function integrationFingerprintForValues(type: string, values: Record<string, unknown>): string {
  if (type === 'homeassistant') {
    return createHash('sha256').update(JSON.stringify(values)).digest('hex')
  }
  return createHash('sha256').update(JSON.stringify({ type, ...values })).digest('hex')
}

export function integrationStatusFor(
  type: string,
  stored: Record<string, Settings>
): StoredIntegrationStatus {
  const unsupportedStatus = unsupportedOutlookStatus(type, stored)
  if (unsupportedStatus) return unsupportedStatus

  const status = readIntegrationStatuses(stored)[type]
  if (!status) return { state: 'unconfigured' }
  if (status.fingerprint !== integrationFingerprint(type, stored)) {
    return { state: 'unconfigured' }
  }
  return status
}

export function writeIntegrationStatus(
  type: TestableIntegration,
  result: { ok: boolean; state?: IntegrationConnectionStatus; error?: string },
  fingerprint = integrationFingerprint(type)
): void {
  const stored = readGuiSettings()
  const statuses = readIntegrationStatuses(stored)
  statuses[type] = {
    state: result.state ?? (result.ok ? 'authenticated' : 'degraded'),
    testedAt: new Date().toISOString(),
    error: result.ok ? undefined : result.error,
    fingerprint
  }
  stored.integrationStatuses = statuses as unknown as Settings
  writeGuiSettings(stored)
}

export function reconcileIntegrationStatuses(): void {
  const stored = readGuiSettings()
  const statuses = readIntegrationStatuses(stored)
  let changed = false
  for (const [key, status] of Object.entries(statuses)) {
    if (
      unsupportedOutlookStatus(key, stored) ||
      !status.fingerprint ||
      status.fingerprint !== integrationFingerprint(key, stored)
    ) {
      delete statuses[key]
      changed = true
    }
  }
  if (changed) {
    stored.integrationStatuses = statuses as unknown as Settings
    writeGuiSettings(stored)
  }
}

export function hasConfiguredEmail(integrations: Record<string, unknown>): boolean {
  if (hasText(integrations.emailClientId) && hasText(integrations.emailClientSecret)) return true // pragma: allowlist secret
  const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
  return accounts.some((raw) => {
    if (!raw || typeof raw !== 'object') return false
    const account = raw as Record<string, unknown>
    if (account.backend === 'imap') {
      return hasText(account.host) && hasText(account.username) && hasText(account.password) // pragma: allowlist secret
    }
    return hasText(account.clientId) && hasText(account.clientSecret) // pragma: allowlist secret
  })
}

function hasDirectEmailCredentials(integrations: Record<string, unknown>): boolean {
  return hasText(integrations.emailClientId) && hasText(integrations.emailClientSecret) // pragma: allowlist secret
}

function hasDirectCalendarCredentials(integrations: Record<string, unknown>): boolean {
  return hasText(integrations.calendarClientId) && hasText(integrations.calendarClientSecret) // pragma: allowlist secret
}

function hasGuiSmsCredentials(integrations: Record<string, unknown>): boolean {
  return (
    hasText(integrations.smsSid) &&
    hasText(integrations.smsAuthToken) &&
    hasText(integrations.smsFromNumber)
  )
}

function hasEnvSmsCredentials(env: Record<string, string>): boolean {
  return hasText(env.TWILIO_ACCOUNT_SID) && hasText(env.TWILIO_AUTH_TOKEN)
}

function hasGuiPhoneCredentials(integrations: Record<string, unknown>): boolean {
  return (
    hasText(integrations.phoneSid) &&
    hasText(integrations.phoneAuthToken) &&
    hasText(integrations.phoneNumber)
  )
}

function hasEnvPhoneCredentials(env: Record<string, string>): boolean {
  return (
    hasText(env.TWILIO_ACCOUNT_SID) &&
    hasText(env.TWILIO_AUTH_TOKEN) &&
    hasText(env.TWILIO_PHONE_NUMBER)
  )
}

function integrationConfiguredResult(configured: boolean): IntegrationTestResult {
  return configured
    ? {
        ok: false,
        state: 'configured',
        error: 'Credentials are present, but this check does not prove reachability or authentication.'
      }
    : { ok: false, state: 'unconfigured', error: 'No credentials configured' }
}

function testEmailIntegration(integrations: Record<string, unknown>): IntegrationTestResult {
  if (hasConfiguredOutlookEmail(integrations)) {
    return { ok: false, state: 'unavailable', error: OUTLOOK_EMAIL_UNSUPPORTED }
  }
  return integrationConfiguredResult(hasDirectEmailCredentials(integrations) || hasConfiguredEmail(integrations))
}

function testCalendarIntegration(integrations: Record<string, unknown>): IntegrationTestResult {
  if (hasConfiguredOutlookCalendar(integrations)) {
    return { ok: false, state: 'unavailable', error: OUTLOOK_CALENDAR_UNSUPPORTED }
  }
  return integrationConfiguredResult(hasDirectCalendarCredentials(integrations))
}

function testSmsIntegration(integrations: Record<string, unknown>): IntegrationTestResult {
  const env = readEnvFile()
  return integrationConfiguredResult(hasGuiSmsCredentials(integrations) || hasEnvSmsCredentials(env))
}

function testPhoneIntegration(integrations: Record<string, unknown>): IntegrationTestResult {
  const env = readEnvFile()
  return integrationConfiguredResult(hasGuiPhoneCredentials(integrations) || hasEnvPhoneCredentials(env))
}

export async function testIntegrationByType(
  type: string,
  integrations: Record<string, unknown>
): Promise<{ type?: TestableIntegration; result: IntegrationTestResult }> {
  if (type === 'email') return { type, result: testEmailIntegration(integrations) }
  if (type === 'calendar') return { type, result: testCalendarIntegration(integrations) }
  if (type === 'sms') return { type, result: testSmsIntegration(integrations) }
  if (type === 'phone') return { type, result: testPhoneIntegration(integrations) }
  if (type === 'homeassistant') {
    const { baseUrl, token } = readSavedHomeAssistantCredentials()
    const result = await testHomeAssistantConnection(baseUrl, token)
    return {
      type,
      result: result.ok
        ? { ...result, state: 'authenticated' }
        : {
            ...result,
            state: hasText(baseUrl) && hasText(token) ? 'degraded' : 'unconfigured'
          }
    }
  }
  return { result: { ok: false, state: 'unavailable', error: 'Unknown integration type' } }
}
