import { createHash } from 'crypto'
import type { IntegrationConnectionStatus, Settings } from '../types/ipc'
import { readGuiSettings, readRexConfigStrict, writeGuiSettings } from './configStore'
import { getVaultReference } from './credentialReferences'
import { vaultHasSecret, type VaultContext } from './credentialVault'
import { readSavedHomeAssistantCredentials, testHomeAssistantConnection } from './homeAssistant'
import { testOpenClawConnection } from './openClaw'
import { defaultSettingsMap } from './settingsDefaults'
import type { ElectronSessionIdentity } from './sessionIdentity'

export type TestableIntegration = 'email' | 'calendar' | 'sms' | 'homeassistant' | 'phone' | 'openclaw'

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

export const OUTLOOK_EMAIL_UNSUPPORTED =
  'Outlook email sync is not implemented yet. Rex cannot read Outlook mail until Microsoft Graph OAuth token support is added.'
export const OUTLOOK_CALENDAR_UNSUPPORTED =
  'Outlook calendar sync is not implemented yet. Rex cannot read or write Outlook events until Microsoft Graph OAuth token support is added.'

export function hasText(value: unknown): boolean {
  return typeof value === 'string' && value.trim() !== ''
}

export function integrationSettingsFrom(stored: Record<string, Settings>): Record<string, unknown> {
  const explicit = (stored.integrations ?? {}) as Record<string, unknown>
  const merged = {
    ...defaultSettingsMap.integrations,
    ...explicit
  } as Record<string, unknown>
  for (const field of [
    'openclawGatewayUrl',
    'openclawToolsEnabled',
    'openclawVoiceEnabled',
    'openclawToken'
  ]) {
    if (!Object.prototype.hasOwnProperty.call(explicit, field)) delete merged[field]
  }
  return merged
}

function readIntegrationStatuses(stored: Record<string, Settings>): Record<string, StoredIntegrationStatus> {
  const raw = stored.integrationStatuses
  if (!raw || typeof raw !== 'object') return {}
  const statuses: Record<string, StoredIntegrationStatus> = {}
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!value || typeof value !== 'object') continue
    const entry = value as Record<string, unknown>
    const state = String(entry.state ?? 'unconfigured')
    if (![
      'unavailable', 'unconfigured', 'configured', 'reachable', 'authenticated',
      'degraded', 'read_only', 'write_capable', 'write_tested', 'verified'
    ].includes(state)) continue
    statuses[key] = {
      state: state as IntegrationConnectionStatus,
      testedAt: typeof entry.testedAt === 'string' ? entry.testedAt : undefined,
      error: typeof entry.error === 'string' ? entry.error : undefined,
      fingerprint: typeof entry.fingerprint === 'string' ? entry.fingerprint : undefined
    }
  }
  return statuses
}

function context(
  scope: 'household' | 'user', integration: string, account: string | null, slot: string
): VaultContext {
  return { scope, integration, account, slot }
}

async function hasRef(
  session: ElectronSessionIdentity,
  config: Record<string, unknown>,
  logicalName: string,
  expected: VaultContext
): Promise<{ ref: string; hasCredential: boolean } | null> {
  const record = getVaultReference(config, logicalName, expected, session.userId)
  if (!record) return null
  return {
    ref: record.ref,
    hasCredential: await vaultHasSecret(session, record.ref, expected)
  }
}

async function evidenceFor(
  session: ElectronSessionIdentity,
  type: string,
  integrations: Record<string, unknown>,
  config: Record<string, unknown>
): Promise<Record<string, unknown>> {
  if (type === 'email') {
    const direct = await hasRef(session, config, 'EMAIL_CLIENT_SECRET', context('user', 'email', 'default', 'client_secret'))
    const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
    const accountEvidence: Array<Record<string, unknown>> = []
    for (const raw of accounts) {
      if (!raw || typeof raw !== 'object') continue
      const account = raw as Record<string, unknown>
      const id = typeof account.id === 'string' ? account.id : ''
      if (!/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(id)) continue
      const slot = account.backend === 'imap' ? 'password' : 'client_secret'
      const credential = await hasRef(session, config, `email:${id}`, context('user', 'email', id, slot))
      accountEvidence.push({ id, backend: account.backend, host: account.host, username: account.username, clientId: account.clientId, credential })
    }
    return { provider: integrations.emailProvider, clientId: integrations.emailClientId, direct, accounts: accountEvidence }
  }
  if (type === 'calendar') {
    return {
      provider: integrations.calendarProvider,
      clientId: integrations.calendarClientId,
      credential: await hasRef(session, config, 'CALENDAR_CLIENT_SECRET', context('user', 'calendar', 'default', 'client_secret'))
    }
  }
  if (type === 'sms') {
    return {
      sid: await hasRef(session, config, 'TWILIO_ACCOUNT_SID', context('household', 'twilio', 'sms', 'account_sid')),
      token: await hasRef(session, config, 'TWILIO_AUTH_TOKEN', context('household', 'twilio', 'sms', 'auth_token')),
      from: await hasRef(session, config, 'TWILIO_FROM_NUMBER', context('household', 'twilio', 'sms', 'from_number'))
    }
  }
  if (type === 'phone') {
    return {
      sid: await hasRef(session, config, 'TWILIO_PHONE_ACCOUNT_SID', context('household', 'twilio', 'phone', 'account_sid')),
      token: await hasRef(session, config, 'TWILIO_PHONE_AUTH_TOKEN', context('household', 'twilio', 'phone', 'auth_token')),
      number: await hasRef(session, config, 'TWILIO_PHONE_NUMBER', context('household', 'twilio', 'phone', 'phone_number'))
    }
  }
  if (type === 'homeassistant') {
    const saved = await readSavedHomeAssistantCredentials(session)
    return { baseUrl: saved.baseUrl, credentialRef: saved.ref }
  }
  if (type === 'openclaw') {
    const openclaw = config.openclaw && typeof config.openclaw === 'object'
      ? config.openclaw as Record<string, unknown>
      : {}
    return {
      gatewayUrl: openclaw.gateway_url,
      useTools: openclaw.use_tools === true,
      useVoiceBackend: openclaw.use_voice_backend === true,
      credential: await hasRef(
        session,
        config,
        'OPENCLAW_GATEWAY_TOKEN',
        context('household', 'openclaw_gateway', null, 'token')
      )
    }
  }
  return {}
}

function credentialPresent(value: unknown): boolean {
  return Boolean(value && typeof value === 'object' && (value as Record<string, unknown>).hasCredential === true)
}

export async function hasConfiguredEmail(
  session: ElectronSessionIdentity,
  integrations: Record<string, unknown>,
  config: Record<string, unknown> = readRexConfigStrict()
): Promise<boolean> {
  const evidence = await evidenceFor(session, 'email', integrations, config)
  if (hasText(evidence.clientId) && credentialPresent(evidence.direct)) return true
  return Array.isArray(evidence.accounts) && evidence.accounts.some((raw) => {
    const account = raw as Record<string, unknown>
    return credentialPresent(account.credential) && (
      account.backend === 'imap'
        ? hasText(account.host) && hasText(account.username)
        : hasText(account.clientId)
    )
  })
}

async function integrationFingerprint(
  session: ElectronSessionIdentity,
  type: string,
  stored: Record<string, Settings> = readGuiSettings()
): Promise<string> {
  const integrations = integrationSettingsFrom(stored)
  const evidence = await evidenceFor(session, type, integrations, readRexConfigStrict())
  // Evidence contains only non-secret config, opaque refs, and booleans.
  return createHash('sha256').update(JSON.stringify({ type, evidence })).digest('hex')
}

function unsupportedOutlookStatus(type: string, integrations: Record<string, unknown>): StoredIntegrationStatus | null {
  if (type === 'email' && integrations.emailProvider === 'outlook' && hasText(integrations.emailClientId)) {
    return { state: 'unavailable', error: OUTLOOK_EMAIL_UNSUPPORTED }
  }
  if (type === 'calendar' && integrations.calendarProvider === 'outlook' && hasText(integrations.calendarClientId)) {
    return { state: 'unavailable', error: OUTLOOK_CALENDAR_UNSUPPORTED }
  }
  return null
}

export async function integrationStatusFor(
  session: ElectronSessionIdentity,
  type: string,
  stored: Record<string, Settings>
): Promise<StoredIntegrationStatus> {
  const unsupported = unsupportedOutlookStatus(type, integrationSettingsFrom(stored))
  if (unsupported) return unsupported
  const status = readIntegrationStatuses(stored)[type]
  if (!status) return { state: 'unconfigured' }
  return status.fingerprint === await integrationFingerprint(session, type, stored)
    ? status
    : { state: 'unconfigured' }
}

export async function writeIntegrationStatus(
  session: ElectronSessionIdentity,
  type: TestableIntegration,
  result: { ok: boolean; state?: IntegrationConnectionStatus; error?: string }
): Promise<void> {
  const stored = readGuiSettings()
  const statuses = readIntegrationStatuses(stored)
  statuses[type] = {
    state: result.state ?? (result.ok ? 'authenticated' : 'degraded'),
    testedAt: new Date().toISOString(),
    error: result.ok ? undefined : result.error,
    fingerprint: await integrationFingerprint(session, type, stored)
  }
  stored.integrationStatuses = statuses as unknown as Settings
  writeGuiSettings(stored)
}

export async function reconcileIntegrationStatuses(session: ElectronSessionIdentity): Promise<void> {
  const stored = readGuiSettings()
  const statuses = readIntegrationStatuses(stored)
  let changed = false
  for (const [key, status] of Object.entries(statuses)) {
    if (!status.fingerprint || status.fingerprint !== await integrationFingerprint(session, key, stored)) {
      delete statuses[key]
      changed = true
    }
  }
  if (changed) {
    stored.integrationStatuses = statuses as unknown as Settings
    writeGuiSettings(stored)
  }
}

function configuredResult(configured: boolean): IntegrationTestResult {
  return configured
    ? { ok: false, state: 'configured', error: 'Credentials are stored, but provider authentication was not tested.' }
    : { ok: false, state: 'unconfigured', error: 'No complete credential set is stored.' }
}

export async function testIntegrationByType(
  session: ElectronSessionIdentity,
  type: string,
  integrations: Record<string, unknown>
): Promise<{ type?: TestableIntegration; result: IntegrationTestResult }> {
  const unsupported = unsupportedOutlookStatus(type, integrations)
  if (unsupported) return { type: type as TestableIntegration, result: { ok: false, state: unsupported.state, error: unsupported.error } }
  const config = readRexConfigStrict()
  if (type === 'email') return { type, result: configuredResult(await hasConfiguredEmail(session, integrations, config)) }
  if (type === 'calendar') {
    const evidence = await evidenceFor(session, type, integrations, config)
    return { type, result: configuredResult(hasText(evidence.clientId) && credentialPresent(evidence.credential)) }
  }
  if (type === 'sms' || type === 'phone') {
    const evidence = await evidenceFor(session, type, integrations, config)
    return { type, result: configuredResult(Object.values(evidence).every(credentialPresent)) }
  }
  if (type === 'homeassistant') {
    const { baseUrl, token } = await readSavedHomeAssistantCredentials(session)
    const result = await testHomeAssistantConnection(baseUrl, token)
    return {
      type,
      result: result.ok
        ? { ...result, state: 'authenticated' }
        : { ...result, state: hasText(baseUrl) && hasText(token) ? 'degraded' : 'unconfigured' }
    }
  }
  if (type === 'openclaw') {
    return { type, result: await testOpenClawConnection(session) }
  }
  return { result: { ok: false, state: 'unavailable', error: 'Unknown integration type' } }
}
