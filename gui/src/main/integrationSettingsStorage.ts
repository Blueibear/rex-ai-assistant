import type { Settings } from '../types/ipc'
import { readGuiSettings, readRexConfigStrict, writeGuiSettings, writeRexConfig } from './configStore'
import { defaultSettingsMap } from './settingsDefaults'
import { mirrorToRexConfig } from './settingsMirror'
import { reconcileIntegrationStatuses } from './integrationStatus'
import {
  vaultDeleteSecret,
  vaultHasSecret,
  vaultSetSecret,
  type VaultContext
} from './credentialVault'
import {
  deleteVaultReference,
  getVaultReference,
  putVaultReference,
  type VaultReferenceRecord
} from './credentialReferences'
import type { ElectronSessionIdentity } from './sessionIdentity'
import { safeIpcErrorMessage, SafeValidationError } from './ipcErrors'

interface SecretFieldSpec {
  field: string
  logicalName: string
  context: VaultContext
}

interface StagedEntry {
  logicalName: string
  ref: string
  context: VaultContext
}

interface ReplacedEntry {
  record: VaultReferenceRecord
  context: VaultContext
}

type SettingsStore = Record<string, Settings>
type RexConfig = Record<string, unknown>
type IntegrationRecord = Record<string, unknown>

const INTEGRATION_SECRET_FIELDS: SecretFieldSpec[] = [
  { field: 'emailClientSecret', logicalName: 'EMAIL_CLIENT_SECRET', context: { scope: 'user', integration: 'email', account: 'default', slot: 'client_secret' } },
  { field: 'calendarClientSecret', logicalName: 'CALENDAR_CLIENT_SECRET', context: { scope: 'user', integration: 'calendar', account: 'default', slot: 'client_secret' } },
  { field: 'smsSid', logicalName: 'TWILIO_ACCOUNT_SID', context: { scope: 'household', integration: 'twilio', account: 'sms', slot: 'account_sid' } },
  { field: 'smsAuthToken', logicalName: 'TWILIO_AUTH_TOKEN', context: { scope: 'household', integration: 'twilio', account: 'sms', slot: 'auth_token' } },
  { field: 'smsFromNumber', logicalName: 'TWILIO_FROM_NUMBER', context: { scope: 'household', integration: 'twilio', account: 'sms', slot: 'from_number' } },
  { field: 'haToken', logicalName: 'HA_TOKEN', context: { scope: 'household', integration: 'home_assistant', account: null, slot: 'token' } },
  { field: 'phoneSid', logicalName: 'TWILIO_PHONE_ACCOUNT_SID', context: { scope: 'household', integration: 'twilio', account: 'phone', slot: 'account_sid' } },
  { field: 'phoneAuthToken', logicalName: 'TWILIO_PHONE_AUTH_TOKEN', context: { scope: 'household', integration: 'twilio', account: 'phone', slot: 'auth_token' } },
  { field: 'phoneNumber', logicalName: 'TWILIO_PHONE_NUMBER', context: { scope: 'household', integration: 'twilio', account: 'phone', slot: 'phone_number' } },
  { field: 'phoneTransferNumber', logicalName: 'TWILIO_TRANSFER_NUMBER', context: { scope: 'household', integration: 'twilio', account: 'phone', slot: 'transfer_number' } },
  { field: 'telegramBotToken', logicalName: 'TELEGRAM_BOT_TOKEN', context: { scope: 'household', integration: 'telegram', account: null, slot: 'token' } },
  { field: 'openclawToken', logicalName: 'OPENCLAW_GATEWAY_TOKEN', context: { scope: 'household', integration: 'openclaw_gateway', account: null, slot: 'token' } }
]

export function validateAccountId(value: unknown): string {
  const id = typeof value === 'string' ? value : ''
  if (!/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(id)) {
    throw new Error('Email account ID is invalid')
  }
  return id
}

function emailBackend(account: IntegrationRecord): 'gmail' | 'outlook' | 'imap' {
  if (account.backend === 'imap') return 'imap'
  return account.backend === 'outlook' ? 'outlook' : 'gmail'
}

function emailContext(id: string, backend: 'gmail' | 'outlook' | 'imap'): VaultContext {
  return {
    scope: 'user',
    integration: 'email',
    account: id,
    slot: backend === 'imap' ? 'password' : 'client_secret'
  }
}

function applyOpenClawConfigFallback(
  integrations: IntegrationRecord,
  explicitIntegrations: IntegrationRecord,
  config: RexConfig
): void {
  const openclaw = config.openclaw && typeof config.openclaw === 'object'
    ? config.openclaw as IntegrationRecord
    : {}
  if (!Object.prototype.hasOwnProperty.call(explicitIntegrations, 'openclawGatewayUrl')) {
    integrations.openclawGatewayUrl = typeof openclaw.gateway_url === 'string' ? openclaw.gateway_url : ''
  }
  if (!Object.prototype.hasOwnProperty.call(explicitIntegrations, 'openclawToolsEnabled')) {
    integrations.openclawToolsEnabled = openclaw.use_tools === true
  }
  if (!Object.prototype.hasOwnProperty.call(explicitIntegrations, 'openclawVoiceEnabled')) {
    integrations.openclawVoiceEnabled = openclaw.use_voice_backend === true
  }
}

async function hydrateFlatSecretStatus(
  session: ElectronSessionIdentity,
  config: RexConfig,
  integrations: IntegrationRecord
): Promise<Record<string, { ref: string; hasCredential: boolean }>> {
  const status: Record<string, { ref: string; hasCredential: boolean }> = {}
  for (const spec of INTEGRATION_SECRET_FIELDS) {
    integrations[spec.field] = ''
    const record = getVaultReference(config, spec.logicalName, spec.context, session.userId)
    if (!record) continue
    status[spec.field] = {
      ref: record.ref,
      hasCredential: await vaultHasSecret(session, record.ref, spec.context)
    }
  }
  return status
}

async function hydrateEmailAccount(
  session: ElectronSessionIdentity,
  config: RexConfig,
  raw: unknown
): Promise<IntegrationRecord> {
  const account: IntegrationRecord = {
    ...((raw ?? {}) as IntegrationRecord),
    password: '',
    clientSecret: ''
  }
  const id = validateAccountId(account.id)
  const context = emailContext(id, emailBackend(account))
  const record = getVaultReference(config, `email:${id}`, context, session.userId)
  account.hasCredential = false
  if (!record) return account
  account.credentialRef = record.ref
  account.hasCredential = await vaultHasSecret(session, record.ref, context)
  return account
}

export async function loadIntegrationSettings(
  session: ElectronSessionIdentity,
  stored: SettingsStore
): Promise<Settings> {
  const integrations = {
    ...defaultSettingsMap.integrations,
    ...((stored.integrations ?? {}) as IntegrationRecord)
  } as IntegrationRecord
  const config = readRexConfigStrict() as RexConfig
  const explicit = (stored.integrations ?? {}) as IntegrationRecord
  applyOpenClawConfigFallback(integrations, explicit, config)
  integrations.credentialStatus = await hydrateFlatSecretStatus(session, config, integrations)
  const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
  integrations.emailAccounts = await Promise.all(
    accounts.map((raw) => hydrateEmailAccount(session, config, raw))
  )
  return integrations as Settings
}

async function stageFlatSecrets(
  session: ElectronSessionIdentity,
  raw: IntegrationRecord,
  originalConfig: RexConfig,
  nextConfig: RexConfig,
  staged: StagedEntry[],
  replaced: ReplacedEntry[]
): Promise<void> {
  for (const spec of INTEGRATION_SECRET_FIELDS) {
    const oldRecord = getVaultReference(originalConfig, spec.logicalName, spec.context, session.userId)
    const secret = typeof raw[spec.field] === 'string' ? raw[spec.field] as string : ''
    if (secret.trim()) {
      const ref = await vaultSetSecret(session, secret, spec.context)
      staged.push({ logicalName: spec.logicalName, ref, context: spec.context })
      putVaultReference(nextConfig, spec.logicalName, ref, spec.context, session.userId)
      if (oldRecord) replaced.push({ record: oldRecord, context: spec.context })
    }
    raw[spec.field] = ''
  }
}

function buildOriginalAccountMap(stored: SettingsStore): Map<string, IntegrationRecord> {
  const integrations = (stored.integrations ?? {}) as IntegrationRecord
  const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
  const result = new Map<string, IntegrationRecord>()
  for (const candidate of accounts) {
    if (!candidate || typeof candidate !== 'object') continue
    const account = candidate as IntegrationRecord
    result.set(validateAccountId(account.id), account)
  }
  return result
}

function validateUniqueAccountIds(accounts: unknown[]): void {
  const seen = new Set<string>()
  for (const candidate of accounts) {
    if (!candidate || typeof candidate !== 'object') continue
    const id = validateAccountId((candidate as IntegrationRecord).id)
    if (seen.has(id)) throw new SafeValidationError(`Duplicate email account id: ${id}`)
    seen.add(id)
  }
}

async function normalizeEmailAccount(
  session: ElectronSessionIdentity,
  source: IntegrationRecord,
  original: IntegrationRecord | undefined,
  originalConfig: RexConfig,
  nextConfig: RexConfig,
  staged: StagedEntry[],
  replaced: ReplacedEntry[]
): Promise<IntegrationRecord> {
  const account = { ...source }
  const id = validateAccountId(account.id)
  const backend = emailBackend(account)
  const context = emailContext(id, backend)
  const logicalName = `email:${id}`
  const oldContext = emailContext(id, emailBackend(original ?? {}))
  const oldRecord = getVaultReference(originalConfig, logicalName, oldContext, session.userId)
  const field = backend === 'imap' ? 'password' : 'clientSecret'
  const secret = typeof account[field] === 'string' ? account[field] as string : ''
  if (oldRecord && oldContext.slot !== context.slot && !secret.trim()) {
    throw new SafeValidationError('A new credential is required when changing an email account backend')
  }
  if (secret.trim()) {
    const ref = await vaultSetSecret(session, secret, context)
    staged.push({ logicalName, ref, context })
    putVaultReference(nextConfig, logicalName, ref, context, session.userId)
    if (oldRecord) replaced.push({ record: oldRecord, context: oldContext })
  }
  delete account.password
  delete account.clientSecret
  delete account.credentialRef
  delete account.hasCredential
  return account
}

async function normalizeEmailAccounts(
  session: ElectronSessionIdentity,
  accounts: unknown[],
  originalStored: SettingsStore,
  originalConfig: RexConfig,
  nextConfig: RexConfig,
  staged: StagedEntry[],
  replaced: ReplacedEntry[]
): Promise<IntegrationRecord[]> {
  validateUniqueAccountIds(accounts)
  const originals = buildOriginalAccountMap(originalStored)
  const normalized: IntegrationRecord[] = []
  for (const candidate of accounts) {
    if (!candidate || typeof candidate !== 'object') continue
    const source = candidate as IntegrationRecord
    const id = validateAccountId(source.id)
    normalized.push(await normalizeEmailAccount(
      session, source, originals.get(id), originalConfig, nextConfig, staged, replaced
    ))
  }
  return normalized
}

async function normalizeIntegrationSettings(
  session: ElectronSessionIdentity,
  values: Settings,
  originalStored: SettingsStore,
  originalConfig: RexConfig,
  nextConfig: RexConfig,
  staged: StagedEntry[],
  replaced: ReplacedEntry[]
): Promise<Settings> {
  const raw = { ...(values as unknown as IntegrationRecord) }
  delete raw.credentialStatus
  await stageFlatSecrets(session, raw, originalConfig, nextConfig, staged, replaced)
  const accounts = Array.isArray(raw.emailAccounts) ? raw.emailAccounts : []
  raw.emailAccounts = await normalizeEmailAccounts(
    session, accounts, originalStored, originalConfig, nextConfig, staged, replaced
  )
  return raw as Settings
}

async function verifyStagedReferences(
  session: ElectronSessionIdentity,
  staged: StagedEntry[]
): Promise<void> {
  const readback = readRexConfigStrict() as RexConfig
  for (const entry of staged) {
    const record = getVaultReference(readback, entry.logicalName, entry.context, session.userId)
    if (record?.ref !== entry.ref) throw new SafeValidationError('Credential reference readback failed')
  }
}

async function deleteReplacedSecrets(session: ElectronSessionIdentity, replaced: ReplacedEntry[]): Promise<void> {
  for (const old of replaced) {
    await vaultDeleteSecret(session, old.record.ref, old.context).catch(() => false)
  }
}

async function rollbackSettingsWrite(
  session: ElectronSessionIdentity,
  originalStored: SettingsStore,
  originalConfig: RexConfig,
  staged: StagedEntry[],
  guiWritten: boolean,
  configWritten: boolean
): Promise<void> {
  let restored = true
  if (guiWritten) {
    try { writeGuiSettings(originalStored) } catch { restored = false }
  }
  if (configWritten) {
    try { writeRexConfig(originalConfig) } catch { restored = false }
  }
  if (!restored) return
  for (const entry of staged) {
    await vaultDeleteSecret(session, entry.ref, entry.context).catch(() => false)
  }
}

export async function persistSettingsSection(
  session: ElectronSessionIdentity,
  section: string,
  values: Settings
): Promise<{ ok: boolean; error?: string }> {
  const stored = readGuiSettings() as SettingsStore
  const originalStored = JSON.parse(JSON.stringify(stored)) as SettingsStore
  const originalConfig = readRexConfigStrict() as RexConfig
  const nextConfig = JSON.parse(JSON.stringify(originalConfig)) as RexConfig
  const staged: StagedEntry[] = []
  const replaced: ReplacedEntry[] = []
  let guiWritten = false
  let configWritten = false
  try {
    const normalized = section === 'integrations'
      ? await normalizeIntegrationSettings(
          session, values, originalStored, originalConfig, nextConfig, staged, replaced
        )
      : values
    stored[section] = normalized
    writeGuiSettings(stored)
    guiWritten = true
    writeRexConfig(nextConfig)
    configWritten = true
    const mirror = mirrorToRexConfig(section, normalized)
    if (!mirror.ok) throw new Error(mirror.error ?? 'Settings mirror failed')
    await verifyStagedReferences(session, staged)
    if (section === 'integrations') await reconcileIntegrationStatuses(session)
    await deleteReplacedSecrets(session, replaced)
    return { ok: true }
  } catch (err) {
    await rollbackSettingsWrite(
      session, originalStored, originalConfig, staged, guiWritten, configWritten
    )
    return { ok: false, error: safeIpcErrorMessage(err, 'Settings persistence failed') }
  }
}

interface RemovalPlan {
  originalStored: SettingsStore
  originalConfig: RexConfig
  nextStored: SettingsStore
  nextConfig: RexConfig
  integrations: IntegrationRecord
  logicalName: string
  context: VaultContext
  record: VaultReferenceRecord | null
}

function buildRemovalPlan(session: ElectronSessionIdentity, id: string): RemovalPlan | null {
  const originalStored = readGuiSettings() as SettingsStore
  const originalConfig = readRexConfigStrict() as RexConfig
  const nextStored = JSON.parse(JSON.stringify(originalStored)) as SettingsStore
  const nextConfig = JSON.parse(JSON.stringify(originalConfig)) as RexConfig
  const integrations = { ...((nextStored.integrations ?? {}) as IntegrationRecord) }
  const accounts = Array.isArray(integrations.emailAccounts)
    ? integrations.emailAccounts as IntegrationRecord[]
    : []
  const account = accounts.find((candidate) => candidate.id === id)
  if (!account) return null
  const context = emailContext(id, emailBackend(account))
  const logicalName = `email:${id}`
  const record = getVaultReference(originalConfig, logicalName, context, session.userId)
  integrations.emailAccounts = accounts.filter((candidate) => candidate.id !== id)
  nextStored.integrations = integrations as Settings
  if (record) deleteVaultReference(nextConfig, logicalName, context, session.userId)
  return {
    originalStored, originalConfig, nextStored, nextConfig,
    integrations, logicalName, context, record
  }
}

async function executeRemovalPlan(
  session: ElectronSessionIdentity,
  plan: RemovalPlan
): Promise<{ ok: boolean; error?: string }> {
  let guiWritten = false
  let configWritten = false
  try {
    writeGuiSettings(plan.nextStored)
    guiWritten = true
    writeRexConfig(plan.nextConfig)
    configWritten = true
    const mirror = mirrorToRexConfig('integrations', plan.integrations as Settings)
    if (!mirror.ok) throw new Error(mirror.error ?? 'Settings mirror failed')
    const readback = readRexConfigStrict() as RexConfig
    if (getVaultReference(readback, plan.logicalName, plan.context, session.userId)) {
      throw new SafeValidationError('Credential reference deletion readback failed')
    }
    if (plan.record) await vaultDeleteSecret(session, plan.record.ref, plan.context)
    return { ok: true }
  } catch (err) {
    if (guiWritten) {
      try { writeGuiSettings(plan.originalStored) } catch { /* preserve deletion error */ }
    }
    if (configWritten) {
      try { writeRexConfig(plan.originalConfig) } catch { /* preserve deletion error */ }
    }
    return { ok: false, error: safeIpcErrorMessage(err, 'Email account removal failed') }
  }
}

export async function removeEmailAccount(
  session: ElectronSessionIdentity,
  idValue: string,
  confirmed: boolean
): Promise<{ ok: boolean; error?: string }> {
  if (confirmed !== true) {
    return { ok: false, error: 'Email account removal requires confirmation' }
  }
  let id: string
  try {
    id = validateAccountId(idValue)
  } catch (err) {
    return { ok: false, error: safeIpcErrorMessage(err, 'Email account ID is invalid') }
  }
  let plan: RemovalPlan | null
  try {
    plan = buildRemovalPlan(session, id)
  } catch (err) {
    return { ok: false, error: safeIpcErrorMessage(err, 'Credential reference is invalid') }
  }
  if (!plan) return { ok: false, error: 'Email account not found' }
  return executeRemovalPlan(session, plan)
}
