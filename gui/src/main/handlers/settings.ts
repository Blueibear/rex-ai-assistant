import { ipcMain } from 'electron'
import { join } from 'path'
import { homedir } from 'os'
import { existsSync, readFileSync } from 'fs'
import type { AiSettings, PreferenceSuggestion, Settings, WakeWordStatus } from '../../types/ipc'
import { readGuiSettings, readRexConfigStrict, writeGuiSettings, writeRexConfig } from '../configStore'
import { buildAiSettings } from '../aiSettings'
import { buildVoiceSettings, buildWakeWordStatus } from '../voiceSettings'
import { defaultSettingsMap } from '../settingsDefaults'
import { mirrorToRexConfig } from '../settingsMirror'
import { reconcileIntegrationStatuses } from '../integrationStatus'
import { vaultDeleteSecret, vaultHasSecret, vaultSetSecret, type VaultContext } from '../credentialVault'
import {
  deleteVaultReference,
  getVaultReference,
  putVaultReference,
  type VaultReferenceRecord
} from '../credentialReferences'
import type { ElectronSessionIdentity } from '../sessionIdentity'
import { safeIpcErrorMessage, SafeValidationError } from '../ipcErrors'

const ALLOWED_API_KEYS = [
  'OPENAI_API_KEY',
  'OPENROUTER_API_KEY',
  'ANTHROPIC_API_KEY',
  'OLLAMA_API_KEY',
  'ELEVENLABS_API_KEY',
  'SERPAPI_KEY',
  'SERPAPI_API_KEY',
  'BRAVE_API_KEY',
  'GOOGLE_API_KEY',
  'OPENWEATHERMAP_API_KEY',
  'REX_SPEAK_API_KEY',
  'OPENCLAW_GATEWAY_TOKEN'
] // pragma: allowlist secret

interface SecretFieldSpec {
  field: string
  logicalName: string
  context: VaultContext
}

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
  { field: 'telegramBotToken', logicalName: 'TELEGRAM_BOT_TOKEN', context: { scope: 'household', integration: 'telegram', account: null, slot: 'token' } }
]

/** Derive a short integration name for vault metadata, e.g. OPENAI_API_KEY -> openai. */
function integrationNameForKey(key: string): string {
  return key.replace(/_API_KEY$|_KEY$/, '').toLowerCase()
}

function apiKeyContext(key: string): VaultContext {
  if (key === 'OPENCLAW_GATEWAY_TOKEN') {
    return { scope: 'household', integration: 'openclaw_gateway', account: null, slot: 'token' }
  }
  return { scope: 'household', integration: integrationNameForKey(key), account: null, slot: 'api_key' }
}

function validateAccountId(value: unknown): string {
  const id = typeof value === 'string' ? value : ''
  if (!/^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(id)) {
    throw new Error('Email account ID is invalid')
  }
  return id
}

export function registerSettingsHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getSettings', async (_event, section: string): Promise<Settings> => {
    const stored = readGuiSettings()
    if (section === 'ai') {
      return buildAiSettings((stored[section] ?? {}) as Settings) as unknown as Settings
    }
    if (section === 'voice') {
      return buildVoiceSettings((stored[section] ?? {}) as Settings) as unknown as Settings
    }
    if (section !== 'integrations') return stored[section] ?? defaultSettingsMap[section] ?? {}

    const integrations = {
      ...defaultSettingsMap.integrations,
      ...((stored.integrations ?? {}) as Record<string, unknown>)
    } as Record<string, unknown>
    const config = readRexConfigStrict()
    const credentialStatus: Record<string, { ref: string; hasCredential: boolean }> = {}
    for (const spec of INTEGRATION_SECRET_FIELDS) {
      integrations[spec.field] = ''
      const record = getVaultReference(config, spec.logicalName, spec.context, session.userId)
      if (record) {
        credentialStatus[spec.field] = {
          ref: record.ref,
          hasCredential: await vaultHasSecret(session, record.ref, spec.context)
        }
      }
    }
    const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
    integrations.emailAccounts = await Promise.all(accounts.map(async (raw) => {
      const account: Record<string, unknown> = {
        ...(raw as Record<string, unknown>),
        password: '',
        clientSecret: ''
      }
      const id = validateAccountId(account.id)
      const backend = account.backend === 'imap' ? 'imap' : account.backend === 'outlook' ? 'outlook' : 'gmail'
      const logicalName = `email:${id}`
      const context: VaultContext = { scope: 'user', integration: 'email', account: id, slot: backend === 'imap' ? 'password' : 'client_secret' }
      const record = getVaultReference(config, logicalName, context, session.userId)
      if (record) {
        account.credentialRef = record.ref
        account.hasCredential = await vaultHasSecret(session, record.ref, context)
      } else {
        account.hasCredential = false
      }
      return account
    }))
    integrations.credentialStatus = credentialStatus
    return integrations as Settings
  })

  ipcMain.handle('rex:getWakeWordStatus', (_event, values?: Settings): WakeWordStatus => {
    const stored = readGuiSettings()
    const source = values ?? ((stored.voice ?? {}) as Settings)
    return buildWakeWordStatus(source)
  })

  ipcMain.handle(
    'rex:setSettings',
    async (_event, section: string, values: Settings): Promise<{ ok: boolean; error?: string }> => {
      const stored = readGuiSettings()
      const originalStored = JSON.parse(JSON.stringify(stored)) as Record<string, Settings>
      const originalConfig = readRexConfigStrict()
      const nextConfig = JSON.parse(JSON.stringify(originalConfig)) as Record<string, unknown>
      let normalizedValues =
        section === 'ai'
          ? (buildAiSettings(values) as unknown as Settings)
          : section === 'voice'
            ? (buildVoiceSettings(values) as unknown as Settings)
            : values
      const newEntries: Array<{ logicalName: string; ref: string; context: VaultContext }> = []
      const replacedEntries: Array<{ record: VaultReferenceRecord; context: VaultContext }> = []
      let guiWritten = false
      let configWritten = false
      try {
        if (section === 'integrations') {
          const raw = { ...(normalizedValues as unknown as Record<string, unknown>) }
          delete raw.credentialStatus
          for (const spec of INTEGRATION_SECRET_FIELDS) {
            const oldRecord = getVaultReference(
              originalConfig, spec.logicalName, spec.context, session.userId
            )
            const secret = typeof raw[spec.field] === 'string' ? (raw[spec.field] as string) : ''
            if (secret.trim()) {
              const ref = await vaultSetSecret(session, secret, spec.context)
              newEntries.push({ logicalName: spec.logicalName, ref, context: spec.context })
              putVaultReference(nextConfig, spec.logicalName, ref, spec.context, session.userId)
              if (oldRecord) replacedEntries.push({ record: oldRecord, context: spec.context })
            }
            raw[spec.field] = ''
          }

          const accounts = Array.isArray(raw.emailAccounts) ? raw.emailAccounts : []
          const originalIntegrations = (originalStored.integrations ?? {}) as Record<string, unknown>
          const originalAccounts = Array.isArray(originalIntegrations.emailAccounts)
            ? originalIntegrations.emailAccounts
            : []
          const originalById = new Map<string, Record<string, unknown>>()
          for (const candidate of originalAccounts) {
            if (!candidate || typeof candidate !== 'object') continue
            const originalAccount = candidate as Record<string, unknown>
            originalById.set(validateAccountId(originalAccount.id), originalAccount)
          }
          raw.emailAccounts = []
          const seenAccountIds = new Set<string>()
          for (const rawAccount of accounts) {
            if (!rawAccount || typeof rawAccount !== 'object') continue
            const id = validateAccountId((rawAccount as Record<string, unknown>).id)
            if (seenAccountIds.has(id)) throw new SafeValidationError(`Duplicate email account id: ${id}`)
            seenAccountIds.add(id)
          }
          for (const rawAccount of accounts) {
            if (!rawAccount || typeof rawAccount !== 'object') continue
            const account = { ...(rawAccount as Record<string, unknown>) }
            const id = validateAccountId(account.id)
            const backend = account.backend === 'imap' ? 'imap' : account.backend === 'outlook' ? 'outlook' : 'gmail'
            const context: VaultContext = {
              scope: 'user', integration: 'email', account: id,
              slot: backend === 'imap' ? 'password' : 'client_secret'
            }
            const logicalName = `email:${id}`
            const originalAccount = originalById.get(id)
            const oldBackend = originalAccount?.backend === 'imap'
              ? 'imap'
              : originalAccount?.backend === 'outlook' ? 'outlook' : 'gmail'
            const oldContext: VaultContext = {
              scope: 'user', integration: 'email', account: id,
              slot: oldBackend === 'imap' ? 'password' : 'client_secret'
            }
            const oldRecord = getVaultReference(
              originalConfig, logicalName, oldContext, session.userId
            )
            const field = backend === 'imap' ? 'password' : 'clientSecret'
            const secret = typeof account[field] === 'string' ? (account[field] as string) : ''
            if (oldRecord && oldContext.slot !== context.slot && !secret.trim()) {
              throw new SafeValidationError(
                'A new credential is required when changing an email account backend'
              )
            }
            if (secret.trim()) {
              const ref = await vaultSetSecret(session, secret, context)
              newEntries.push({ logicalName, ref, context })
              putVaultReference(nextConfig, logicalName, ref, context, session.userId)
              if (oldRecord) replacedEntries.push({ record: oldRecord, context: oldContext })
            }
            delete account.password
            delete account.clientSecret
            delete account.credentialRef
            delete account.hasCredential
            ;(raw.emailAccounts as Record<string, unknown>[]).push(account)
          }
          normalizedValues = raw as Settings
        }

        stored[section] = normalizedValues
        writeGuiSettings(stored)
        guiWritten = true
        writeRexConfig(nextConfig)
        configWritten = true
        const mirrorResult = mirrorToRexConfig(section, normalizedValues)
        if (!mirrorResult.ok) throw new Error(mirrorResult.error ?? 'Settings mirror failed')
        const readback = readRexConfigStrict()
        for (const entry of newEntries) {
          const record = getVaultReference(
            readback, entry.logicalName, entry.context, session.userId
          )
          if (record?.ref !== entry.ref) throw new SafeValidationError('Credential reference readback failed')
        }
        if (section === 'integrations') await reconcileIntegrationStatuses(session)
        for (const old of replacedEntries) {
          await vaultDeleteSecret(session, old.record.ref, old.context).catch(() => false)
        }
        return { ok: true }
      } catch (err) {
        let restored = true
        if (guiWritten) {
          try { writeGuiSettings(originalStored) } catch { restored = false }
        }
        if (configWritten) {
          try { writeRexConfig(originalConfig) } catch { restored = false }
        }
        if (restored) {
          for (const entry of newEntries) {
            await vaultDeleteSecret(session, entry.ref, entry.context).catch(() => false)
          }
        }
        return { ok: false, error: safeIpcErrorMessage(err, 'Settings persistence failed') }
      }
    }
  )

  ipcMain.handle(
    'rex:removeEmailAccount',
    async (
      _event,
      idValue: string,
      confirmed: boolean
    ): Promise<{ ok: boolean; error?: string }> => {
      if (confirmed !== true) {
        return { ok: false, error: 'Email account removal requires confirmation' }
      }
      let id: string
      try {
        id = validateAccountId(idValue)
      } catch (err) {
        return { ok: false, error: safeIpcErrorMessage(err, 'Email account ID is invalid') }
      }

      const originalStored = readGuiSettings()
      const originalConfig = readRexConfigStrict()
      const nextStored = JSON.parse(JSON.stringify(originalStored)) as Record<string, Settings>
      const nextConfig = JSON.parse(JSON.stringify(originalConfig)) as Record<string, unknown>
      const integrations = {
        ...((nextStored.integrations ?? {}) as Record<string, unknown>)
      }
      const accounts = Array.isArray(integrations.emailAccounts)
        ? integrations.emailAccounts as Array<Record<string, unknown>>
        : []
      const account = accounts.find((candidate) => candidate.id === id)
      if (!account) return { ok: false, error: 'Email account not found' }
      const backend = account.backend === 'imap'
        ? 'imap'
        : account.backend === 'outlook' ? 'outlook' : 'gmail'
      const context: VaultContext = {
        scope: 'user', integration: 'email', account: id,
        slot: backend === 'imap' ? 'password' : 'client_secret'
      }
      const logicalName = `email:${id}`
      let record: VaultReferenceRecord | null
      try {
        record = getVaultReference(originalConfig, logicalName, context, session.userId)
      } catch (err) {
        return { ok: false, error: safeIpcErrorMessage(err, 'Credential reference is invalid') }
      }
      integrations.emailAccounts = accounts.filter((candidate) => candidate.id !== id)
      nextStored.integrations = integrations as Settings
      if (record) deleteVaultReference(nextConfig, logicalName, context, session.userId)

      let guiWritten = false
      let configWritten = false
      try {
        writeGuiSettings(nextStored)
        guiWritten = true
        writeRexConfig(nextConfig)
        configWritten = true
        const mirror = mirrorToRexConfig('integrations', integrations as Settings)
        if (!mirror.ok) throw new Error(mirror.error ?? 'Settings mirror failed')
        if (getVaultReference(readRexConfigStrict(), logicalName, context, session.userId)) {
          throw new SafeValidationError('Credential reference deletion readback failed')
        }
        if (record) await vaultDeleteSecret(session, record.ref, context)
        return { ok: true }
      } catch (err) {
        if (guiWritten) {
          try { writeGuiSettings(originalStored) } catch { /* preserve deletion error */ }
        }
        if (configWritten) {
          try { writeRexConfig(originalConfig) } catch { /* preserve deletion error */ }
        }
        return { ok: false, error: safeIpcErrorMessage(err, 'Email account removal failed') }
      }
    }
  )

  ipcMain.handle('rex:testVoice', () => {
    // Stub: in production this would invoke the TTS engine with a test phrase
    return { ok: true }
  })

  ipcMain.handle('rex:getPreferenceSuggestions', (): PreferenceSuggestion[] => {
    const prefsPath = join(homedir(), '.rex', 'preferences.json')
    let profile: Record<string, unknown> = {}
    try {
      if (existsSync(prefsPath)) {
        profile = JSON.parse(readFileSync(prefsPath, 'utf8')) as Record<string, unknown>
      }
    } catch {
      return []
    }

    const stored = readGuiSettings()
    const aiSettings = (stored['ai'] ?? defaultSettingsMap['ai'] ?? {}) as unknown as AiSettings

    const suggestions: PreferenceSuggestion[] = []

    // Autonomy mode - highest impact
    const preferredMode =
      typeof profile.preferred_autonomy_mode === 'string' ? profile.preferred_autonomy_mode : null
    if (preferredMode && preferredMode !== aiSettings.autonomyMode) {
      suggestions.push({
        field: 'autonomyMode',
        current_value: aiSettings.autonomyMode,
        suggested_value: preferredMode,
        reason: `You typically run Rex in "${preferredMode}" mode`
      })
    }

    // Model
    const preferredModel =
      typeof profile.preferred_model === 'string' && profile.preferred_model
        ? profile.preferred_model
        : null
    if (preferredModel && preferredModel !== aiSettings.model) {
      suggestions.push({
        field: 'model',
        current_value: aiSettings.model,
        suggested_value: preferredModel,
        reason: `You most frequently use ${preferredModel}`
      })
    }

    // Budget - suggest 2x avg if no budget is set
    const avgBudget =
      typeof profile.avg_budget_usd === 'number' ? profile.avg_budget_usd : 0
    if (avgBudget > 0 && aiSettings.budgetPerPlan === 0) {
      const suggested = Math.round(avgBudget * 2 * 100) / 100
      suggestions.push({
        field: 'budgetPerPlan',
        current_value: aiSettings.budgetPerPlan,
        suggested_value: suggested,
        reason: `Your average plan cost is $${avgBudget.toFixed(2)} - a $${suggested.toFixed(2)} budget would prevent overruns`
      })
    }

    return suggestions
  })

  ipcMain.handle(
    'rex:applyPreferenceSuggestion',
    (_event, field: string, value: string | number) => {
      const stored = readGuiSettings()
      const originalStored = JSON.parse(JSON.stringify(stored)) as Record<string, Settings>
      const aiSection = buildAiSettings((stored['ai'] ?? defaultSettingsMap['ai'] ?? {}) as Settings) as unknown as Record<string, unknown>
      aiSection[field] = value
      stored['ai'] = aiSection as Settings
      writeGuiSettings(stored)
      const result = mirrorToRexConfig('ai', aiSection as Settings)
      if (result.ok) return { ok: true }
      try { writeGuiSettings(originalStored) } catch { /* preserve mirror error */ }
      return { ok: false, error: result.error }
    }
  )

  ipcMain.handle(
    'rex:getApiKeys',
    async (): Promise<{ openai_key_set: boolean; openrouter_key_set: boolean; error?: string }> => {
      try {
        const config = readRexConfigStrict()
        const hasKey = async (name: 'OPENAI_API_KEY' | 'OPENROUTER_API_KEY'): Promise<boolean> => {
          const context = apiKeyContext(name)
          const record = getVaultReference(config, name, context, session.userId)
          return record ? await vaultHasSecret(session, record.ref, context) : false
        }
        const [openaiKeySet, openrouterKeySet] = await Promise.all([
          hasKey('OPENAI_API_KEY'),
          hasKey('OPENROUTER_API_KEY')
        ])
        return { openai_key_set: openaiKeySet, openrouter_key_set: openrouterKeySet }
      } catch {
        // Vault unavailable: fail closed and report "not configured" rather
        // than reading any legacy plaintext credential location (S4).
        return {
          openai_key_set: false,
          openrouter_key_set: false,
          error: 'Stored API-key state could not be verified'
        }
      }
    }
  )

  ipcMain.handle(
    'rex:setApiKey',
    async (_event, name: string, value: string): Promise<{ ok: boolean; error?: string }> => {
      // Validate key name to prevent arbitrary vault writes
      if (!ALLOWED_API_KEYS.includes(name)) {
        return { ok: false, error: `Key "${name}" is not allowed` }
      }
      if (!value.trim()) return { ok: true }
      const context = apiKeyContext(name)
      let newRef: string | null = null
      let originalConfig: Record<string, unknown> | null = null
      let configWritten = false
      try {
        originalConfig = readRexConfigStrict()
        const nextConfig = JSON.parse(JSON.stringify(originalConfig)) as Record<string, unknown>
        const oldRecord = getVaultReference(originalConfig, name, context, session.userId)
        newRef = await vaultSetSecret(session, value, context)
        putVaultReference(nextConfig, name, newRef, context, session.userId)
        writeRexConfig(nextConfig)
        configWritten = true
        const readback = getVaultReference(
          readRexConfigStrict(), name, context, session.userId
        )
        if (readback?.ref !== newRef) throw new SafeValidationError('Credential reference readback failed')
        if (oldRecord) {
          await vaultDeleteSecret(session, oldRecord.ref, context).catch(() => false)
        }
        return { ok: true }
      } catch (err) {
        let restored = true
        if (configWritten && originalConfig) {
          try { writeRexConfig(originalConfig) } catch { restored = false }
        }
        if (newRef && restored) {
          await vaultDeleteSecret(session, newRef, context).catch(() => false)
        }
        return { ok: false, error: safeIpcErrorMessage(err, 'API key persistence failed') }
      }
    }
  )
}
