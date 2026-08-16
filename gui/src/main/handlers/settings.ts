import { ipcMain } from 'electron'
import { join } from 'path'
import { homedir } from 'os'
import { existsSync, readFileSync } from 'fs'
import type {
  ModelDiscoveryProvider,
  ModelDiscoveryResponse,
  PreferenceSuggestion,
  Settings,
  WakeWordStatus
} from '../../types/ipc'
import { readGuiSettings, readRexConfigStrict, writeGuiSettings, writeRexConfig } from '../configStore'
import { buildAiSettings, buildAiSettingsForSave } from '../aiSettings'
import { migrateLegacyAutonomySettings, stripLegacyAutonomyMode } from '../autonomySettings'
import { buildVoiceSettings, buildWakeWordStatus } from '../voiceSettings'
import { defaultSettingsMap } from '../settingsDefaults'
import { mirrorToRexConfig } from '../settingsMirror'
import { vaultDeleteSecret, vaultHasSecret, vaultSetSecret, type VaultContext } from '../credentialVault'
import { getVaultReference, putVaultReference } from '../credentialReferences'
import type { ElectronSessionIdentity } from '../sessionIdentity'
import { safeIpcErrorMessage, SafeValidationError } from '../ipcErrors'
import { loadIntegrationSettings, persistSettingsSection, removeEmailAccount } from '../integrationSettingsStorage'
import { discoverAiModelsAtEndpoint } from '../modelDiscovery'

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

function objectSection(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

export function registerSettingsHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getSettings', async (_event, section: string): Promise<Settings> => {
    const stored = section === 'ai' ? migrateLegacyAutonomySettings() : readGuiSettings()
    if (section === 'ai') {
      return buildAiSettings((stored[section] ?? {}) as Settings) as unknown as Settings
    }
    if (section === 'voice') {
      return buildVoiceSettings((stored[section] ?? {}) as Settings) as unknown as Settings
    }
    if (section === 'integrations') {
      return loadIntegrationSettings(session, stored as Record<string, Settings>)
    }
    return stored[section] ?? defaultSettingsMap[section] ?? {}
  })

  ipcMain.handle(
    'rex:discoverAiModels',
    async (_event, provider: ModelDiscoveryProvider | string): Promise<ModelDiscoveryResponse> => {
      if (provider !== 'ollama' && provider !== 'lmstudio') {
        return { ok: false, models: [], error: 'Unsupported model discovery provider' }
      }
      try {
        const config = readRexConfigStrict()
        const section = objectSection(provider === 'ollama' ? config.ollama : config.openai)
        const endpoint = typeof section.base_url === 'string' ? section.base_url : ''
        return discoverAiModelsAtEndpoint(provider, endpoint)
      } catch {
        return {
          ok: false,
          models: [],
          error: 'Model discovery configuration could not be read'
        }
      }
    }
  )

  ipcMain.handle('rex:getWakeWordStatus', (_event, values?: Settings): WakeWordStatus => {
    const stored = readGuiSettings()
    const source = values ?? ((stored.voice ?? {}) as Settings)
    return buildWakeWordStatus(source)
  })

  ipcMain.handle(
    'rex:setSettings',
    async (_event, section: string, values: Settings): Promise<{ ok: boolean; error?: string }> => {
      const normalizedValues =
        section === 'ai'
          ? (buildAiSettingsForSave(values) as unknown as Settings)
          : section === 'voice'
            ? (buildVoiceSettings(values) as unknown as Settings)
            : values
      return persistSettingsSection(session, section, normalizedValues)
    }
  )

  ipcMain.handle(
    'rex:removeEmailAccount',
    async (
      _event,
      idValue: string,
      confirmed: boolean
    ): Promise<{ ok: boolean; error?: string }> => removeEmailAccount(session, idValue, confirmed)
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
    const aiSettings = buildAiSettings((stored['ai'] ?? defaultSettingsMap['ai'] ?? {}) as Settings)

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
      stored['ai'] = stripLegacyAutonomyMode(aiSection as Settings)
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
