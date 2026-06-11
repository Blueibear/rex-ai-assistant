import { ipcMain } from 'electron'
import { join } from 'path'
import { homedir } from 'os'
import { existsSync, readFileSync } from 'fs'
import type { AiSettings, PreferenceSuggestion, Settings, WakeWordStatus } from '../../types/ipc'
import { readEnvFile, readGuiSettings, writeEnvKey, writeGuiSettings } from '../configStore'
import { buildAiSettings } from '../aiSettings'
import { buildVoiceSettings, buildWakeWordStatus } from '../voiceSettings'
import { defaultSettingsMap } from '../settingsDefaults'
import { mirrorToRexConfig } from '../settingsMirror'
import { reconcileIntegrationStatuses } from '../integrationStatus'

export function registerSettingsHandlers(): void {
  ipcMain.handle('rex:getSettings', (_event, section: string): Settings => {
    const stored = readGuiSettings()
    if (section === 'ai') {
      return buildAiSettings((stored[section] ?? {}) as Settings) as unknown as Settings
    }
    if (section === 'voice') {
      return buildVoiceSettings((stored[section] ?? {}) as Settings) as unknown as Settings
    }
    return stored[section] ?? defaultSettingsMap[section] ?? {}
  })

  ipcMain.handle('rex:getWakeWordStatus', (_event, values?: Settings): WakeWordStatus => {
    const stored = readGuiSettings()
    const source = values ?? ((stored.voice ?? {}) as Settings)
    return buildWakeWordStatus(source)
  })

  ipcMain.handle('rex:setSettings', (_event, section: string, values: Settings) => {
    const stored = readGuiSettings()
    let normalizedValues =
      section === 'ai'
        ? (buildAiSettings(values) as unknown as Settings)
        : section === 'voice'
          ? (buildVoiceSettings(values) as unknown as Settings)
          : values
    // Secrets must not be stored in gui_settings - redirect HA token to .env
    if (section === 'integrations') {
      const raw = normalizedValues as unknown as Record<string, unknown>
      const haToken = raw['haToken']
      if (typeof haToken === 'string' && haToken.trim()) {
        writeEnvKey('HA_TOKEN', haToken.trim())
      }
      normalizedValues = { ...raw, haToken: '' } as unknown as Settings
    }
    stored[section] = normalizedValues
    writeGuiSettings(stored)
    // Mirror relevant fields to rex_config.json so the Python backend picks them up
    mirrorToRexConfig(section, normalizedValues)
    if (section === 'integrations') {
      reconcileIntegrationStatuses()
    }
    return { ok: true }
  })

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
      const aiSection = buildAiSettings((stored['ai'] ?? defaultSettingsMap['ai'] ?? {}) as Settings) as unknown as Record<string, unknown>
      aiSection[field] = value
      stored['ai'] = aiSection as Settings
      writeGuiSettings(stored)
      mirrorToRexConfig('ai', aiSection as Settings)
      return { ok: true }
    }
  )

  ipcMain.handle('rex:getApiKeys', () => {
    const env = readEnvFile()
    return {
      openai_key_set: typeof env['OPENAI_API_KEY'] === 'string' && env['OPENAI_API_KEY'].trim() !== '' // pragma: allowlist secret
    }
  })

  ipcMain.handle(
    'rex:setApiKey',
    (_event, name: string, value: string): { ok: boolean; error?: string } => {
      try {
        // Validate key name to prevent arbitrary env writes
        const allowedKeys = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY', 'SERPAPI_KEY', 'BRAVE_API_KEY'] // pragma: allowlist secret
        if (!allowedKeys.includes(name)) {
          return { ok: false, error: `Key "${name}" is not allowed` }
        }
        writeEnvKey(name, value)
        return { ok: true }
      } catch (err) {
        return { ok: false, error: err instanceof Error ? err.message : String(err) }
      }
    }
  )
}
