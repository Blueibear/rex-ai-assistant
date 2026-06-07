import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { basename, dirname, join } from 'path'
import { homedir } from 'os'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { createTray, destroyTray } from './tray'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import { createHash } from 'crypto'
import type {
  Settings,
  GeneralSettings,
  VoiceSettings,
  WakeWordBackend,
  WakeWordStatus,
  AiSettings,
  IntegrationsSettings,
  EmailAccount,
  PreferenceSuggestion,
  SystemSettings,
  IntegrationConnectionStatus,
  IntegrationInventoryItem,
  CapabilityInfo
} from '../types/ipc'
import { registerChatHandlers } from './handlers/chat'
import { getCurrentVoiceState, registerVoiceHandlers } from './handlers/voice'
import { registerTaskHandlers } from './handlers/tasks'
import { registerCalendarHandlers } from './handlers/calendar'
import { registerRemindersHandlers } from './handlers/reminders'
import { registerMemoriesHandlers } from './handlers/memories'
import { registerEmailHandlers } from './handlers/email'
import { registerSMSHandlers } from './handlers/sms'
import { registerNotificationHandlers } from './handlers/notifications'
import { registerSpeakerHandlers } from './handlers/speakers'
import { registerFileHandlers } from './handlers/files'
import { registerShoppingHandlers } from './handlers/shopping'
import { appendElectronLog, registerLogsHandlers, writeElectronSessionStart } from './handlers/logs'
import { registerUsageHandlers } from './handlers/usage'
import { validateBridges } from './bridgeResolver'

// ---------------------------------------------------------------------------
// Config file helpers
// ---------------------------------------------------------------------------

function getConfigDir(): string {
  // app.getAppPath() returns the gui/ directory in dev; ../config is the Rex config dir
  return join(app.getAppPath(), '..', 'config')
}

function getGuiSettingsPath(): string {
  return join(getConfigDir(), 'gui_settings.json')
}

function getRexConfigPath(): string {
  return join(getConfigDir(), 'rex_config.json')
}

function readGuiSettings(): Record<string, Settings> {
  try {
    const p = getGuiSettingsPath()
    if (!existsSync(p)) return {}
    return JSON.parse(readFileSync(p, 'utf8')) as Record<string, Settings>
  } catch {
    return {}
  }
}

function writeGuiSettings(settings: Record<string, Settings>): void {
  const dir = getConfigDir()
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(getGuiSettingsPath(), JSON.stringify(settings, null, 2), 'utf8')
}

function readRexConfig(): Record<string, unknown> {
  try {
    const p = getRexConfigPath()
    if (!existsSync(p)) return {}
    return JSON.parse(readFileSync(p, 'utf8')) as Record<string, unknown>
  } catch {
    return {}
  }
}

function writeRexConfig(config: Record<string, unknown>): void {
  const dir = getConfigDir()
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(getRexConfigPath(), JSON.stringify(config, null, 2), 'utf8')
}

// ---------------------------------------------------------------------------
// .env file helpers (API keys)
// ---------------------------------------------------------------------------

function getEnvFilePath(): string {
  return join(app.getAppPath(), '..', '.env')
}

function readEnvFile(): Record<string, string> {
  try {
    const p = getEnvFilePath()
    if (!existsSync(p)) return {}
    const lines = readFileSync(p, 'utf8').split('\n')
    const result: Record<string, string> = {}
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) continue
      const eq = trimmed.indexOf('=')
      if (eq === -1) continue
      const key = trimmed.slice(0, eq).trim()
      const val = trimmed.slice(eq + 1).trim()
      result[key] = val
    }
    return result
  } catch {
    return {}
  }
}

function writeEnvKey(name: string, value: string): void {
  const p = getEnvFilePath()
  let lines: string[] = []
  try {
    if (existsSync(p)) {
      lines = readFileSync(p, 'utf8').split('\n')
    }
  } catch {
    lines = []
  }
  const keyPrefix = `${name}=`
  const newLine = `${name}=${value}`
  let found = false
  lines = lines.map((line) => {
    if (line.startsWith(keyPrefix) || line.trim().startsWith(keyPrefix)) {
      found = true
      return newLine
    }
    return line
  })
  if (!found) {
    lines.push(newLine)
  }
  // Trim trailing empty lines then add single newline at end
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop()
  writeFileSync(p, lines.join('\n') + '\n', 'utf8')
}

interface HaTestResult {
  ok: boolean
  error?: string
}

interface HaState {
  entity_id: string
  state: string
  friendly_name: string
  last_updated: string
}

interface HaStatesResult extends HaTestResult {
  states?: HaState[]
  not_configured?: boolean
}

function normalizeHaUrl(value: unknown): string {
  return typeof value === 'string' ? value.trim().replace(/\/+$/, '') : ''
}

function readSavedHomeAssistantCredentials(): { baseUrl: string; token: string } {
  const stored = readGuiSettings()
  const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
  const rexConfig = readRexConfig()
  const haConfig = ((rexConfig.home_assistant ?? {}) as Record<string, unknown>)
  const env = readEnvFile()

  return {
    baseUrl: normalizeHaUrl(integrations.haUrl) || normalizeHaUrl(haConfig.base_url),
    // HA token is stored in .env only - never in gui_settings (canonical secret store)
    token: (typeof env.HA_TOKEN === 'string' && env.HA_TOKEN.trim()) || ''
  }
}

function saveHomeAssistantCredentials(baseUrl: string, token: string): void {
  const stored = readGuiSettings()
  const integrations = { ...((stored['integrations'] ?? {}) as Record<string, unknown>) }
  integrations.haUrl = baseUrl
  // haToken is NOT stored in gui_settings - canonical secret store is .env only
  stored['integrations'] = integrations as Settings
  writeGuiSettings(stored)
  mirrorToRexConfig('integrations', integrations as Settings)
  if (token) {
    writeEnvKey('HA_TOKEN', token)
  }
}

function describeHaError(error: unknown): string {
  if (error instanceof Error) {
    return error.name === 'AbortError' ? 'Connection timed out.' : error.message
  }
  return String(error)
}

async function requestHomeAssistant(
  baseUrl: string,
  token: string,
  path: string,
  timeoutMs = 5000
): Promise<Response> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(`${baseUrl}${path}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      signal: controller.signal
    })
  } finally {
    clearTimeout(timeout)
  }
}

async function testHomeAssistantConnection(baseUrl: string, token: string): Promise<HaTestResult> {
  const normalizedUrl = normalizeHaUrl(baseUrl)
  if (!normalizedUrl) return { ok: false, error: 'Home Assistant URL is required.' }
  try {
    const resp = await requestHomeAssistant(normalizedUrl, token.trim(), '/api/')
    if (!resp.ok) return { ok: false, error: `HA returned HTTP ${resp.status}` }
    return { ok: true }
  } catch (err) {
    return { ok: false, error: describeHaError(err) }
  }
}

async function getHomeAssistantStates(): Promise<HaStatesResult> {
  const { baseUrl, token } = readSavedHomeAssistantCredentials()
  if (!baseUrl || !token) {
    return {
      ok: false,
      not_configured: true,
      error: 'Home Assistant is not configured.'
    }
  }
  try {
    const resp = await requestHomeAssistant(baseUrl, token, '/api/states', 10000)
    if (!resp.ok) return { ok: false, error: `HA returned HTTP ${resp.status}` }
    const rawStates = (await resp.json()) as Array<Record<string, unknown>>
    const states = rawStates.filter((s) => typeof s === 'object' && s !== null).map((s) => {
      const attrs = s.attributes && typeof s.attributes === 'object'
        ? (s.attributes as Record<string, unknown>)
        : {}
      const entityId = typeof s.entity_id === 'string' ? s.entity_id : ''
      return {
        entity_id: entityId,
        state: typeof s.state === 'string' ? s.state : 'unknown',
        friendly_name: typeof attrs.friendly_name === 'string' ? attrs.friendly_name : entityId,
        last_updated: typeof s.last_updated === 'string' ? s.last_updated : ''
      }
    })
    return { ok: true, states }
  } catch (err) {
    return { ok: false, error: describeHaError(err) }
  }
}

function normalizeAiModelRouting(raw: unknown): AiSettings['modelRouting'] {
  const source = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  return {
    default: typeof source.default === 'string' ? source.default : '',
    coding: typeof source.coding === 'string' ? source.coding : '',
    reasoning: typeof source.reasoning === 'string' ? source.reasoning : '',
    search: typeof source.search === 'string' ? source.search : '',
    vision: typeof source.vision === 'string' ? source.vision : '',
    fast: typeof source.fast === 'string' ? source.fast : ''
  }
}

function normalizeWakeWordBackend(raw: unknown): WakeWordBackend {
  return raw === 'custom_onnx' || raw === 'custom_embedding' ? raw : 'openwakeword'
}

function normalizeWakeWordId(raw: unknown): string {
  if (typeof raw !== 'string') return ''
  return raw.trim().replace(/\s+/g, '_').toLowerCase()
}

function wakeWordIdToPhrase(raw: unknown): string {
  return typeof raw === 'string' ? raw.trim().replace(/_/g, ' ') : ''
}

function inferCustomWakeWordIdFromPath(assetPath: string): string {
  if (!assetPath) return ''
  const parent = basename(dirname(assetPath))
  if (parent && parent !== 'wake_words') return parent
  return basename(assetPath).replace(/\.[^.]+$/, '')
}

function defaultCustomWakeWordAssetPath(
  backend: WakeWordBackend,
  phraseOrId: string
): string {
  const slug = normalizeWakeWordId(phraseOrId || 'hey rex') || 'hey_rex'
  const baseDir = join(getConfigDir(), 'wake_words', slug)
  return backend === 'custom_onnx'
    ? join(baseDir, 'model.onnx')
    : join(baseDir, 'embedding.pt')
}

function buildVoiceSettings(raw: Settings = {}): VoiceSettings {
  const rexConfig = readRexConfig()
  const models =
    rexConfig.models && typeof rexConfig.models === 'object'
      ? (rexConfig.models as Record<string, unknown>)
      : {}
  const wakeword =
    rexConfig.wakeword && typeof rexConfig.wakeword === 'object'
      ? (rexConfig.wakeword as Record<string, unknown>)
      : {}

  const rawEngine =
    typeof raw.ttsEngine === 'string'
      ? raw.ttsEngine
      : typeof models.tts_provider === 'string'
        ? models.tts_provider
        : null
  const ttsEngine: VoiceSettings['ttsEngine'] =
    rawEngine === 'xtts' || rawEngine === 'edge-tts' || rawEngine === 'pyttsx3'
      ? rawEngine
      : rawEngine === 'elevenlabs'
        ? 'xtts'
        : rawEngine === 'openai'
          ? 'edge-tts'
          : 'pyttsx3'

  const rawSttDevice =
    typeof raw.sttDevice === 'string'
      ? raw.sttDevice
      : typeof models.whisper_device === 'string'
        ? models.whisper_device
        : null
  const sttDevice: VoiceSettings['sttDevice'] =
    rawSttDevice === 'cpu' || rawSttDevice === 'cuda' ? rawSttDevice : 'auto'

  const backend = normalizeWakeWordBackend(
    raw.wakeWordBackend ?? wakeword.backend
  )
  const configuredPhrase =
    typeof wakeword.wakeword === 'string' ? wakeword.wakeword.trim() : ''
  const configuredKeyword =
    typeof wakeword.keyword === 'string' ? wakeword.keyword.trim() : ''
  const configuredFallbackKeyword =
    typeof wakeword.fallback_keyword === 'string' ? wakeword.fallback_keyword.trim() : ''
  const legacyWakeWordModel =
    typeof wakeword.model === 'string' ? wakeword.model.trim() : ''

  const wakeWord = normalizeWakeWordId(
    typeof raw.wakeWord === 'string' && raw.wakeWord.trim()
      ? raw.wakeWord
      : backend === 'openwakeword'
        ? configuredKeyword || configuredPhrase || legacyWakeWordModel
        : configuredFallbackKeyword
  )

  const rawPhrase =
    typeof raw.wakeWordPhrase === 'string' && raw.wakeWordPhrase.trim()
      ? raw.wakeWordPhrase.trim()
      : configuredPhrase
  const embeddingPath =
    typeof raw.wakeWordEmbeddingPath === 'string' && raw.wakeWordEmbeddingPath.trim()
      ? raw.wakeWordEmbeddingPath.trim()
      : typeof wakeword.embedding_path === 'string'
        ? wakeword.embedding_path
        : ''
  const customWakeWordId =
    typeof raw.customWakeWordId === 'string' && raw.customWakeWordId.trim()
      ? raw.customWakeWordId.trim()
      : inferCustomWakeWordIdFromPath(embeddingPath)
  const wakeWordPhrase =
    rawPhrase || wakeWordIdToPhrase(customWakeWordId || wakeWord) || 'hey rex'
  const modelPath =
    typeof raw.wakeWordModelPath === 'string' && raw.wakeWordModelPath.trim()
      ? raw.wakeWordModelPath.trim()
      : typeof wakeword.model_path === 'string'
        ? wakeword.model_path
        : ''

  return {
    microphoneDeviceId:
      typeof raw.microphoneDeviceId === 'string' ? raw.microphoneDeviceId : '',
    speakerDeviceId:
      typeof raw.speakerDeviceId === 'string' ? raw.speakerDeviceId : '',
    ttsEngine,
    ttsVoice:
      typeof raw.ttsVoice === 'string'
        ? raw.ttsVoice
        : typeof models.tts_voice === 'string'
          ? models.tts_voice
          : '',
    speechRate:
      typeof raw.speechRate === 'number'
        ? raw.speechRate
        : typeof models.tts_speed === 'number'
          ? models.tts_speed
          : typeof models.tts_speed === 'string'
            ? parseFloat(models.tts_speed) || 1.0
            : 1.0,
    volume: typeof raw.volume === 'number' ? raw.volume : 1.0,
    sttModel:
      typeof raw.sttModel === 'string'
        ? raw.sttModel
        : typeof models.whisper_model === 'string'
          ? models.whisper_model
          : 'base',
    sttLanguage:
      typeof raw.sttLanguage === 'string'
        ? raw.sttLanguage
        : typeof models.stt_language === 'string'
          ? models.stt_language
          : 'auto',
    sttDevice,
    wakeWord,
    wakeWordBackend: backend,
    customWakeWordId,
    wakeWordPhrase,
    wakeWordModelPath:
      modelPath || (backend === 'custom_onnx' ? defaultCustomWakeWordAssetPath('custom_onnx', wakeWordPhrase) : ''),
    wakeWordEmbeddingPath:
      embeddingPath || (backend === 'custom_embedding'
        ? defaultCustomWakeWordAssetPath('custom_embedding', customWakeWordId || wakeWordPhrase)
        : '')
  }
}

function buildWakeWordStatus(raw: Settings = {}): WakeWordStatus {
  const voice = buildVoiceSettings(raw)
  const rexConfig = readRexConfig()
  const wakeword =
    rexConfig.wakeword && typeof rexConfig.wakeword === 'object'
      ? (rexConfig.wakeword as Record<string, unknown>)
      : {}
  const fallbackEnabled =
    typeof wakeword.fallback_to_builtin === 'boolean' ? wakeword.fallback_to_builtin : true
  const fallbackKeyword =
    wakeWordIdToPhrase(voice.wakeWord)
    || (typeof wakeword.fallback_keyword === 'string' ? wakeword.fallback_keyword.trim() : '')
    || 'hey jarvis'

  if (voice.wakeWordBackend === 'openwakeword') {
    return {
      requestedBackend: 'openwakeword',
      configuredPhrase: wakeWordIdToPhrase(voice.wakeWord) || 'hey jarvis',
      fallbackEnabled,
      fallbackKeyword,
      assetKind: 'builtin',
      assetPath: '',
      assetExists: false,
      fallbackActive: false,
      status: 'built_in',
      statusLabel: 'Built-in wake path active',
      detail: 'Built-in openWakeWord is configured. No custom asset file is required.'
    }
  }

  const configuredPhrase = voice.wakeWordPhrase.trim() || 'hey rex'
  const assetPath =
    voice.wakeWordBackend === 'custom_onnx'
      ? voice.wakeWordModelPath.trim() || defaultCustomWakeWordAssetPath('custom_onnx', configuredPhrase)
      : voice.wakeWordEmbeddingPath.trim()
        || defaultCustomWakeWordAssetPath(
          'custom_embedding',
          voice.customWakeWordId.trim() || configuredPhrase
        )
  const assetExists = existsSync(assetPath)
  const fallbackActive = fallbackEnabled && !assetExists

  return {
    requestedBackend: voice.wakeWordBackend,
    configuredPhrase,
    fallbackEnabled,
    fallbackKeyword,
    assetKind: voice.wakeWordBackend === 'custom_onnx' ? 'onnx' : 'embedding',
    assetPath,
    assetExists,
    fallbackActive,
    status: assetExists ? 'asset_ready' : 'missing_asset',
    statusLabel: assetExists
      ? `${voice.wakeWordBackend === 'custom_onnx' ? 'Custom ONNX' : 'Custom embedding'} asset found`
      : 'Custom asset missing',
    detail: assetExists
      ? 'Asset file is present. Runtime validation happens when the voice loop starts.'
      : fallbackEnabled
        ? `Missing asset file. Built-in fallback '${fallbackKeyword}' will be used until this asset is added.`
        : 'Missing asset file. Voice startup will fail until a valid custom asset is added.'
  }
}

function normalizeGuiAiProvider(raw: unknown): AiSettings['provider'] {
  if (raw === 'openai' || raw === 'ollama' || raw === 'local') {
    return raw
  }
  if (raw === 'transformers') {
    return 'local'
  }
  return 'openai'
}

function toRuntimeAiProvider(provider: AiSettings['provider']): string {
  return provider === 'local' ? 'transformers' : provider
}

function buildAiSettings(raw: Settings = {}): AiSettings {
  const rexConfig = readRexConfig()
  const models = rexConfig.models && typeof rexConfig.models === 'object'
    ? (rexConfig.models as Record<string, unknown>)
    : {}
  const ollama = rexConfig.ollama && typeof rexConfig.ollama === 'object'
    ? (rexConfig.ollama as Record<string, unknown>)
    : {}
  const rawModel = typeof raw.model === 'string' ? raw.model : null
  const model = rawModel === 'gpt-4o' || rawModel === 'gpt-4-turbo' || rawModel === 'claude-opus-4' || rawModel === 'claude-sonnet-4' || rawModel === 'gemini-1.5-pro'
    ? rawModel
    : 'claude-sonnet-4'
  const routingSource =
    raw.modelRouting && typeof raw.modelRouting === 'object'
      ? raw.modelRouting
      : rexConfig.model_routing
  const rawProvider =
    typeof models.llm_provider === 'string'
      ? models.llm_provider
      : typeof raw.provider === 'string'
        ? raw.provider
        : null
  const provider = normalizeGuiAiProvider(rawProvider)
  const rawCustomModelId = typeof raw.customModelId === 'string' ? raw.customModelId : ''
  const runtimeModelId = typeof models.llm_model === 'string' ? models.llm_model : ''
  const customModelId =
    rawCustomModelId || (provider !== 'openai' ? runtimeModelId : '')
  const ollamaBaseUrl =
    typeof raw.ollamaBaseUrl === 'string' && raw.ollamaBaseUrl
      ? raw.ollamaBaseUrl
      : typeof ollama.base_url === 'string'
        ? ollama.base_url
        : 'http://localhost:11434'

  const VALID_PERSONALITIES = ['Friendly', 'Professional', 'Minimal']
  const rawPersonality = typeof raw.personality === 'string' ? raw.personality : null
  const personality = rawPersonality && VALID_PERSONALITIES.includes(rawPersonality)
    ? rawPersonality
    : typeof rexConfig.personality === 'string' && VALID_PERSONALITIES.includes(rexConfig.personality as string)
      ? rexConfig.personality as string
      : 'Friendly'

  return {
    model,
    provider,
    customModelId,
    ollamaBaseUrl,
    temperature:
      typeof raw.temperature === 'number'
        ? raw.temperature
        : typeof models.llm_temperature === 'string'
          ? parseFloat(models.llm_temperature) || 0.7
          : 0.7,
    maxTokens:
      typeof raw.maxTokens === 'number'
        ? raw.maxTokens
        : typeof models.llm_max_tokens === 'number'
          ? models.llm_max_tokens
          : 2048,
    systemPrompt: typeof raw.systemPrompt === 'string' ? raw.systemPrompt : '',
    autonomyMode:
      raw.autonomyMode === 'supervised' || raw.autonomyMode === 'full-auto'
        ? raw.autonomyMode
        : 'manual',
    budgetPerPlan: typeof raw.budgetPerPlan === 'number' ? raw.budgetPerPlan : 0,
    budgetPerStep: typeof raw.budgetPerStep === 'number' ? raw.budgetPerStep : 0,
    modelRouting: normalizeAiModelRouting(routingSource),
    personality
  }
}

/** Mirror GUI settings into rex_config.json for sections that overlap. */
function mirrorToRexConfig(section: string, values: Settings): void {
  try {
    const rexConfig = readRexConfig()

    if (section === 'ai') {
      const models = ((rexConfig.models ?? {}) as Record<string, unknown>)
      if (typeof values.temperature === 'number') models.llm_temperature = String(values.temperature)
      if (typeof values.maxTokens === 'number') models.llm_max_tokens = values.maxTokens
      const provider = normalizeGuiAiProvider(values.provider)
      models.llm_provider = toRuntimeAiProvider(provider)
      const llm = ((rexConfig.llm ?? {}) as Record<string, unknown>)
      llm.provider = provider
      rexConfig.llm = llm
      if (typeof values.customModelId === 'string' && values.customModelId.trim()) {
        models.llm_model = values.customModelId.trim()
      }
      if (provider === 'openai' && typeof values.model === 'string') {
        const openai = ((rexConfig.openai ?? {}) as Record<string, unknown>)
        openai.model = values.model
        rexConfig.openai = openai
      }
      if (typeof values.ollamaBaseUrl === 'string' && values.ollamaBaseUrl.trim()) {
        const ollama = ((rexConfig.ollama ?? {}) as Record<string, unknown>)
        ollama.base_url = values.ollamaBaseUrl.trim()
        rexConfig.ollama = ollama
      }
      rexConfig.models = models
      rexConfig.model_routing = normalizeAiModelRouting(values.modelRouting)
      if (typeof values.personality === 'string' && values.personality) {
        rexConfig.personality = values.personality
      }
      writeRexConfig(rexConfig)
    }

    if (section === 'voice') {
      const voice = buildVoiceSettings(values)
      const models = ((rexConfig.models ?? {}) as Record<string, unknown>)
      if (typeof voice.ttsEngine === 'string') models.tts_provider = voice.ttsEngine
      if (typeof voice.ttsVoice === 'string') models.tts_voice = voice.ttsVoice
      if (typeof voice.speechRate === 'number') models.tts_speed = voice.speechRate
      if (typeof voice.sttModel === 'string') models.whisper_model = voice.sttModel
      if (typeof voice.sttDevice === 'string') models.whisper_device = voice.sttDevice
      if (typeof voice.sttLanguage === 'string') models.stt_language = voice.sttLanguage
      rexConfig.models = models
      const wakeword = ((rexConfig.wakeword ?? {}) as Record<string, unknown>)
      delete wakeword.model
      wakeword.backend = voice.wakeWordBackend
      wakeword.fallback_to_builtin = true
      wakeword.fallback_keyword = wakeWordIdToPhrase(voice.wakeWord) || 'hey jarvis'
      if (voice.wakeWordBackend === 'openwakeword') {
        const builtInPhrase = wakeWordIdToPhrase(voice.wakeWord)
        wakeword.wakeword = builtInPhrase
        wakeword.keyword = builtInPhrase || null
        wakeword.model_path = null
        wakeword.embedding_path = null
      } else if (voice.wakeWordBackend === 'custom_onnx') {
        const customPhrase = voice.wakeWordPhrase.trim() || 'hey rex'
        wakeword.wakeword = customPhrase
        wakeword.keyword = customPhrase
        wakeword.model_path =
          voice.wakeWordModelPath.trim() || defaultCustomWakeWordAssetPath('custom_onnx', customPhrase)
        wakeword.embedding_path = null
      } else {
        const customPhrase = voice.wakeWordPhrase.trim() || 'hey rex'
        const embeddingId = voice.customWakeWordId.trim() || customPhrase
        wakeword.wakeword = customPhrase
        wakeword.keyword = customPhrase
        wakeword.model_path = null
        wakeword.embedding_path =
          voice.wakeWordEmbeddingPath.trim()
          || defaultCustomWakeWordAssetPath('custom_embedding', embeddingId)
      }
      rexConfig.wakeword = wakeword
      writeRexConfig(rexConfig)
    }

    if (section === 'general') {
      const ui = ((rexConfig.ui ?? {}) as Record<string, unknown>)
      if (typeof values.startMinimized === 'boolean') ui.start_minimized = values.startMinimized
      rexConfig.ui = ui
      if (typeof values.timezone === 'string' && values.timezone.trim()) {
        const location = ((rexConfig.location ?? {}) as Record<string, unknown>)
        location.default_timezone = values.timezone.trim()
        rexConfig.location = location
      }
      writeRexConfig(rexConfig)
    }

    if (section === 'integrations') {
      if (typeof values.emailProvider === 'string') {
        const email = ((rexConfig.email ?? {}) as Record<string, unknown>)
        email.provider = values.emailProvider
        rexConfig.email = email
      }
      if (typeof values.calendarProvider === 'string') {
        const calendar = ((rexConfig.calendar ?? {}) as Record<string, unknown>)
        calendar.provider = values.calendarProvider === 'gmail' ? 'google' : values.calendarProvider
        rexConfig.calendar = calendar
      }
      if (typeof values.haUrl === 'string' && values.haUrl.trim()) {
        const ha = ((rexConfig.home_assistant ?? {}) as Record<string, unknown>)
        ha.base_url = values.haUrl.trim()
        rexConfig.home_assistant = ha
      }
      if (typeof values.telegramChatId === 'string' && values.telegramChatId.trim()) {
        const telegram = ((rexConfig.telegram ?? {}) as Record<string, unknown>)
        telegram.chat_id = values.telegramChatId.trim()
        rexConfig.telegram = telegram
      }
      writeRexConfig(rexConfig)
    }

    if (section === 'system') {
      if (typeof values.toolTimeoutSeconds === 'number') {
        rexConfig.tool_timeout_seconds = values.toolTimeoutSeconds
      }
      if (typeof values.requireConfirmSystemChanges === 'boolean') {
        const windows = ((rexConfig.windows ?? {}) as Record<string, unknown>)
        windows.require_confirm_system_changes = values.requireConfirmSystemChanges
        rexConfig.windows = windows
      }
      if (typeof values.allowedFileRoots === 'string' && values.allowedFileRoots.trim()) {
        rexConfig.allowed_file_roots = values.allowedFileRoots.split(',').map((s: string) => s.trim()).filter(Boolean)
      }
      if (typeof values.debugLogging === 'boolean') {
        const runtime = ((rexConfig.runtime ?? {}) as Record<string, unknown>)
        runtime.log_level = values.debugLogging ? 'DEBUG' : 'INFO'
        rexConfig.runtime = runtime
      }
      if (typeof values.autonomyMode === 'string') {
        const models = ((rexConfig.models ?? {}) as Record<string, unknown>)
        models.autonomy_mode = values.autonomyMode
        rexConfig.models = models
      }
      writeRexConfig(rexConfig)
    }
  } catch {
    // Non-fatal: GUI settings were already persisted; rex_config mirror is best-effort
  }
}

// ---------------------------------------------------------------------------
// Default settings per section
// ---------------------------------------------------------------------------

const defaultSettingsMap: Record<string, Settings> = {
  general: {
    displayName: '',
    timezone: 'America/New_York',
    language: 'English',
    launchAtLogin: false,
    startMinimized: false
  } satisfies GeneralSettings,
  voice: {
    microphoneDeviceId: '',
    speakerDeviceId: '',
    ttsEngine: 'pyttsx3',
    ttsVoice: '',
    speechRate: 1.0,
    volume: 1.0,
    sttModel: 'base',
    sttLanguage: 'auto',
    sttDevice: 'auto',
    wakeWord: '',
    wakeWordBackend: 'openwakeword',
    customWakeWordId: '',
    wakeWordPhrase: 'hey rex',
    wakeWordModelPath: '',
    wakeWordEmbeddingPath: ''
  } satisfies VoiceSettings,
  ai: {
    model: 'claude-sonnet-4',
    provider: 'openai',
    customModelId: '',
    ollamaBaseUrl: 'http://localhost:11434',
    temperature: 0.7,
    maxTokens: 2048,
    systemPrompt: '',
    autonomyMode: 'manual',
    budgetPerPlan: 0,
    budgetPerStep: 0,
    modelRouting: normalizeAiModelRouting({}),
    personality: 'Friendly'
  } satisfies AiSettings as unknown as Settings,
  users: {
    names: {}
  },
  integrations: {
    emailProvider: 'gmail',
    emailClientId: '',
    emailClientSecret: '', // pragma: allowlist secret
    emailAccounts: [] as EmailAccount[],
    calendarProvider: 'gmail',
    calendarClientId: '',
    calendarClientSecret: '', // pragma: allowlist secret
    smsSid: '',
    smsAuthToken: '',
    smsFromNumber: '',
    haUrl: '',
    haToken: '',
    phoneSid: '',
    phoneAuthToken: '',
    phoneNumber: '',
    phoneTransferNumber: '',
    voicemailNotificationsEnabled: false,
    contactsFilePath: '',
    telegramBotToken: '',
    telegramChatId: ''
  } satisfies IntegrationsSettings,
  system: {
    autonomyMode: 'manual',
    toolTimeoutSeconds: 10,
    requireConfirmSystemChanges: true,
    allowedFileRoots: '',
    debugLogging: false
  } satisfies SystemSettings
}

type TestableIntegration = 'email' | 'calendar' | 'sms' | 'homeassistant' | 'phone'

type IntegrationTestResult = { ok: boolean; error?: string }

interface StoredIntegrationStatus {
  status: IntegrationConnectionStatus
  testedAt?: string
  error?: string
  fingerprint?: string
}

function hasText(value: unknown): boolean {
  return typeof value === 'string' && value.trim() !== ''
}

function integrationSettingsFrom(stored: Record<string, Settings>): Record<string, unknown> {
  return {
    ...defaultSettingsMap.integrations,
    ...((stored.integrations ?? {}) as Record<string, unknown>)
  }
}

const OUTLOOK_EMAIL_UNSUPPORTED =
  'Outlook email sync is not implemented yet. The current Outlook settings only store app credentials; Rex cannot read Outlook mail until Microsoft Graph OAuth token support is added.'

const OUTLOOK_CALENDAR_UNSUPPORTED =
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
    return { status: 'error', error: OUTLOOK_EMAIL_UNSUPPORTED }
  }
  if (type === 'calendar' && hasConfiguredOutlookCalendar(integrations)) {
    return { status: 'error', error: OUTLOOK_CALENDAR_UNSUPPORTED }
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
    const status = entry.status
    if (status !== 'connected' && status !== 'error' && status !== 'untested') continue
    statuses[key] = {
      status,
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
  }

  return createHash('sha256').update(JSON.stringify(payload)).digest('hex')
}

function integrationFingerprintForValues(type: string, values: Record<string, unknown>): string {
  if (type === 'homeassistant') {
    return createHash('sha256').update(JSON.stringify(values)).digest('hex')
  }
  return createHash('sha256').update(JSON.stringify({ type, ...values })).digest('hex')
}

function integrationStatusFor(
  type: string,
  stored: Record<string, Settings>
): StoredIntegrationStatus {
  const unsupportedStatus = unsupportedOutlookStatus(type, stored)
  if (unsupportedStatus) return unsupportedStatus

  const status = readIntegrationStatuses(stored)[type]
  if (!status || status.status === 'untested') return { status: 'untested' }
  if (status.fingerprint !== integrationFingerprint(type, stored)) {
    return { status: 'untested' }
  }
  return status
}

function writeIntegrationStatus(
  type: TestableIntegration,
  result: { ok: boolean; error?: string },
  fingerprint = integrationFingerprint(type)
): void {
  const stored = readGuiSettings()
  const statuses = readIntegrationStatuses(stored)
  statuses[type] = {
    status: result.ok ? 'connected' : 'error',
    testedAt: new Date().toISOString(),
    error: result.ok ? undefined : result.error,
    fingerprint
  }
  stored.integrationStatuses = statuses as unknown as Settings
  writeGuiSettings(stored)
}

function reconcileIntegrationStatuses(): void {
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

function hasConfiguredEmail(integrations: Record<string, unknown>): boolean {
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
  return configured ? { ok: true } : { ok: false, error: 'No credentials configured' }
}

function testEmailIntegration(integrations: Record<string, unknown>): IntegrationTestResult {
  if (hasConfiguredOutlookEmail(integrations)) {
    return { ok: false, error: OUTLOOK_EMAIL_UNSUPPORTED }
  }
  return integrationConfiguredResult(hasDirectEmailCredentials(integrations) || hasConfiguredEmail(integrations))
}

function testCalendarIntegration(integrations: Record<string, unknown>): IntegrationTestResult {
  if (hasConfiguredOutlookCalendar(integrations)) {
    return { ok: false, error: OUTLOOK_CALENDAR_UNSUPPORTED }
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

async function testIntegrationByType(
  type: string,
  integrations: Record<string, unknown>
): Promise<{ type?: TestableIntegration; result: IntegrationTestResult }> {
  if (type === 'email') return { type, result: testEmailIntegration(integrations) }
  if (type === 'calendar') return { type, result: testCalendarIntegration(integrations) }
  if (type === 'sms') return { type, result: testSmsIntegration(integrations) }
  if (type === 'phone') return { type, result: testPhoneIntegration(integrations) }
  if (type === 'homeassistant') {
    const { baseUrl, token } = readSavedHomeAssistantCredentials()
    return { type, result: await testHomeAssistantConnection(baseUrl, token) }
  }
  return { result: { ok: false, error: 'Unknown integration type' } }
}

function buildIntegrationInventory(): IntegrationInventoryItem[] {
  const stored = readGuiSettings()
  const integrations = integrationSettingsFrom(stored)
  const rexConfig = readRexConfig()
  const env = readEnvFile()
  const ha = readSavedHomeAssistantCredentials()
  const openai = rexConfig.openai && typeof rexConfig.openai === 'object'
    ? (rexConfig.openai as Record<string, unknown>)
    : {}
  const ollama = rexConfig.ollama && typeof rexConfig.ollama === 'object'
    ? (rexConfig.ollama as Record<string, unknown>)
    : {}

  const make = (
    item: Omit<IntegrationInventoryItem, 'status' | 'testedAt' | 'error'>
  ): IntegrationInventoryItem => {
    const status = integrationStatusFor(item.key, stored)
    return {
      ...item,
      status: status.status,
      testedAt: status.testedAt,
      error: status.error
    }
  }

  return [
    make({
      name: 'Home Assistant',
      key: 'homeassistant',
      configured: hasText(ha.baseUrl) && hasText(ha.token),
      configure_url: '/settings/home-assistant',
      testable: true
    }),
    make({
      name: 'Email',
      key: 'email',
      configured: hasConfiguredEmail(integrations),
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Calendar',
      key: 'calendar',
      configured: hasText(integrations.calendarClientId) && hasText(integrations.calendarClientSecret), // pragma: allowlist secret
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'SMS (Twilio)',
      key: 'sms',
      configured:
        (hasText(integrations.smsSid) && hasText(integrations.smsAuthToken) && hasText(integrations.smsFromNumber)) ||
        (hasText(env.TWILIO_ACCOUNT_SID) && hasText(env.TWILIO_AUTH_TOKEN)),
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Phone (Twilio)',
      key: 'phone',
      configured:
        (hasText(integrations.phoneSid) && hasText(integrations.phoneAuthToken) && hasText(integrations.phoneNumber)) ||
        (hasText(env.TWILIO_ACCOUNT_SID) && hasText(env.TWILIO_AUTH_TOKEN) && hasText(env.TWILIO_PHONE_NUMBER)),
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Telegram',
      key: 'telegram',
      configured: hasText(integrations.telegramChatId) && (hasText(integrations.telegramBotToken) || hasText(env.TELEGRAM_BOT_TOKEN)),
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'Web Search',
      key: 'search',
      configured: hasText(env.SERPAPI_API_KEY) || hasText(env.BRAVE_API_KEY) || hasText(env.GOOGLE_CSE_ID),
      configure_url: '/settings?section=ai'
    }),
    make({
      name: 'MQTT',
      key: 'mqtt',
      configured: hasText(env.MQTT_BROKER_HOST),
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'OpenAI',
      key: 'openai',
      configured: hasText(env.OPENAI_API_KEY) || hasText(openai.api_key), // pragma: allowlist secret
      configure_url: '/settings?section=ai'
    }),
    make({
      name: 'Ollama',
      key: 'ollama',
      configured: hasText(ollama.base_url),
      configure_url: '/settings?section=ai'
    }),
    make({
      name: 'Push Notifications',
      key: 'push',
      configured: hasText(integrations.pushProvider) && hasText(integrations.pushToken),
      configure_url: '/settings?section=integrations'
    })
  ]
}

function buildCapabilityInventory(): CapabilityInfo[] {
  const integrations = buildIntegrationInventory()
  const configured = new Set(integrations.filter((item) => item.configured).map((item) => item.key))
  const connected = new Set(
    integrations
      .filter((item) => item.configured && item.status === 'connected')
      .map((item) => item.key)
  )
  return [
    { name: 'chat', description: 'Text chat with Rex', category: 'Core', enabled: true },
    { name: 'voice', description: 'Wake-word and hold-to-talk voice interaction', category: 'Core', enabled: true },
    { name: 'home_assistant', description: 'Control and inspect Home Assistant entities', category: 'Integrations', enabled: configured.has('homeassistant') },
    { name: 'email', description: 'Read and draft email through configured accounts', category: 'Integrations', enabled: connected.has('email') },
    { name: 'calendar', description: 'Read and create calendar events', category: 'Integrations', enabled: connected.has('calendar') },
    { name: 'sms', description: 'Send SMS through Twilio', category: 'Integrations', enabled: configured.has('sms') }
  ]
}

function registerIpcHandlers(mainWindow: BrowserWindow | null = null): void {
  registerChatHandlers()
  registerVoiceHandlers()
  registerTaskHandlers()
  registerCalendarHandlers()
  registerRemindersHandlers()
  registerMemoriesHandlers()
  registerEmailHandlers()
  registerSMSHandlers()
  registerNotificationHandlers(mainWindow)
  registerSpeakerHandlers()
  registerFileHandlers()
  registerShoppingHandlers()
  registerLogsHandlers()
  registerUsageHandlers()

  ipcMain.handle('rex:getStatus', () => {
    return { ok: true, status: getCurrentVoiceState() }
  })

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

  ipcMain.handle(
    'rex:testHomeAssistant',
    async (_event, baseUrl: string, token: string): Promise<HaTestResult> => {
      const normalizedUrl = normalizeHaUrl(baseUrl)
      const trimmedToken = token.trim()
      const result = await testHomeAssistantConnection(normalizedUrl, trimmedToken)
      writeIntegrationStatus(
        'homeassistant',
        result,
        integrationFingerprintForValues('homeassistant', { haUrl: normalizedUrl, haToken: trimmedToken })
      )
      return result
    }
  )

  ipcMain.handle(
    'rex:saveHomeAssistant',
    async (_event, baseUrl: string, token: string): Promise<HaTestResult> => {
      const normalizedUrl = normalizeHaUrl(baseUrl)
      if (!normalizedUrl) return { ok: false, error: 'Home Assistant URL is required.' }
      try {
        saveHomeAssistantCredentials(normalizedUrl, token.trim())
        reconcileIntegrationStatuses()
        return { ok: true }
      } catch (err) {
        return { ok: false, error: err instanceof Error ? err.message : String(err) }
      }
    }
  )

  ipcMain.handle('rex:getHomeAssistantStates', async (): Promise<HaStatesResult> => {
    return getHomeAssistantStates()
  })

  ipcMain.handle('rex:getIntegrations', () => {
    try {
      return { ok: true, integrations: buildIntegrationInventory() }
    } catch (err) {
      return {
        ok: false,
        integrations: [],
        error: err instanceof Error ? err.message : String(err)
      }
    }
  })

  ipcMain.handle('rex:getCapabilities', () => {
    try {
      return { ok: true, capabilities: buildCapabilityInventory() }
    } catch (err) {
      return {
        ok: false,
        capabilities: [],
        error: err instanceof Error ? err.message : String(err)
      }
    }
  })

  ipcMain.handle('rex:testIntegration', async (_event, type: string) => {
    // Check whether credentials for the requested integration are configured
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    const testResult = await testIntegrationByType(type, integrations)

    if (testResult.type) {
      writeIntegrationStatus(testResult.type, testResult.result)
    }

    return testResult.result
  })

  ipcMain.handle('rex:pickFolder', async (): Promise<{ ok: boolean; path?: string; error?: string }> => {
    const result = await dialog.showOpenDialog({
      title: 'Select Folder',
      properties: ['openDirectory']
    })
    if (result.canceled || result.filePaths.length === 0) {
      return { ok: false, error: 'No folder selected' }
    }
    return { ok: true, path: result.filePaths[0] }
  })

  ipcMain.handle('rex:uploadContactsFile', async (): Promise<{ ok: boolean; path?: string; error?: string }> => {
    const result = await dialog.showOpenDialog({
      title: 'Select Contacts File',
      filters: [
        { name: 'Contacts', extensions: ['vcf', 'json'] },
        { name: 'All Files', extensions: ['*'] }
      ],
      properties: ['openFile']
    })
    if (result.canceled || result.filePaths.length === 0) {
      return { ok: false, error: 'No file selected' }
    }
    const selectedPath = result.filePaths[0]
    // Persist path to integrations settings
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    integrations.contactsFilePath = selectedPath
    stored['integrations'] = integrations
    writeGuiSettings(stored)
    return { ok: true, path: selectedPath }
  })

  ipcMain.handle('rex:testEmailAccount', (_event, id: string) => {
    // Check that the identified account has the required credentials configured
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
    const account = (accounts as EmailAccount[]).find((a) => a.id === id)
    if (!account) return { ok: false, error: 'Account not found' }
    if (account.backend === 'imap') {
      const ok =
        typeof account.host === 'string' && account.host.trim() !== '' &&
        typeof account.username === 'string' && account.username.trim() !== '' &&
        typeof account.password === 'string' && account.password.trim() !== '' // pragma: allowlist secret
      return ok ? { ok: true } : { ok: false, error: 'IMAP host, username, and password are required' }
    }
    // gmail / outlook OAuth
    if (account.backend === 'outlook') {
      return { ok: false, error: OUTLOOK_EMAIL_UNSUPPORTED }
    }
    const ok =
      typeof account.clientId === 'string' && account.clientId.trim() !== '' &&
      typeof account.clientSecret === 'string' && account.clientSecret.trim() !== '' // pragma: allowlist secret
    return ok ? { ok: true } : { ok: false, error: 'OAuth Client ID and Secret are required' }
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

  ipcMain.handle('rex:getVersionInfo', () => {
    let rexVersion = '1.0.0'
    try {
      const pkgPath = join(__dirname, '../../../../package.json')
      const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { version?: string }
      rexVersion = pkg.version ?? rexVersion
    } catch {
      // fallback to default
    }
    return {
      rex: rexVersion,
      electron: process.versions.electron ?? 'unknown',
      node: process.versions.node ?? 'unknown'
    }
  })

  ipcMain.handle('rex:restartRex', (): { ok: boolean; error?: string } => {
    try {
      app.relaunch()
      app.exit(0)
      return { ok: true }
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) }
    }
  })

  ipcMain.handle('rex:resetToDefaults', (): { ok: boolean; error?: string } => {
    try {
      const configDir = getConfigDir()
      const examplePath = join(configDir, 'rex_config.example.json')
      const targetPath = getRexConfigPath()
      if (!existsSync(examplePath)) {
        return { ok: false, error: `Example config not found at ${examplePath}` }
      }
      const exampleContent = readFileSync(examplePath, 'utf8')
      writeFileSync(targetPath, exampleContent, 'utf8')
      return { ok: true }
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) }
    }
  })
}

function createWindow(): BrowserWindow {
  const appIconPath = app.isPackaged
    ? join(process.resourcesPath, 'assets', 'brand', 'icon.ico')
    : join(__dirname, '../../../assets', 'brand', 'icon.ico')

  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    title: 'AskRex Assistant',
    icon: appIconPath,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  return mainWindow
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.rex-ai.rex-gui')
  writeElectronSessionStart()

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  validateBridges()
  appendElectronLog('INFO', 'Electron bridge validation completed', { event: 'bridge_validation' })
  mirrorToRexConfig('integrations', integrationSettingsFrom(readGuiSettings()) as Settings)
  const mainWindow = createWindow()
  registerIpcHandlers(mainWindow)
  createTray(mainWindow)
  appendElectronLog('INFO', 'Electron GUI main window created', { event: 'window_created' })

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  destroyTray()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
