import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { homedir } from 'os'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { createTray, destroyTray } from './tray'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import type { Settings, GeneralSettings, VoiceSettings, AiSettings, IntegrationsSettings, EmailAccount, PreferenceSuggestion, SystemSettings } from '../types/ipc'
import { registerChatHandlers } from './handlers/chat'
import { registerVoiceHandlers } from './handlers/voice'
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
import { registerLogsHandlers } from './handlers/logs'

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

function buildAiSettings(raw: Settings = {}): AiSettings {
  const rexConfig = readRexConfig()
  const models = rexConfig.models && typeof rexConfig.models === 'object'
    ? (rexConfig.models as Record<string, unknown>)
    : {}
  const rawModel = typeof raw.model === 'string' ? raw.model : null
  const model = rawModel === 'gpt-4o' || rawModel === 'gpt-4-turbo' || rawModel === 'claude-opus-4' || rawModel === 'claude-sonnet-4' || rawModel === 'gemini-1.5-pro'
    ? rawModel
    : 'claude-sonnet-4'
  const routingSource =
    raw.modelRouting && typeof raw.modelRouting === 'object'
      ? raw.modelRouting
      : rexConfig.model_routing
  const rawProvider = typeof raw.provider === 'string' ? raw.provider : null
  const provider: AiSettings['provider'] =
    rawProvider === 'openai' || rawProvider === 'ollama' || rawProvider === 'local'
      ? rawProvider
      : 'openai'

  return {
    model,
    provider,
    customModelId: typeof raw.customModelId === 'string' ? raw.customModelId : '',
    ollamaBaseUrl: typeof raw.ollamaBaseUrl === 'string' ? raw.ollamaBaseUrl : 'http://localhost:11434',
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
    modelRouting: normalizeAiModelRouting(routingSource)
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
      if (typeof values.provider === 'string') models.llm_provider = values.provider
      if (typeof values.ollamaBaseUrl === 'string') models.ollama_base_url = values.ollamaBaseUrl
      if (typeof values.customModelId === 'string') models.custom_model_id = values.customModelId
      rexConfig.models = models
      rexConfig.model_routing = normalizeAiModelRouting(values.modelRouting)
      writeRexConfig(rexConfig)
    }

    if (section === 'voice') {
      const models = ((rexConfig.models ?? {}) as Record<string, unknown>)
      if (typeof values.ttsEngine === 'string') models.tts_provider = values.ttsEngine
      if (typeof values.ttsVoice === 'string') models.tts_voice = values.ttsVoice
      if (typeof values.speechRate === 'number') models.tts_speed = values.speechRate
      if (typeof values.sttModel === 'string') models.whisper_model = values.sttModel
      if (typeof values.sttDevice === 'string') models.whisper_device = values.sttDevice
      if (typeof values.sttLanguage === 'string') models.stt_language = values.sttLanguage
      rexConfig.models = models
      if (typeof values.wakeWord === 'string' && values.wakeWord) {
        const wakeword = ((rexConfig.wakeword ?? {}) as Record<string, unknown>)
        wakeword.model = values.wakeWord
        rexConfig.wakeword = wakeword
      }
      writeRexConfig(rexConfig)
    }

    if (section === 'general') {
      const ui = ((rexConfig.ui ?? {}) as Record<string, unknown>)
      if (typeof values.startMinimized === 'boolean') ui.start_minimized = values.startMinimized
      rexConfig.ui = ui
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
    wakeWord: ''
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
    modelRouting: normalizeAiModelRouting({})
  } satisfies AiSettings as unknown as Settings,
  users: {
    names: {}
  },
  integrations: {
    emailProvider: 'gmail',
    emailClientId: '',
    emailClientSecret: '',
    emailAccounts: [] as EmailAccount[],
    calendarProvider: 'gmail',
    calendarClientId: '',
    calendarClientSecret: '',
    smsSid: '',
    smsAuthToken: '',
    smsFromNumber: '',
    haUrl: '',
    haToken: ''
  } satisfies IntegrationsSettings,
  system: {
    autonomyMode: 'manual',
    toolTimeoutSeconds: 10,
    requireConfirmSystemChanges: true,
    allowedFileRoots: '',
    debugLogging: false
  } satisfies SystemSettings
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

  ipcMain.handle('rex:getStatus', () => {
    return { ok: true, status: 'idle' }
  })

  ipcMain.handle('rex:getSettings', (_event, section: string): Settings => {
    const stored = readGuiSettings()
    if (section === 'ai') {
      return buildAiSettings((stored[section] ?? {}) as Settings) as unknown as Settings
    }
    return stored[section] ?? defaultSettingsMap[section] ?? {}
  })

  ipcMain.handle('rex:setSettings', (_event, section: string, values: Settings) => {
    const stored = readGuiSettings()
    const normalizedValues =
      section === 'ai' ? (buildAiSettings(values) as unknown as Settings) : values
    stored[section] = normalizedValues
    writeGuiSettings(stored)
    // Mirror relevant fields to rex_config.json so the Python backend picks them up
    mirrorToRexConfig(section, normalizedValues)
    return { ok: true }
  })

  ipcMain.handle('rex:testVoice', () => {
    // Stub: in production this would invoke the TTS engine with a test phrase
    return { ok: true }
  })

  ipcMain.handle('rex:testIntegration', (_event, type: string) => {
    // Check whether credentials for the requested integration are configured
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>

    if (type === 'email') {
      const hasCredentials =
        typeof integrations.emailClientId === 'string' && integrations.emailClientId.trim() !== '' &&
        typeof integrations.emailClientSecret === 'string' && integrations.emailClientSecret.trim() !== ''
      if (!hasCredentials) return { ok: false, error: 'No credentials configured' }
      return { ok: true }
    }

    if (type === 'calendar') {
      const hasCredentials =
        typeof integrations.calendarClientId === 'string' && integrations.calendarClientId.trim() !== '' &&
        typeof integrations.calendarClientSecret === 'string' && integrations.calendarClientSecret.trim() !== ''
      if (!hasCredentials) return { ok: false, error: 'No credentials configured' }
      return { ok: true }
    }

    if (type === 'sms') {
      const hasCredentials =
        typeof integrations.smsSid === 'string' && integrations.smsSid.trim() !== '' &&
        typeof integrations.smsAuthToken === 'string' && integrations.smsAuthToken.trim() !== '' &&
        typeof integrations.smsFromNumber === 'string' && integrations.smsFromNumber.trim() !== ''
      if (!hasCredentials) return { ok: false, error: 'No credentials configured' }
      return { ok: true }
    }

    if (type === 'homeassistant') {
      const hasCredentials =
        typeof integrations.haUrl === 'string' && integrations.haUrl.trim() !== '' &&
        typeof integrations.haToken === 'string' && integrations.haToken.trim() !== ''
      if (!hasCredentials) return { ok: false, error: 'No credentials configured' }
      return { ok: true }
    }

    return { ok: false, error: 'Unknown integration type' }
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
        typeof account.password === 'string' && account.password.trim() !== ''
      return ok ? { ok: true } : { ok: false, error: 'IMAP host, username, and password are required' }
    }
    // gmail / outlook OAuth
    const ok =
      typeof account.clientId === 'string' && account.clientId.trim() !== '' &&
      typeof account.clientSecret === 'string' && account.clientSecret.trim() !== ''
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

    // Autonomy mode — highest impact
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

    // Budget — suggest 2× avg if no budget is set
    const avgBudget =
      typeof profile.avg_budget_usd === 'number' ? profile.avg_budget_usd : 0
    if (avgBudget > 0 && aiSettings.budgetPerPlan === 0) {
      const suggested = Math.round(avgBudget * 2 * 100) / 100
      suggestions.push({
        field: 'budgetPerPlan',
        current_value: aiSettings.budgetPerPlan,
        suggested_value: suggested,
        reason: `Your average plan cost is $${avgBudget.toFixed(2)} — a $${suggested.toFixed(2)} budget would prevent overruns`
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
      openai_key_set: typeof env['OPENAI_API_KEY'] === 'string' && env['OPENAI_API_KEY'].trim() !== ''
    }
  })

  ipcMain.handle(
    'rex:setApiKey',
    (_event, name: string, value: string): { ok: boolean; error?: string } => {
      try {
        // Validate key name to prevent arbitrary env writes
        const allowedKeys = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY', 'SERPAPI_KEY', 'BRAVE_API_KEY']
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
}

function createWindow(): BrowserWindow {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
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

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  const mainWindow = createWindow()
  registerIpcHandlers(mainWindow)
  createTray(mainWindow)

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
