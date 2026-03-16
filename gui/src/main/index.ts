import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { createTray, destroyTray } from './tray'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import type { Settings, GeneralSettings, VoiceSettings, AiSettings, IntegrationsSettings } from '../types/ipc'
import { registerChatHandlers } from './handlers/chat'
import { registerVoiceHandlers } from './handlers/voice'
import { registerTaskHandlers } from './handlers/tasks'
import { registerCalendarHandlers } from './handlers/calendar'
import { registerRemindersHandlers } from './handlers/reminders'
import { registerMemoriesHandlers } from './handlers/memories'

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

/** Mirror GUI settings into rex_config.json for sections that overlap. */
function mirrorToRexConfig(section: string, values: Settings): void {
  try {
    const rexConfig = readRexConfig()

    if (section === 'ai') {
      const models = ((rexConfig.models ?? {}) as Record<string, unknown>)
      if (typeof values.temperature === 'number') models.llm_temperature = String(values.temperature)
      if (typeof values.maxTokens === 'number') models.llm_max_tokens = values.maxTokens
      rexConfig.models = models
      writeRexConfig(rexConfig)
    }

    if (section === 'voice') {
      const models = ((rexConfig.models ?? {}) as Record<string, unknown>)
      if (typeof values.ttsEngine === 'string') models.tts_provider = values.ttsEngine
      if (typeof values.ttsVoice === 'string') models.tts_voice = values.ttsVoice
      if (typeof values.speechRate === 'number') models.tts_speed = values.speechRate
      rexConfig.models = models
      writeRexConfig(rexConfig)
    }

    if (section === 'general') {
      const ui = ((rexConfig.ui ?? {}) as Record<string, unknown>)
      if (typeof values.startMinimized === 'boolean') ui.start_minimized = values.startMinimized
      rexConfig.ui = ui
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
    ttsEngine: 'system',
    ttsVoice: '',
    speechRate: 1.0,
    volume: 1.0
  } satisfies VoiceSettings,
  ai: {
    model: 'claude-sonnet-4',
    temperature: 0.7,
    maxTokens: 2048,
    systemPrompt: '',
    autonomyMode: 'manual',
    budgetPerPlan: 0,
    budgetPerStep: 0
  } satisfies AiSettings,
  integrations: {
    emailProvider: 'gmail',
    emailClientId: '',
    emailClientSecret: '',
    calendarProvider: 'gmail',
    calendarClientId: '',
    calendarClientSecret: '',
    smsSid: '',
    smsAuthToken: '',
    smsFromNumber: ''
  } satisfies IntegrationsSettings
}

function registerIpcHandlers(): void {
  registerChatHandlers()
  registerVoiceHandlers()
  registerTaskHandlers()
  registerCalendarHandlers()
  registerRemindersHandlers()
  registerMemoriesHandlers()

  ipcMain.handle('rex:getStatus', () => {
    return { ok: true, status: 'idle' }
  })

  ipcMain.handle('rex:getSettings', (_event, section: string): Settings => {
    const stored = readGuiSettings()
    return stored[section] ?? defaultSettingsMap[section] ?? {}
  })

  ipcMain.handle('rex:setSettings', (_event, section: string, values: Settings) => {
    const stored = readGuiSettings()
    stored[section] = values
    writeGuiSettings(stored)
    // Mirror relevant fields to rex_config.json so the Python backend picks them up
    mirrorToRexConfig(section, values)
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

    return { ok: false, error: 'Unknown integration type' }
  })

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

  registerIpcHandlers()
  const mainWindow = createWindow()
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
