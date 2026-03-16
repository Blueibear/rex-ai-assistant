import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { readFileSync } from 'fs'
import type { Settings, GeneralSettings } from '../types/ipc'
import { registerChatHandlers } from './handlers/chat'
import { registerVoiceHandlers } from './handlers/voice'
import { registerTaskHandlers } from './handlers/tasks'
import { registerCalendarHandlers } from './handlers/calendar'
import { registerRemindersHandlers } from './handlers/reminders'
import { registerMemoriesHandlers } from './handlers/memories'

const settingsStore: Record<string, Settings> = {}

const defaultSettingsMap: Record<string, Settings> = {
  general: {
    displayName: '',
    timezone: 'America/New_York',
    language: 'English',
    launchAtLogin: false,
    startMinimized: false
  } satisfies GeneralSettings
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
    return settingsStore[section] ?? defaultSettingsMap[section] ?? {}
  })

  ipcMain.handle('rex:setSettings', (_event, section: string, values: Settings) => {
    settingsStore[section] = values
    return { ok: true }
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

function createWindow(): void {
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
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.rex-ai.rex-gui')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  registerIpcHandlers()
  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
