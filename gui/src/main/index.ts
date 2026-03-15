import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import type { Settings, Task } from '../types/ipc'
import { registerChatHandlers } from './handlers/chat'
import { registerVoiceHandlers } from './handlers/voice'

function registerIpcHandlers(): void {
  registerChatHandlers()
  registerVoiceHandlers()

  ipcMain.handle('rex:getStatus', () => {
    return { ok: true, status: 'idle' }
  })

  ipcMain.handle('rex:getSettings', () => {
    return { ok: true, settings: {} }
  })

  ipcMain.handle('rex:setSettings', (_event, settings: Settings) => {
    console.log('[rex:setSettings]', settings)
    return { ok: true }
  })

  ipcMain.handle('rex:getTasks', (): Task[] => {
    return [
      {
        id: 'task-1',
        name: 'Morning briefing',
        schedule: 'Every day at 8:00am',
        nextRun: 'Tomorrow 8:00am',
        status: 'active'
      },
      {
        id: 'task-2',
        name: 'Weekly summary email',
        schedule: 'Every Monday at 9:00am',
        nextRun: 'Mon 9:00am',
        status: 'paused'
      },
      {
        id: 'task-3',
        name: 'Server health check',
        schedule: 'Every hour',
        nextRun: 'In 23 minutes',
        status: 'error'
      }
    ]
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
