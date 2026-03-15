import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import type { Settings, Reminder, ReminderInput } from '../types/ipc'
import { registerChatHandlers } from './handlers/chat'
import { registerVoiceHandlers } from './handlers/voice'
import { registerTaskHandlers } from './handlers/tasks'
import { registerCalendarHandlers } from './handlers/calendar'

function registerIpcHandlers(): void {
  registerChatHandlers()
  registerVoiceHandlers()
  registerTaskHandlers()
  registerCalendarHandlers()

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

  // Reminder stubs
  const stubReminders: Reminder[] = [
    {
      id: 'rem-1',
      title: 'Review Q1 report',
      notes: 'Check revenue figures and YoY comparisons',
      dueAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
      priority: 'high',
      done: false
    },
    {
      id: 'rem-2',
      title: 'Call dentist',
      dueAt: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
      priority: 'medium',
      done: false
    },
    {
      id: 'rem-3',
      title: 'Team standup',
      notes: 'Discuss sprint goals',
      dueAt: (() => {
        const d = new Date()
        d.setHours(10, 0, 0, 0)
        return d.toISOString()
      })(),
      priority: 'high',
      done: false
    },
    {
      id: 'rem-4',
      title: 'Buy groceries',
      dueAt: (() => {
        const d = new Date()
        d.setHours(18, 30, 0, 0)
        return d.toISOString()
      })(),
      priority: 'low',
      done: false
    },
    {
      id: 'rem-5',
      title: 'Submit expense report',
      dueAt: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(),
      priority: 'medium',
      done: false
    },
    {
      id: 'rem-6',
      title: 'Schedule performance reviews',
      dueAt: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(),
      priority: 'high',
      done: false
    }
  ]

  ipcMain.handle('rex:getReminders', () => {
    return stubReminders.filter((r) => !r.done)
  })

  ipcMain.handle('rex:completeReminder', (_event, id: string) => {
    const r = stubReminders.find((x) => x.id === id)
    if (r) r.done = true
    return { ok: true }
  })

  ipcMain.handle('rex:saveReminder', (_event, input: ReminderInput): Reminder => {
    if (input.id) {
      const existing = stubReminders.find((r) => r.id === input.id)
      if (existing) {
        existing.title = input.title
        existing.notes = input.notes
        existing.dueAt = input.dueAt
        existing.priority = input.priority
        existing.repeat = input.repeat
        return { ...existing }
      }
    }
    const created: Reminder = {
      id: `rem-${Date.now()}`,
      title: input.title,
      notes: input.notes,
      dueAt: input.dueAt,
      priority: input.priority,
      repeat: input.repeat,
      done: false
    }
    stubReminders.push(created)
    return { ...created }
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
