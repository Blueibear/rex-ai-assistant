import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import type { Settings, Memory, MemoryUpdateInput } from '../types/ipc'
import { registerChatHandlers } from './handlers/chat'
import { registerVoiceHandlers } from './handlers/voice'
import { registerTaskHandlers } from './handlers/tasks'
import { registerCalendarHandlers } from './handlers/calendar'
import { registerRemindersHandlers } from './handlers/reminders'

function registerIpcHandlers(): void {
  registerChatHandlers()
  registerVoiceHandlers()
  registerTaskHandlers()
  registerCalendarHandlers()
  registerRemindersHandlers()

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

  const now = new Date().toISOString()
  const stubMemories: Memory[] = [
    { id: 'mem-1', text: 'User prefers concise answers without filler words.', category: 'preferences', createdAt: now, updatedAt: now },
    { id: 'mem-2', text: 'User is a software engineer working on a local AI assistant project called Rex.', category: 'profile', createdAt: now, updatedAt: now },
    { id: 'mem-3', text: 'User dislikes meetings before 10am.', category: 'preferences', createdAt: now, updatedAt: now },
    { id: 'mem-4', text: 'User is based in the UK (GMT+0 / BST).', category: 'profile', createdAt: now, updatedAt: now },
    { id: 'mem-5', text: 'User frequently asks Rex to draft emails and summarize documents.', category: 'usage', createdAt: now, updatedAt: now }
  ]
  ipcMain.handle('rex:getMemories', () => stubMemories)
  ipcMain.handle('rex:addMemory', (_event, data: MemoryUpdateInput): Memory => {
    const newMemory: Memory = {
      id: `mem-${Date.now()}`,
      text: data.text,
      category: data.category,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
    stubMemories.push(newMemory)
    return newMemory
  })
  ipcMain.handle('rex:updateMemory', (_event, id: string, data: MemoryUpdateInput): Memory => {
    const idx = stubMemories.findIndex((m) => m.id === id)
    if (idx === -1) throw new Error(`Memory ${id} not found`)
    const updated: Memory = { ...stubMemories[idx], ...data, updatedAt: new Date().toISOString() }
    stubMemories[idx] = updated
    return updated
  })
  ipcMain.handle('rex:deleteMemory', (_event, id: string): void => {
    const idx = stubMemories.findIndex((m) => m.id === id)
    if (idx !== -1) stubMemories.splice(idx, 1)
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
