import { app, BrowserWindow } from 'electron'
import { electronApp, optimizer } from '@electron-toolkit/utils'
import type { Settings } from '../types/ipc'
import { createTray, destroyTray } from './tray'
import { appendElectronLog, writeElectronSessionStart } from './handlers/logs'
import { validateBridges } from './bridgeResolver'
import { readGuiSettings } from './configStore'
import { integrationSettingsFrom } from './integrationStatus'
import { mirrorToRexConfig } from './settingsMirror'
import { registerIpcHandlers } from './ipc'
import { createWindow } from './window'

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
