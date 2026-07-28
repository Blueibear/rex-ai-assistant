import { app, BrowserWindow, dialog } from 'electron'
import { electronApp, optimizer } from '@electron-toolkit/utils'
import type { Settings } from '../types/ipc'
import { createTray, destroyTray } from './tray'
import { appendElectronLog, writeElectronSessionStart } from './handlers/logs'
import { validateBridges } from './bridgeResolver'
import { readGuiSettings } from './configStore'
import { integrationSettingsFrom } from './integrationStatus'
import { mirrorToRexConfig } from './settingsMirror'
import { registerIpcHandlers } from './ipc'
import { resolveElectronSessionIdentity } from './sessionIdentity'
import { createWindow } from './window'
import { runInstalledArtifactSmoke } from './artifactSmoke'

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.rex-ai.rex-gui')
  writeElectronSessionStart()

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  validateBridges()
  appendElectronLog('INFO', 'Electron bridge validation completed', {
    event: 'bridge_validation'
  })
  mirrorToRexConfig('integrations', integrationSettingsFrom(readGuiSettings()) as Settings)
  let sessionIdentity
  try {
    sessionIdentity = resolveElectronSessionIdentity()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    appendElectronLog('ERROR', 'Electron user session could not be established', {
      event: 'electron_identity_failed',
      error: message
    })
    dialog.showErrorBox('AskRex needs an active user', message)
    app.quit()
    return
  }
  const mainWindow = createWindow()
  appendElectronLog('INFO', 'Electron user session established', {
    event: 'electron_identity_established',
    user_id: sessionIdentity.userId,
    authentication: sessionIdentity.authentication
  })
  registerIpcHandlers(mainWindow, sessionIdentity)
  const runningArtifactSmoke = runInstalledArtifactSmoke(mainWindow)
  if (!runningArtifactSmoke) createTray(mainWindow)
  appendElectronLog('INFO', 'Electron GUI main window created', {
    event: 'window_created'
  })

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
