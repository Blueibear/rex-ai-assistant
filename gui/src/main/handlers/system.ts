import { app, dialog, ipcMain } from 'electron'
import { join } from 'path'
import { existsSync, readFileSync, writeFileSync } from 'fs'
import { spawnSync } from 'child_process'
import { getConfigDir, getRexConfigPath } from '../configStore'
import { getCurrentVoiceState } from './voice'

export function registerSystemHandlers(): void {
  ipcMain.handle('rex:getStatus', () => {
    return { ok: true, status: getCurrentVoiceState() }
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

  ipcMain.handle('rex:getAppStatus', () => {
    let version = '1.0.0'
    try {
      const pkgPath = join(__dirname, '../../../../package.json')
      const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { version?: string }
      version = pkg.version ?? version
    } catch {
      // fallback to default
    }

    let pythonVersion = 'unknown'
    for (const cmd of ['python', 'python3']) {
      try {
        const result = spawnSync(cmd, ['--version'], { encoding: 'utf8', timeout: 2000 })
        const output = (result.stdout || result.stderr || '').trim()
        if (result.status === 0 && output) {
          pythonVersion = output.replace(/^Python\s+/i, '')
          break
        }
      } catch {
        // try next
      }
    }

    return {
      version,
      python_version: pythonVersion,
      platform: process.platform
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
