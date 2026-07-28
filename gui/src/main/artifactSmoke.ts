import { app, type BrowserWindow } from 'electron'
import { writeFileSync } from 'fs'
import { appendElectronLog } from './handlers/logs'

interface ArtifactSmokeResult {
  ok: boolean
  typed_ipc: boolean
  version?: unknown
  chat?: string
  memories_count?: number
  error?: string
}

export function runInstalledArtifactSmoke(mainWindow: BrowserWindow): boolean {
  const outputPath = process.env['ASKREX_ARTIFACT_SMOKE_OUTPUT']
  if (!outputPath) return false

  const finish = (result: ArtifactSmokeResult): void => {
    writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8')
    appendElectronLog(result.ok ? 'INFO' : 'ERROR', 'Installed artifact smoke completed', {
      event: 'installed_artifact_smoke',
      ok: result.ok
    })
    app.quit()
  }

  mainWindow.webContents.once('did-finish-load', () => {
    void mainWindow.webContents
      .executeJavaScript(`(async () => {
        const api = window.rex;
        if (!api || typeof api.getVersionInfo !== 'function' ||
            typeof api.sendChat !== 'function' || typeof api.getMemories !== 'function') {
          throw new Error('Typed AskRex preload API is unavailable');
        }
        const version = await api.getVersionInfo();
        const chat = await api.sendChat('AskRex installed artifact smoke test');
        const memories = await api.getMemories();
        return {
          ok: chat === 'AskRex installed artifact chat verified',
          typed_ipc: true,
          version,
          chat,
          memories_count: Array.isArray(memories) ? memories.length : -1
        };
      })()`)
      .then((result: ArtifactSmokeResult) => finish(result))
      .catch((error: unknown) =>
        finish({
          ok: false,
          typed_ipc: false,
          error: error instanceof Error ? error.message : String(error)
        })
      )
  })

  return true
}
