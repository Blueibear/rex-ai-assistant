import { app, type BrowserWindow } from 'electron'
import { writeFileSync } from 'fs'
import { appendElectronLog } from './handlers/logs'

interface ArtifactSmokeResult {
  ok: boolean
  typed_ipc: boolean
  version?: unknown
  chat?: string
  memories_count?: number
  openclaw_settings?: boolean
  openclaw_settings_read_write?: boolean
  openclaw_settings_failure?: string
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
        const waitFor = async (predicate, timeoutMs = 5000) => {
          const deadline = Date.now() + timeoutMs;
          while (Date.now() < deadline) {
            if (predicate()) return true;
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
          return Boolean(predicate());
        };
        window.location.hash = '#/settings';
        const integrationsNavigationReady = await waitFor(() =>
          Array.from(document.querySelectorAll('nav button')).some(
            (node) => node.textContent?.trim() === 'Integrations'
          )
        );
        const integrationsButton = Array.from(document.querySelectorAll('nav button')).find(
          (node) => node.textContent?.trim() === 'Integrations'
        );
        if (integrationsButton instanceof HTMLElement) integrationsButton.click();
        const openclawSettings = integrationsNavigationReady && await waitFor(() => {
          const bodyText = document.body?.innerText ?? '';
          return (
            bodyText.includes('OpenClaw') &&
            bodyText.includes('Experimental - off by default') &&
            bodyText.includes('Enable OpenClaw tools') &&
            bodyText.includes('Enable OpenClaw voice backend') &&
            document.getElementById('openclawGatewayUrl') !== null &&
            document.getElementById('openclawToken') !== null
          );
        });
        const openclawSettingsFailure = openclawSettings
          ? ''
          : 'openclaw-ui:' + window.location.hash + ':' +
            (document.querySelector('main h2')?.textContent?.trim() ?? '') + ':' +
            (document.body?.innerText ?? '').slice(0, 240);
        const originalIntegrations = await api.getSettings('integrations');
        const smokeGatewayUrl = 'http://127.0.0.1:18789';
        const writeResult = await api.setSettings('integrations', {
          ...originalIntegrations,
          openclawGatewayUrl: smokeGatewayUrl,
          openclawToolsEnabled: false,
          openclawVoiceEnabled: false,
          openclawToken: ''
        });
        const rereadIntegrations = await api.getSettings('integrations');
        const openclawSettingsReadWrite =
          writeResult?.ok === true &&
          rereadIntegrations?.openclawGatewayUrl === smokeGatewayUrl &&
          rereadIntegrations?.openclawToolsEnabled === false &&
          rereadIntegrations?.openclawVoiceEnabled === false &&
          rereadIntegrations?.openclawToken === '';
        await api.setSettings('integrations', originalIntegrations);
        return {
          ok:
            chat === 'AskRex installed artifact chat verified' &&
            openclawSettings &&
            openclawSettingsReadWrite,
          typed_ipc: true,
          version,
          chat,
          memories_count: Array.isArray(memories) ? memories.length : -1,
          openclaw_settings: openclawSettings,
          openclaw_settings_read_write: openclawSettingsReadWrite,
          openclaw_settings_failure: openclawSettingsFailure
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
