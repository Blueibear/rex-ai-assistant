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
  settings_sections?: boolean
  settings_section_failure?: string
  openclaw_settings_failure?: string
  error?: string
}

interface FirstRunArtifactSmokeResult {
  ok: boolean
  setup_ui: boolean
  preauth_ipc: boolean
  setup_completed: boolean
  authenticated_ipc: boolean
  background_voice_enabled: boolean
  error?: string
}

export function runInstalledFirstRunSmoke(mainWindow: BrowserWindow): boolean {
  const outputPath = process.env['ASKREX_ARTIFACT_SMOKE_OUTPUT']
  if (!outputPath) return false

  const finish = (result: FirstRunArtifactSmokeResult): void => {
    writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8')
    appendElectronLog(result.ok ? 'INFO' : 'ERROR', 'Installed first-run smoke completed', {
      event: 'installed_first_run_smoke',
      ok: result.ok
    })
    app.quit()
  }

  mainWindow.webContents.once('did-finish-load', () => {
    void mainWindow.webContents
      .executeJavaScript(`(async () => {
        const api = window.rex;
        if (!api || typeof api.getSetupStatus !== 'function' ||
            typeof api.completeSetup !== 'function') {
          throw new Error('Typed AskRex setup preload API is unavailable');
        }
        const waitFor = async (predicate, timeoutMs = 10000) => {
          const deadline = Date.now() + timeoutMs;
          while (Date.now() < deadline) {
            if (predicate()) return true;
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
          return Boolean(predicate());
        };
        const initialStatus = await api.getSetupStatus();
        const setupUi = await waitFor(() =>
          (document.body?.innerText ?? '').includes('Set up Account')
        );
        const preauthMethods = [
          'getSetupAudioDevices',
          'testSetupAudioDevice',
          'listVoices',
          'previewVoice',
          'listWakeWords',
          'previewWakeWordSample',
          'getWakeWordStatus'
        ];
        const preauthIpc = preauthMethods.every((name) => typeof api[name] === 'function');
        if (!initialStatus.needs_setup || !setupUi || !preauthIpc) {
          throw new Error('Fresh install did not expose the first-run setup surface');
        }

        const wakeStatus = await api.getWakeWordStatus();
        const wakeInventory = await api.listWakeWords();
        if (!wakeStatus || !wakeInventory || typeof wakeInventory.ok !== 'boolean') {
          throw new Error('Pre-auth wake-word setup IPC did not return a typed result');
        }

        const completed = await api.completeSetup({
          username: 'artifact-first-run-user',
          password: 'artifact-smoke-password', // pragma: allowlist secret
          llm_provider: 'local',
          tts_provider: 'pyttsx3',
          tts_voice_id: '',
          microphone_device_index: null,
          speaker_device_index: null,
          local_device_id: 'local_voice',
          wake_word_id: 'hey_rex',
          room_name: 'Artifact Room',
          background_voice_enabled: false,
          ha_base_url: '',
          ha_token: '',
          defer_home_assistant: true
        });
        if (!completed.ok) {
          throw new Error('First-run setup completion failed: ' + (completed.error ?? 'unknown error'));
        }

        const postSetup = await api.getSetupStatus();
        const authenticatedStatus = await api.getStatus();
        const authenticatedIpc =
          postSetup.needs_setup === false &&
          authenticatedStatus &&
          typeof authenticatedStatus.status === 'string';
        return {
          ok: setupUi && preauthIpc && completed.ok && authenticatedIpc,
          setup_ui: setupUi,
          preauth_ipc: preauthIpc,
          setup_completed: completed.ok,
          authenticated_ipc: authenticatedIpc,
          background_voice_enabled: false
        };
      })()`)
      .then((result: FirstRunArtifactSmokeResult) => finish(result))
      .catch((error: unknown) =>
        finish({
          ok: false,
          setup_ui: false,
          preauth_ipc: false,
          setup_completed: false,
          authenticated_ipc: false,
          background_voice_enabled: false,
          error: error instanceof Error ? error.message : String(error)
        })
      )
  })

  return true
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
        let settingsSectionFailure = '';
        const settingsNavigationReady = await waitFor(() =>
          Array.from(document.querySelectorAll('nav button')).some(
            (node) => node.textContent?.trim() === 'General'
          )
        );
        if (!settingsNavigationReady) {
          const navLabels = Array.from(document.querySelectorAll('nav button'))
            .map((node) => node.textContent?.trim() ?? '')
            .filter(Boolean)
            .join(',');
          settingsSectionFailure = 'settings-nav-not-ready:' + window.location.hash + ':' + navLabels + ':' + (document.body?.innerText ?? '').slice(0, 240);
        }
        const settingsExpectations = [
          ['General', 'General'],
          ['Voice', 'Voice'],
          ['AI', 'AI'],
          ['Integrations', 'Integrations'],
          ['Notifications', 'Notifications'],
          ['Users', 'Users'],
          ['Audio Output', 'Audio Output'],
          ['System', 'System & Advanced'],
          ['About', 'AskRex Assistant']
        ];
        let settingsSections = true;
        for (const [label, expectedHeading] of settingsExpectations) {
          const button = Array.from(document.querySelectorAll('nav button')).find(
            (node) => node.textContent?.trim() === label
          );
          if (!(button instanceof HTMLElement)) {
            settingsSections = false;
            if (!settingsSectionFailure) settingsSectionFailure = 'missing-settings-button:' + label;
            break;
          }
          button.click();
          const rendered = await waitFor(
            () => (document.querySelector('main h2')?.textContent?.trim() ?? '') === expectedHeading
          );
          if (!rendered) {
            settingsSections = false;
            const actualHeading = document.querySelector('main h2')?.textContent?.trim() ?? '';
            if (!settingsSectionFailure) settingsSectionFailure = 'settings-heading:' + label + ':' + actualHeading;
            break;
          }
        }
        const integrationsButton = Array.from(document.querySelectorAll('nav button')).find(
          (node) => node.textContent?.trim() === 'Integrations'
        );
        if (integrationsButton instanceof HTMLElement) integrationsButton.click();
        const openclawSettings = integrationsButton instanceof HTMLElement && await waitFor(() => {
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
            openclawSettingsReadWrite &&
            settingsSections,
          typed_ipc: true,
          version,
          chat,
          memories_count: Array.isArray(memories) ? memories.length : -1,
          openclaw_settings: openclawSettings,
          openclaw_settings_read_write: openclawSettingsReadWrite,
          settings_sections: settingsSections,
          settings_section_failure: settingsSectionFailure,
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
