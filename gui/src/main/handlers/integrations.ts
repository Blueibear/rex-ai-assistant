import { dialog, ipcMain } from 'electron'
import type { EmailAccount } from '../../types/ipc'
import { readGuiSettings, writeGuiSettings } from '../configStore'
import {
  getHomeAssistantStates,
  normalizeHaUrl,
  saveHomeAssistantCredentials,
  testHomeAssistantConnection
} from '../homeAssistant'
import type { HaStatesResult, HaTestResult } from '../homeAssistant'
import {
  OUTLOOK_EMAIL_UNSUPPORTED,
  integrationFingerprintForValues,
  reconcileIntegrationStatuses,
  testIntegrationByType,
  writeIntegrationStatus
} from '../integrationStatus'
import { buildCapabilityInventory, buildIntegrationInventory } from '../integrationInventory'

export function registerIntegrationsHandlers(): void {
  ipcMain.handle(
    'rex:testHomeAssistant',
    async (_event, baseUrl: string, token: string): Promise<HaTestResult> => {
      const normalizedUrl = normalizeHaUrl(baseUrl)
      const trimmedToken = token.trim()
      const result = await testHomeAssistantConnection(normalizedUrl, trimmedToken)
      writeIntegrationStatus(
        'homeassistant',
        result,
        integrationFingerprintForValues('homeassistant', { haUrl: normalizedUrl, haToken: trimmedToken })
      )
      return result
    }
  )

  ipcMain.handle(
    'rex:saveHomeAssistant',
    async (_event, baseUrl: string, token: string): Promise<HaTestResult> => {
      const normalizedUrl = normalizeHaUrl(baseUrl)
      if (!normalizedUrl) return { ok: false, error: 'Home Assistant URL is required.' }
      try {
        saveHomeAssistantCredentials(normalizedUrl, token.trim())
        reconcileIntegrationStatuses()
        return { ok: true }
      } catch (err) {
        return { ok: false, error: err instanceof Error ? err.message : String(err) }
      }
    }
  )

  ipcMain.handle('rex:getHomeAssistantStates', async (): Promise<HaStatesResult> => {
    return getHomeAssistantStates()
  })

  ipcMain.handle('rex:getIntegrations', () => {
    try {
      return { ok: true, integrations: buildIntegrationInventory() }
    } catch (err) {
      return {
        ok: false,
        integrations: [],
        error: err instanceof Error ? err.message : String(err)
      }
    }
  })

  ipcMain.handle('rex:getCapabilities', () => {
    try {
      return { ok: true, capabilities: buildCapabilityInventory() }
    } catch (err) {
      return {
        ok: false,
        capabilities: [],
        error: err instanceof Error ? err.message : String(err)
      }
    }
  })

  ipcMain.handle('rex:testIntegration', async (_event, type: string) => {
    // Check whether credentials for the requested integration are configured
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    const testResult = await testIntegrationByType(type, integrations)

    if (testResult.type) {
      writeIntegrationStatus(testResult.type, testResult.result)
    }

    return testResult.result
  })

  ipcMain.handle('rex:uploadContactsFile', async (): Promise<{ ok: boolean; path?: string; error?: string }> => {
    const result = await dialog.showOpenDialog({
      title: 'Select Contacts File',
      filters: [
        { name: 'Contacts', extensions: ['vcf', 'json'] },
        { name: 'All Files', extensions: ['*'] }
      ],
      properties: ['openFile']
    })
    if (result.canceled || result.filePaths.length === 0) {
      return { ok: false, error: 'No file selected' }
    }
    const selectedPath = result.filePaths[0]
    // Persist path to integrations settings
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    integrations.contactsFilePath = selectedPath
    stored['integrations'] = integrations
    writeGuiSettings(stored)
    return { ok: true, path: selectedPath }
  })

  ipcMain.handle('rex:testEmailAccount', (_event, id: string) => {
    // Check that the identified account has the required credentials configured
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
    const account = (accounts as EmailAccount[]).find((a) => a.id === id)
    if (!account) return { ok: false, error: 'Account not found' }
    if (account.backend === 'imap') {
      const ok =
        typeof account.host === 'string' && account.host.trim() !== '' &&
        typeof account.username === 'string' && account.username.trim() !== '' &&
        typeof account.password === 'string' && account.password.trim() !== '' // pragma: allowlist secret
      return ok ? { ok: true } : { ok: false, error: 'IMAP host, username, and password are required' }
    }
    // gmail / outlook OAuth
    if (account.backend === 'outlook') {
      return { ok: false, error: OUTLOOK_EMAIL_UNSUPPORTED }
    }
    const ok =
      typeof account.clientId === 'string' && account.clientId.trim() !== '' &&
      typeof account.clientSecret === 'string' && account.clientSecret.trim() !== '' // pragma: allowlist secret
    return ok ? { ok: true } : { ok: false, error: 'OAuth Client ID and Secret are required' }
  })
}
