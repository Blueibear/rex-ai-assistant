import { dialog, ipcMain } from 'electron'
import type { EmailAccount } from '../../types/ipc'
import { readGuiSettings, readRexConfigStrict, writeGuiSettings } from '../configStore'
import {
  getHomeAssistantStates,
  normalizeHaUrl,
  readSavedHomeAssistantCredentials,
  saveHomeAssistantCredentials,
  testHomeAssistantConnection
} from '../homeAssistant'
import type { HaStatesResult, HaTestResult } from '../homeAssistant'
import {
  OUTLOOK_EMAIL_UNSUPPORTED,
  reconcileIntegrationStatuses,
  testIntegrationByType,
  writeIntegrationStatus
} from '../integrationStatus'
import { buildCapabilityInventory, buildIntegrationInventory } from '../integrationInventory'
import type { ElectronSessionIdentity } from '../sessionIdentity'
import { getVaultReference } from '../credentialReferences'
import { vaultHasSecret, type VaultContext } from '../credentialVault'
import { safeIpcErrorMessage } from '../ipcErrors'

export function registerIntegrationsHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle(
    'rex:testHomeAssistant',
    async (_event, baseUrl: string, token: string): Promise<HaTestResult> => {
      const normalizedUrl = normalizeHaUrl(baseUrl)
      const saved = await readSavedHomeAssistantCredentials(session)
      const trimmedToken = token.trim() || saved.token
      const result = await testHomeAssistantConnection(normalizedUrl, trimmedToken)
      if (!token.trim() && saved.ref) {
        await writeIntegrationStatus(session, 'homeassistant', result)
      }
      return result
    }
  )

  ipcMain.handle(
    'rex:saveHomeAssistant',
    async (_event, baseUrl: string, token: string): Promise<HaTestResult> => {
      const normalizedUrl = normalizeHaUrl(baseUrl)
      if (!normalizedUrl) return { ok: false, error: 'Home Assistant URL is required.' }
      try {
        await saveHomeAssistantCredentials(session, normalizedUrl, token.trim())
        await reconcileIntegrationStatuses(session)
        return { ok: true }
      } catch (err) {
        return { ok: false, error: safeIpcErrorMessage(err, 'Home Assistant credentials could not be saved') }
      }
    }
  )

  ipcMain.handle('rex:getHomeAssistantStates', async (): Promise<HaStatesResult> => {
    return getHomeAssistantStates(session)
  })

  ipcMain.handle('rex:getIntegrations', async () => {
    try {
      return { ok: true, integrations: await buildIntegrationInventory(session) }
    } catch (err) {
      return {
        ok: false,
        integrations: [],
        error: safeIpcErrorMessage(err, 'Integration inventory could not be loaded')
      }
    }
  })

  ipcMain.handle('rex:getCapabilities', async () => {
    try {
      return { ok: true, capabilities: await buildCapabilityInventory(session) }
    } catch (err) {
      return {
        ok: false,
        capabilities: [],
        error: safeIpcErrorMessage(err, 'Capability inventory could not be loaded')
      }
    }
  })

  ipcMain.handle('rex:testIntegration', async (_event, type: string) => {
    // Check whether credentials for the requested integration are configured
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    const testResult = await testIntegrationByType(session, type, integrations)

    if (testResult.type) {
      await writeIntegrationStatus(session, testResult.type, testResult.result)
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

  ipcMain.handle('rex:testEmailAccount', async (_event, id: string) => {
    // This is a configuration check only. It does not contact the provider.
    const stored = readGuiSettings()
    const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
    const accounts = Array.isArray(integrations.emailAccounts) ? integrations.emailAccounts : []
    const account = (accounts as EmailAccount[]).find((a) => a.id === id)
    if (!account) return { ok: false, error: 'Account not found' }
    const validatedId = /^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$/.test(id) ? id : ''
    if (!validatedId) return { ok: false, error: 'Account not found' }
    const context: VaultContext = {
      scope: 'user', integration: 'email', account: validatedId,
      slot: account.backend === 'imap' ? 'password' : 'client_secret'
    }
    const record = getVaultReference(
      readRexConfigStrict(), `email:${validatedId}`, context, session.userId
    )
    const hasCredential = record
      ? await vaultHasSecret(session, record.ref, context)
      : false
    if (account.backend === 'imap') {
      const ok =
        typeof account.host === 'string' && account.host.trim() !== '' &&
        typeof account.username === 'string' && account.username.trim() !== '' &&
        hasCredential
      return ok
        ? { ok: false, state: 'configured', error: 'Credentials are present, but provider authentication was not tested.' }
        : { ok: false, state: 'unconfigured', error: 'IMAP host, username, and password are required' }
    }
    // gmail / outlook OAuth
    if (account.backend === 'outlook') {
      return { ok: false, state: 'unavailable', error: OUTLOOK_EMAIL_UNSUPPORTED }
    }
    const ok =
      typeof account.clientId === 'string' && account.clientId.trim() !== '' &&
      hasCredential
    return ok
      ? { ok: false, state: 'configured', error: 'Credentials are present, but provider authentication was not tested.' }
      : { ok: false, state: 'unconfigured', error: 'OAuth Client ID and Secret are required' }
  })
}
