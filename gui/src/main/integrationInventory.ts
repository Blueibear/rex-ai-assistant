import type { CapabilityInfo, IntegrationInventoryItem } from '../types/ipc'
import { readGuiSettings, readRexConfigStrict } from './configStore'
import { readSavedHomeAssistantCredentials } from './homeAssistant'
import {
  hasConfiguredEmail,
  hasText,
  integrationSettingsFrom,
  integrationStatusFor,
  testIntegrationByType
} from './integrationStatus'
import type { ElectronSessionIdentity } from './sessionIdentity'
import { getVaultReference } from './credentialReferences'
import { vaultHasSecret, type VaultContext } from './credentialVault'

function statusCopy(
  state: IntegrationInventoryItem['state'],
  error?: string
): Pick<IntegrationInventoryItem, 'detail' | 'next_action'> {
  if ((state === 'unavailable' || state === 'degraded') && error) {
    return {
      detail: error,
      next_action: state === 'unavailable'
        ? 'Review the availability note before changing configuration.'
        : 'Review the error, correct configuration, and run the connection test again.'
    }
  }
  const copy: Record<IntegrationInventoryItem['state'], [string, string]> = {
    unavailable: ['This integration is unavailable in the current build.', 'Review supported providers before configuring it.'],
    unconfigured: ['No complete configuration is stored.', 'Open settings and enter the required non-secret values and credentials.'],
    configured: ['Configuration is stored, but provider access has not been tested.', 'Run the connection test before relying on this integration.'],
    reachable: ['The service endpoint responded, but authentication is not established.', 'Complete authentication and test again.'],
    authenticated: ['Provider authentication was tested successfully.', 'Review permissions before enabling any write action.'],
    degraded: ['The last connection test failed or returned incomplete evidence.', 'Review the error and test again.'],
    read_only: ['Read access was tested; write access is not available.', 'Use read operations only or update provider permissions.'],
    write_capable: ['The provider reports write capability, but no write was verified.', 'Use confirmation and verify the first write result.'],
    write_tested: ['A write test completed, but ongoing actions still require verification.', 'Verify every consequential action before reporting success.'],
    verified: ['The configured capability has current verified evidence.', 'Retest after credential or configuration changes.']
  }
  const [detail, next_action] = copy[state]
  return { detail, next_action }
}

export async function buildIntegrationInventory(session: ElectronSessionIdentity): Promise<IntegrationInventoryItem[]> {
  const stored = readGuiSettings()
  const integrations = integrationSettingsFrom(stored)
  const rexConfig = readRexConfigStrict()
  const ha = await readSavedHomeAssistantCredentials(session)
  const ollama = rexConfig.ollama && typeof rexConfig.ollama === 'object'
    ? (rexConfig.ollama as Record<string, unknown>)
    : {}
  const openclaw = rexConfig.openclaw && typeof rexConfig.openclaw === 'object'
    ? (rexConfig.openclaw as Record<string, unknown>)
    : {}

  const make = async (
    item: Omit<
      IntegrationInventoryItem,
      'state' | 'testedAt' | 'error' | 'available' | 'read_capable' | 'write_capable' | 'detail' | 'next_action'
    >
  ): Promise<IntegrationInventoryItem> => {
    const status = await integrationStatusFor(session, item.key, stored)
    const state = status.state === 'unconfigured' && item.configured
      ? 'configured'
      : status.state
    const readCapable = ['authenticated', 'read_only', 'write_capable', 'write_tested', 'verified'].includes(state)
    const writeCapable = ['write_capable', 'write_tested', 'verified'].includes(state)
    return {
      ...item,
      state,
      available: state !== 'unavailable',
      read_capable: readCapable,
      write_capable: writeCapable,
      testedAt: status.testedAt,
      error: status.error,
      ...statusCopy(state, status.error)
    }
  }

  const hasStored = async (logicalName: string, context: VaultContext): Promise<boolean> => {
    const record = getVaultReference(rexConfig, logicalName, context, session.userId)
    return record ? vaultHasSecret(session, record.ref, context) : false
  }
  const calendarConfigured = (await testIntegrationByType(session, 'calendar', integrations)).result.state === 'configured'
  const smsConfigured = (await testIntegrationByType(session, 'sms', integrations)).result.state === 'configured'
  const phoneConfigured = (await testIntegrationByType(session, 'phone', integrations)).result.state === 'configured'
  const telegramConfigured = hasText(integrations.telegramChatId) && await hasStored(
    'TELEGRAM_BOT_TOKEN', { scope: 'household', integration: 'telegram', account: null, slot: 'token' }
  )
  const openaiConfigured = await hasStored(
    'OPENAI_API_KEY', { scope: 'household', integration: 'openai', account: null, slot: 'api_key' }
  )
  const searchConfigured = await hasStored(
    'SERPAPI_KEY', { scope: 'household', integration: 'serpapi', account: null, slot: 'api_key' }
  ) || await hasStored(
    'BRAVE_API_KEY', { scope: 'household', integration: 'brave', account: null, slot: 'api_key' }
  )
  const openclawConfigured = hasText(openclaw.gateway_url) && await hasStored(
    'OPENCLAW_GATEWAY_TOKEN', { scope: 'household', integration: 'openclaw_gateway', account: null, slot: 'token' }
  )

  return Promise.all([
    make({
      name: 'Home Assistant',
      key: 'homeassistant',
      configured: hasText(ha.baseUrl) && hasText(ha.token),
      configure_url: '/settings/home-assistant',
      testable: true
    }),
    make({
      name: 'Email',
      key: 'email',
      configured: await hasConfiguredEmail(session, integrations, rexConfig),
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Calendar',
      key: 'calendar',
      configured: calendarConfigured,
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'SMS (Twilio)',
      key: 'sms',
      configured: smsConfigured,
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Phone (Twilio)',
      key: 'phone',
      configured: phoneConfigured,
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Telegram',
      key: 'telegram',
      configured: telegramConfigured,
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'Web Search',
      key: 'search',
      configured: searchConfigured,
      configure_url: '/settings?section=ai'
    }),
    make({
      name: 'MQTT',
      key: 'mqtt',
      configured: false,
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'OpenAI',
      key: 'openai',
      configured: openaiConfigured,
      configure_url: '/settings?section=ai'
    }),
    make({
      name: 'Ollama',
      key: 'ollama',
      configured: hasText(ollama.base_url),
      configure_url: '/settings?section=ai'
    }),
    make({
      name: 'Push Notifications',
      key: 'push',
      configured: false,
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'OpenClaw',
      key: 'openclaw',
      configured: openclawConfigured,
      configure_url: '/settings?section=integrations',
      testable: true
    })
  ])
}

export async function buildCapabilityInventory(session: ElectronSessionIdentity): Promise<CapabilityInfo[]> {
  const integrations = await buildIntegrationInventory(session)
  const readable = new Set(integrations.filter((item) => item.read_capable).map((item) => item.key))
  const stateFor = (key: string): IntegrationInventoryItem | undefined =>
    integrations.find((item) => item.key === key)
  return [
    { name: 'chat', description: 'Text chat with Rex', category: 'Core', enabled: true },
    { name: 'voice', description: 'Wake-word and hold-to-talk voice interaction', category: 'Core', enabled: true },
    { name: 'home_assistant', description: 'Inspect authenticated Home Assistant entities; writes use separate verification', category: 'Integrations', enabled: readable.has('homeassistant'), state: stateFor('homeassistant')?.state, read_capable: stateFor('homeassistant')?.read_capable, write_capable: stateFor('homeassistant')?.write_capable },
    { name: 'email', description: 'Read and draft email through authenticated accounts; GUI sending is unavailable', category: 'Integrations', enabled: readable.has('email'), state: stateFor('email')?.state, read_capable: stateFor('email')?.read_capable, write_capable: false },
    { name: 'calendar', description: 'Read calendar events after provider authentication', category: 'Integrations', enabled: readable.has('calendar'), state: stateFor('calendar')?.state, read_capable: stateFor('calendar')?.read_capable, write_capable: stateFor('calendar')?.write_capable },
    { name: 'sms', description: 'Send SMS only after provider write capability is tested', category: 'Integrations', enabled: stateFor('sms')?.write_capable ?? false, state: stateFor('sms')?.state, read_capable: stateFor('sms')?.read_capable, write_capable: stateFor('sms')?.write_capable }
  ]
}
