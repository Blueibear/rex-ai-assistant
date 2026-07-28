import type { CapabilityInfo, IntegrationInventoryItem } from '../types/ipc'
import { readEnvFile, readGuiSettings, readRexConfig } from './configStore'
import { readSavedHomeAssistantCredentials } from './homeAssistant'
import {
  hasConfiguredEmail,
  hasText,
  integrationSettingsFrom,
  integrationStatusFor
} from './integrationStatus'

export function buildIntegrationInventory(): IntegrationInventoryItem[] {
  const stored = readGuiSettings()
  const integrations = integrationSettingsFrom(stored)
  const rexConfig = readRexConfig()
  const env = readEnvFile()
  const ha = readSavedHomeAssistantCredentials()
  const openai = rexConfig.openai && typeof rexConfig.openai === 'object'
    ? (rexConfig.openai as Record<string, unknown>)
    : {}
  const ollama = rexConfig.ollama && typeof rexConfig.ollama === 'object'
    ? (rexConfig.ollama as Record<string, unknown>)
    : {}
  const openclaw = rexConfig.openclaw && typeof rexConfig.openclaw === 'object'
    ? (rexConfig.openclaw as Record<string, unknown>)
    : {}

  const make = (
    item: Omit<
      IntegrationInventoryItem,
      'state' | 'testedAt' | 'error' | 'available' | 'read_capable' | 'write_capable'
    >
  ): IntegrationInventoryItem => {
    const status = integrationStatusFor(item.key, stored)
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
      error: status.error
    }
  }

  return [
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
      configured: hasConfiguredEmail(integrations),
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Calendar',
      key: 'calendar',
      configured: hasText(integrations.calendarClientId) && hasText(integrations.calendarClientSecret), // pragma: allowlist secret
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'SMS (Twilio)',
      key: 'sms',
      configured:
        (hasText(integrations.smsSid) && hasText(integrations.smsAuthToken) && hasText(integrations.smsFromNumber)) ||
        (hasText(env.TWILIO_ACCOUNT_SID) && hasText(env.TWILIO_AUTH_TOKEN) && hasText(env.TWILIO_FROM_NUMBER)),
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Phone (Twilio)',
      key: 'phone',
      configured:
        (hasText(integrations.phoneSid) && hasText(integrations.phoneAuthToken) && hasText(integrations.phoneNumber)) ||
        (hasText(env.TWILIO_ACCOUNT_SID) && hasText(env.TWILIO_AUTH_TOKEN) && hasText(env.TWILIO_PHONE_NUMBER)),
      configure_url: '/settings?section=integrations',
      testable: true
    }),
    make({
      name: 'Telegram',
      key: 'telegram',
      configured: hasText(integrations.telegramChatId) && (hasText(integrations.telegramBotToken) || hasText(env.TELEGRAM_BOT_TOKEN)),
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'Web Search',
      key: 'search',
      configured: hasText(env.SERPAPI_API_KEY) || hasText(env.BRAVE_API_KEY) || hasText(env.GOOGLE_CSE_ID),
      configure_url: '/settings?section=ai'
    }),
    make({
      name: 'MQTT',
      key: 'mqtt',
      configured: hasText(env.MQTT_BROKER_HOST),
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'OpenAI',
      key: 'openai',
      configured: hasText(env.OPENAI_API_KEY) || hasText(openai.api_key), // pragma: allowlist secret
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
      configured: hasText(integrations.pushProvider) && hasText(integrations.pushToken),
      configure_url: '/settings?section=integrations'
    }),
    make({
      name: 'OpenClaw',
      key: 'openclaw',
      configured: hasText(openclaw.gateway_url),
      configure_url: '/settings?section=ai'
    })
  ]
}

export function buildCapabilityInventory(): CapabilityInfo[] {
  const integrations = buildIntegrationInventory()
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
