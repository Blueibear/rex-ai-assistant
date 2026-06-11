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

  const make = (
    item: Omit<IntegrationInventoryItem, 'status' | 'testedAt' | 'error'>
  ): IntegrationInventoryItem => {
    const status = integrationStatusFor(item.key, stored)
    return {
      ...item,
      status: status.status,
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
        (hasText(env.TWILIO_ACCOUNT_SID) && hasText(env.TWILIO_AUTH_TOKEN)),
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
    })
  ]
}

export function buildCapabilityInventory(): CapabilityInfo[] {
  const integrations = buildIntegrationInventory()
  const configured = new Set(integrations.filter((item) => item.configured).map((item) => item.key))
  const connected = new Set(
    integrations
      .filter((item) => item.configured && item.status === 'connected')
      .map((item) => item.key)
  )
  return [
    { name: 'chat', description: 'Text chat with Rex', category: 'Core', enabled: true },
    { name: 'voice', description: 'Wake-word and hold-to-talk voice interaction', category: 'Core', enabled: true },
    { name: 'home_assistant', description: 'Control and inspect Home Assistant entities', category: 'Integrations', enabled: configured.has('homeassistant') },
    { name: 'email', description: 'Read and draft email through configured accounts', category: 'Integrations', enabled: connected.has('email') },
    { name: 'calendar', description: 'Read and create calendar events', category: 'Integrations', enabled: connected.has('calendar') },
    { name: 'sms', description: 'Send SMS through Twilio', category: 'Integrations', enabled: configured.has('sms') }
  ]
}
