import type {
  AiSettings,
  EmailAccount,
  GeneralSettings,
  IntegrationsSettings,
  Settings,
  SystemSettings,
  VoiceSettings
} from '../types/ipc'
import { normalizeAiModelRouting } from './aiSettings'

// ---------------------------------------------------------------------------
// Default settings per section
// ---------------------------------------------------------------------------

export const defaultSettingsMap: Record<string, Settings> = {
  general: {
    displayName: '',
    timezone: 'America/New_York',
    language: 'English',
    launchAtLogin: false,
    startMinimized: false
  } satisfies GeneralSettings,
  voice: {
    microphoneDeviceId: '',
    speakerDeviceId: '',
    ttsEngine: 'pyttsx3',
    ttsVoice: '',
    speechRate: 1.0,
    volume: 1.0,
    sttModel: 'base',
    sttLanguage: 'auto',
    sttDevice: 'auto',
    wakeWord: '',
    wakeWordBackend: 'openwakeword',
    customWakeWordId: '',
    wakeWordPhrase: 'hey rex',
    wakeWordModelPath: '',
    wakeWordEmbeddingPath: ''
  } satisfies VoiceSettings,
  ai: {
    model: 'claude-sonnet-4',
    provider: 'openai',
    customModelId: '',
    ollamaBaseUrl: 'http://localhost:11434',
    temperature: 0.7,
    maxTokens: 2048,
    systemPrompt: '',
    autonomyMode: 'manual',
    budgetPerPlan: 0,
    budgetPerStep: 0,
    modelRouting: normalizeAiModelRouting({}),
    personality: 'Friendly'
  } satisfies AiSettings as unknown as Settings,
  users: {
    names: {}
  },
  integrations: {
    emailProvider: 'gmail',
    emailClientId: '',
    emailClientSecret: '', // pragma: allowlist secret
    emailAccounts: [] as EmailAccount[],
    calendarProvider: 'gmail',
    calendarClientId: '',
    calendarClientSecret: '', // pragma: allowlist secret
    smsSid: '',
    smsAuthToken: '',
    smsFromNumber: '',
    haUrl: '',
    haToken: '',
    phoneSid: '',
    phoneAuthToken: '',
    phoneNumber: '',
    phoneTransferNumber: '',
    voicemailNotificationsEnabled: false,
    contactsFilePath: '',
    telegramBotToken: '',
    telegramChatId: ''
  } satisfies IntegrationsSettings,
  system: {
    autonomyMode: 'manual',
    toolTimeoutSeconds: 10,
    requireConfirmSystemChanges: true,
    allowedFileRoots: '',
    debugLogging: false
  } satisfies SystemSettings
}
