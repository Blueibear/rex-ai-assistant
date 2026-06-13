export interface StatusResponse {
  ok: boolean
  status?: string
}

export interface HomeAssistantConnectionResponse {
  ok: boolean
  error?: string
}

export interface HomeAssistantState {
  entity_id: string
  state: string
  friendly_name: string
  last_updated: string
}

export interface HomeAssistantStatesResponse extends HomeAssistantConnectionResponse {
  states?: HomeAssistantState[]
  not_configured?: boolean
}

export interface Settings {
  [key: string]: unknown
}

export interface SettingsResponse {
  ok: boolean
  settings?: Settings
}

export interface SetSettingsResponse {
  ok: boolean
}

export interface VoiceInfo {
  id: string
  name: string
  language: string
  gender: string | null
  engine?: string
}

export interface WakeWordInfo {
  id: string
  name: string
  engine: string
  has_sample?: boolean
  model_path?: string
}

export type WakeWordBackend = 'openwakeword' | 'custom_onnx' | 'custom_embedding'

export interface WakeWordStatus {
  requestedBackend: WakeWordBackend
  configuredPhrase: string
  fallbackEnabled: boolean
  fallbackKeyword: string
  assetKind: 'builtin' | 'onnx' | 'embedding'
  assetPath: string
  assetExists: boolean
  fallbackActive: boolean
  status: 'built_in' | 'asset_ready' | 'missing_asset'
  statusLabel: string
  detail: string
}

export interface VoiceTranscriptEntry {
  text: string
  role: 'user' | 'rex'
  timestamp: number
}

export interface TaskRun {
  id: string
  taskId: string
  timestamp: string
  result: 'success' | 'failed'
  output: string[]
}

export interface Task {
  id: string
  name: string
  prompt: string
  schedule: string
  nextRun: string
  status: 'active' | 'paused' | 'error'
  lastRun?: { timestamp: string; result: 'success' | 'failed' }
}

export interface TaskInput {
  id?: string
  name: string
  prompt: string
  schedule: string
  active: boolean
}

export interface CalendarEvent {
  id: string
  title: string
  start: string // ISO date string
  end: string
  color?: string
  location?: string
  description?: string
  attendees?: string[]
  source?: 'rex' | 'synced'
}

export interface CalendarEventInput {
  title: string
  start: string // ISO date string
  end: string
  color?: string
  location?: string
  description?: string
}

export interface Reminder {
  id: string
  title: string
  notes?: string
  dueAt: string // ISO date string
  priority: 'low' | 'medium' | 'high'
  done: boolean
  repeat?: 'none' | 'daily' | 'weekly' | 'custom'
}

export interface ReminderInput {
  id?: string
  title: string
  notes?: string
  dueAt: string // ISO date string
  priority: 'low' | 'medium' | 'high'
  repeat: 'none' | 'daily' | 'weekly' | 'custom'
}

export interface Memory {
  id: string
  text: string
  category: string
  createdAt: string // ISO date string
  updatedAt: string // ISO date string
}

export interface MemoryUpdateInput {
  text: string
  category: string
}

export interface SystemSettings {
  autonomyMode: 'manual' | 'supervised' | 'full-auto'
  toolTimeoutSeconds: number
  requireConfirmSystemChanges: boolean
  allowedFileRoots: string
  debugLogging: boolean
}

export interface GeneralSettings {
  displayName: string
  timezone: string
  language: string
  launchAtLogin: boolean
  startMinimized: boolean
}

export interface VoiceSettings {
  microphoneDeviceId: string
  speakerDeviceId: string
  ttsEngine: 'system' | 'openai' | 'elevenlabs' | 'xtts' | 'edge-tts' | 'pyttsx3'
  ttsVoice: string
  speechRate: number
  volume: number
  sttModel: string
  sttLanguage: string
  sttDevice: 'auto' | 'cpu' | 'cuda'
  wakeWord: string
  wakeWordBackend: WakeWordBackend
  customWakeWordId: string
  wakeWordPhrase: string
  wakeWordModelPath: string
  wakeWordEmbeddingPath: string
}

export interface SmartSpeaker {
  provider: string
  name: string
  ip: string
  model: string
}

export interface VoiceEnrollment {
  user_id: string
  sample_count: number
  updated_at: string
  model_id: string
}

export interface AiModelRoutingSettings {
  default: string
  coding: string
  reasoning: string
  search: string
  vision: string
  fast: string
}

export interface AiSettings {
  model: 'gpt-4o' | 'gpt-4-turbo' | 'claude-opus-4' | 'claude-sonnet-4' | 'gemini-1.5-pro'
  provider: 'openai' | 'ollama' | 'local'
  customModelId: string
  ollamaBaseUrl: string
  temperature: number
  maxTokens: number
  systemPrompt: string
  autonomyMode: 'manual' | 'supervised' | 'full-auto'
  budgetPerPlan: number
  budgetPerStep: number
  modelRouting: AiModelRoutingSettings
  personality: string
}

export interface EmailAccount {
  id: string
  backend: 'imap' | 'gmail' | 'outlook'
  displayName: string
  // OAuth fields (gmail / outlook)
  clientId: string
  clientSecret: string
  // IMAP fields
  host: string
  port: number
  username: string
  password: string
  // State
  lastSynced?: string
}

export interface IntegrationsSettings {
  emailProvider: 'gmail' | 'outlook'
  emailClientId: string
  emailClientSecret: string
  emailAccounts: EmailAccount[]
  calendarProvider: 'gmail' | 'outlook'
  calendarClientId: string
  calendarClientSecret: string
  smsSid: string
  smsAuthToken: string
  smsFromNumber: string
  haUrl: string
  haToken: string
  // Phone / Twilio (US-PH-004)
  phoneSid: string
  phoneAuthToken: string
  phoneNumber: string
  phoneTransferNumber: string
  voicemailNotificationsEnabled: boolean
  contactsFilePath: string
  // Telegram
  telegramBotToken: string
  telegramChatId: string
}

export type IntegrationConnectionStatus = 'untested' | 'connected' | 'error'

export interface IntegrationStatus {
  status: IntegrationConnectionStatus
  testedAt?: string
  error?: string
}

export interface IntegrationInventoryItem extends IntegrationStatus {
  name: string
  key: string
  configured: boolean
  configure_url?: string
  testable?: boolean
}

export interface IntegrationInventoryResponse {
  ok: boolean
  integrations: IntegrationInventoryItem[]
  error?: string
}

export interface CapabilityInfo {
  name: string
  description: string
  category: string
  enabled: boolean
}

export interface CapabilitiesResponse {
  ok: boolean
  capabilities: CapabilityInfo[]
  error?: string
}

export interface NotificationsSettings {
  quietHoursEnabled: boolean
  quietHoursStart: string
  quietHoursEnd: string
  digestModeEnabled: boolean
  digestDeliveryTime: string
  highPriorityThreshold: 'critical_only' | 'high_and_critical'
  autoEscalationDelay: number
  desktopNotificationsEnabled: boolean
  soundAlertsEnabled: boolean
}

export interface VersionInfo {
  rex: string
  electron: string
  node: string
}

export interface AppStatus {
  version: string
  python_version: string
  platform: string
}

export interface CommandHistoryEntry {
  id: number
  timestamp: string
  command: string
  result: string
  success: boolean
}

export interface ShoppingItem {
  id: string
  name: string
  quantity: number
  unit: string
  added_by: string
  checked: boolean
  added_at: string
  checked_at: string | null
}

export interface FileExtractResult {
  ok: boolean
  isImage: boolean
  extractedText?: string
  error?: string
}

export interface PreferenceSuggestion {
  field: string
  current_value: string | number
  suggested_value: string | number
  reason: string
}

export type EmailPriority = 'low' | 'medium' | 'high' | 'critical'

export interface EmailMessage {
  id: string
  thread_id: string
  subject: string
  sender: string
  recipients: string[]
  body_text: string
  body_html?: string
  received_at: string // ISO date string
  labels: string[]
  is_read: boolean
  priority: EmailPriority
}

export type SMSDirection = 'inbound' | 'outbound'
export type SMSStatus = 'sent' | 'delivered' | 'failed' | 'stub'

export interface SMSMessage {
  id: string
  thread_id: string
  direction: SMSDirection
  body: string
  from_number: string
  to_number: string
  sent_at: string // ISO date string
  status: SMSStatus
}

// ---------------------------------------------------------------------------
// Notifications
// ---------------------------------------------------------------------------

export type NotificationPriority = 'low' | 'medium' | 'high' | 'critical'
export type NotificationChannel = 'desktop' | 'digest' | 'sms' | 'email'

export interface GuiNotification {
  id: string
  title: string
  body: string
  source: string
  priority: NotificationPriority
  channel: NotificationChannel
  digest_eligible: boolean
  quiet_hours_exempt: boolean
  created_at: string // ISO date string
  delivered_at?: string
  read_at?: string
  escalation_due_at?: string
  action_url?: string
  action_label?: string
}

export interface SMSThread {
  id: string
  contact_name: string
  contact_number: string
  messages: SMSMessage[]
  last_message_at: string // ISO date string
  unread_count: number
}

export interface TimeSlot {
  start: string // ISO date string
  end: string   // ISO date string
  confidence: number // 0–1
}

export interface FindMeetingSlotsParams {
  durationMinutes: number
  earliest: string  // ISO date string
  latest: string    // ISO date string
  timezone: string
}


export interface LogEntry {
  timestamp: string
  level: string
  logger: string
  message: string
  extra: Record<string, unknown>
  raw?: string
}

export interface LogsResponse {
  ok: boolean
  entries: LogEntry[]
  log_path?: string
  legacy_log_path?: string
  log_source?: string
  timestamp_basis?: string
  error?: string
}

export interface UsageBucket {
  requests: number
  tokens: number
}

export interface UsagePeriodSplit {
  local: UsageBucket
  cloud: UsageBucket
}

export interface UsageSummary {
  ok: boolean
  local: UsageBucket
  cloud: UsageBucket
  by_period: {
    today: UsagePeriodSplit
    week: UsagePeriodSplit
    month: UsagePeriodSplit
  }
  error?: string
}

export interface RexAPI {
  sendChat: (message: string) => Promise<string>
  sendChatStream: (message: string, onToken: (token: string) => void) => Promise<void>
  getStatus: () => Promise<StatusResponse>
  onStatusChange: (cb: (status: string) => void) => (() => void)
  getSettings: (section: string) => Promise<Settings>
  setSettings: (section: string, values: Settings) => Promise<SetSettingsResponse>
  startVoice: (
    onStateChange: (state: string) => void,
    onTranscript: (entry: VoiceTranscriptEntry) => void,
    onError: (error: string) => void,
    onStatus?: (status: string, label: string) => void
  ) => Promise<void>
  attachVoiceSession: (
    onStateChange: (state: string) => void,
    onTranscript: (entry: VoiceTranscriptEntry) => void,
    onError: (error: string) => void,
    onStatus?: (status: string, label: string) => void
  ) => (() => void)
  stopVoice: () => Promise<void>
  getTasks: () => Promise<Task[]>
  saveTask: (task: TaskInput) => Promise<Task>
  deleteTask: (taskId: string) => Promise<void>
  setTaskEnabled: (taskId: string, enabled: boolean) => Promise<Task>
  getTaskHistory: (taskId: string) => Promise<TaskRun[]>
  getCalendarEvents: (start: string, end: string) => Promise<CalendarEvent[]>
  createCalendarEvent: (event: CalendarEventInput) => Promise<CalendarEvent>
  updateCalendarEvent: (event: CalendarEvent) => Promise<CalendarEvent>
  deleteCalendarEvent: (id: string) => Promise<void>
  getReminders: () => Promise<Reminder[]>
  completeReminder: (id: string) => Promise<void>
  saveReminder: (reminder: ReminderInput) => Promise<Reminder>
  deleteReminder: (id: string) => Promise<void>
  getMemories: () => Promise<Memory[]>
  addMemory: (data: MemoryUpdateInput) => Promise<Memory>
  updateMemory: (id: string, data: MemoryUpdateInput) => Promise<Memory>
  deleteMemory: (id: string) => Promise<void>
  getVersionInfo: () => Promise<VersionInfo>
  getAppStatus: () => Promise<AppStatus>
  getCommandHistory: (
    limit?: number
  ) => Promise<{ ok: boolean; history: CommandHistoryEntry[]; error?: string }>
  testVoice: (settings: VoiceSettings) => Promise<{ ok: boolean; error?: string }>
  testIntegration: (type: 'email' | 'calendar' | 'sms' | 'homeassistant' | 'phone') => Promise<{ ok: boolean; error?: string }>
  getIntegrations: () => Promise<IntegrationInventoryResponse>
  getCapabilities: () => Promise<CapabilitiesResponse>
  testHomeAssistant: (baseUrl: string, token: string) => Promise<HomeAssistantConnectionResponse>
  saveHomeAssistant: (baseUrl: string, token: string) => Promise<HomeAssistantConnectionResponse>
  getHomeAssistantStates: () => Promise<HomeAssistantStatesResponse>
  uploadContactsFile: () => Promise<{ ok: boolean; path?: string; error?: string }>
  pickFolder: () => Promise<{ ok: boolean; path?: string; error?: string }>
  testEmailAccount: (id: string) => Promise<{ ok: boolean; error?: string }>
  getPreferenceSuggestions: () => Promise<PreferenceSuggestion[]>
  applyPreferenceSuggestion: (field: string, value: string | number) => Promise<{ ok: boolean }>
  getEmailInbox: () => Promise<EmailMessage[]>
  generateEmailReply: (id: string) => Promise<string>
  findMeetingSlots: (params: FindMeetingSlotsParams) => Promise<TimeSlot[]>
  getSMSThreads: () => Promise<SMSThread[]>
  getSMSThread: (threadId: string) => Promise<SMSThread>
  sendSMS: (to: string, body: string) => Promise<SMSMessage>
  getNotifications: () => Promise<GuiNotification[]>
  markNotificationRead: (id: string) => Promise<void>
  dismissNotification: (id: string) => Promise<void>
  getUnreadNotificationCount: () => Promise<number>
  onNewNotification: (cb: (notification: GuiNotification) => void) => void
  listWakeWords: () => Promise<{ ok: boolean; wake_words: WakeWordInfo[]; error?: string; warning?: string }>
  getWakeWordStatus: (settings?: VoiceSettings) => Promise<WakeWordStatus>
  previewWakeWordSample: (
    wakeWordId: string
  ) => Promise<{ ok: boolean; audio_base64?: string; has_sample?: boolean; error?: string }>
  trainWakeWord: (
    phrase: string,
    positiveSamples: number[][],
    negativeSamples: number[][]
  ) => Promise<{ ok: boolean; model_path?: string; phrase?: string; error?: string }>
  listVoices: (provider: string) => Promise<{ ok: boolean; voices: VoiceInfo[]; error?: string }>
  previewVoice: (
    provider: string,
    voiceId: string
  ) => Promise<{ ok: boolean; audio_base64?: string; error?: string }>
  uploadCustomVoice: (
    filePath: string,
    voiceName: string
  ) => Promise<{ ok: boolean; voice_id?: string; voice_name?: string; duration?: number; error?: string }>
  getVoiceEnrollments: () => Promise<{
    ok: boolean
    active_user_id: string
    enrollments: VoiceEnrollment[]
    error?: string
  }>
  enrollVoice: (
    userId: string,
    samples: number[][]
  ) => Promise<{ ok: boolean; enrollment?: VoiceEnrollment; error?: string }>
  deleteVoiceEnrollment: (userId: string) => Promise<{ ok: boolean; deleted?: boolean; error?: string }>
  sendChatAudio: (
    audioBase64: string
  ) => Promise<{ ok: boolean; transcript?: string; error?: string }>
  getApiKeys: () => Promise<{ openai_key_set: boolean }>
  setApiKey: (name: string, value: string) => Promise<{ ok: boolean; error?: string }>
  getSmartSpeakers: () => Promise<{ ok: boolean; speakers: SmartSpeaker[]; error?: string }>
  restartRex: () => Promise<{ ok: boolean; error?: string }>
  resetToDefaults: () => Promise<{ ok: boolean; error?: string }>
  extractFileForChat: (params: {
    filename: string
    dataBase64: string
    mimeType: string
    sizeBytes: number
  }) => Promise<FileExtractResult>
  getShoppingItems: () => Promise<{ ok: boolean; items: ShoppingItem[]; error?: string }>
  addShoppingItem: (name: string, quantity: number, unit: string) => Promise<{ ok: boolean; item?: ShoppingItem; error?: string }>
  checkShoppingItem: (id: string) => Promise<{ ok: boolean; error?: string }>
  uncheckShoppingItem: (id: string) => Promise<{ ok: boolean; error?: string }>
  clearCheckedShoppingItems: () => Promise<{ ok: boolean; count?: number; error?: string }>
  getLogs: (limit?: number) => Promise<LogsResponse>
  startLogTail: () => Promise<{ ok: boolean; log_path?: string; error?: string }>
  stopLogTail: () => Promise<{ ok: boolean }>
  downloadLogs: () => Promise<{ ok: boolean; content?: string; filename?: string; log_path?: string; error?: string }>
  onLogEntry: (cb: (entry: LogEntry) => void) => void
  getUsage: () => Promise<UsageSummary>
}
