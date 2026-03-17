export interface StatusResponse {
  ok: boolean
  status?: string
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
  ttsEngine: 'system' | 'openai' | 'elevenlabs'
  ttsVoice: string
  speechRate: number
  volume: number
}

export interface AiSettings {
  model: 'gpt-4o' | 'gpt-4-turbo' | 'claude-opus-4' | 'claude-sonnet-4' | 'gemini-1.5-pro'
  temperature: number
  maxTokens: number
  systemPrompt: string
  autonomyMode: 'manual' | 'supervised' | 'full-auto'
  budgetPerPlan: number
  budgetPerStep: number
}

export interface IntegrationsSettings {
  emailProvider: 'gmail' | 'outlook'
  emailClientId: string
  emailClientSecret: string
  calendarProvider: 'gmail' | 'outlook'
  calendarClientId: string
  calendarClientSecret: string
  smsSid: string
  smsAuthToken: string
  smsFromNumber: string
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

export interface RexAPI {
  sendChat: (message: string) => Promise<string>
  sendChatStream: (message: string, onToken: (token: string) => void) => Promise<void>
  getStatus: () => Promise<StatusResponse>
  getSettings: (section: string) => Promise<Settings>
  setSettings: (section: string, values: Settings) => Promise<SetSettingsResponse>
  startVoice: (
    onStateChange: (state: string) => void,
    onTranscript: (entry: VoiceTranscriptEntry) => void,
    onError: (error: string) => void
  ) => Promise<void>
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
  testVoice: (settings: VoiceSettings) => Promise<{ ok: boolean; error?: string }>
  testIntegration: (type: 'email' | 'calendar' | 'sms') => Promise<{ ok: boolean; error?: string }>
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
  listVoices: (provider: string) => Promise<{ ok: boolean; voices: VoiceInfo[]; error?: string }>
  previewVoice: (
    provider: string,
    voiceId: string
  ) => Promise<{ ok: boolean; audio_base64?: string; error?: string }>
  sendChatAudio: (
    audioBase64: string
  ) => Promise<{ ok: boolean; transcript?: string; error?: string }>
}
