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
}
