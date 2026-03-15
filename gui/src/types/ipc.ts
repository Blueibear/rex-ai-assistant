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

export interface RexAPI {
  sendChat: (message: string) => Promise<string>
  sendChatStream: (message: string, onToken: (token: string) => void) => Promise<void>
  getStatus: () => Promise<StatusResponse>
  getSettings: () => Promise<SettingsResponse>
  setSettings: (settings: Settings) => Promise<SetSettingsResponse>
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
}
