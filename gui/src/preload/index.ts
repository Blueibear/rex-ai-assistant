import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type {
  Settings,
  SetSettingsResponse,
  VoiceTranscriptEntry,
  Task,
  TaskInput,
  TaskRun,
  CalendarEvent,
  CalendarEventInput,
  Reminder,
  ReminderInput,
  Memory,
  MemoryUpdateInput,
  VersionInfo,
  VoiceSettings,
  PreferenceSuggestion,
  EmailMessage,
  FindMeetingSlotsParams,
  TimeSlot,
  SMSThread,
  SMSMessage,
  GuiNotification
} from '../types/ipc'

function makeSendChatStream(
  message: string,
  onToken: (token: string) => void
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const streamId = `${Date.now()}-${Math.random()}`

    function tokenHandler(_e: unknown, data: { streamId: string; token: string }): void {
      if (data.streamId === streamId) onToken(data.token)
    }
    function doneHandler(_e: unknown, data: { streamId: string }): void {
      if (data.streamId === streamId) {
        cleanup()
        resolve()
      }
    }
    function errorHandler(_e: unknown, data: { streamId: string; error: string }): void {
      if (data.streamId === streamId) {
        cleanup()
        reject(new Error(data.error))
      }
    }

    function cleanup(): void {
      ipcRenderer.removeListener('rex:chatToken', tokenHandler)
      ipcRenderer.removeListener('rex:chatDone', doneHandler)
      ipcRenderer.removeListener('rex:chatError', errorHandler)
    }

    ipcRenderer.on('rex:chatToken', tokenHandler)
    ipcRenderer.on('rex:chatDone', doneHandler)
    ipcRenderer.on('rex:chatError', errorHandler)

    ipcRenderer.invoke('rex:startChatStream', { message, streamId }).catch((err: unknown) => {
      cleanup()
      reject(err)
    })
  })
}

// Module-level cleanup reference for active voice session.
let voiceCleanup: (() => void) | null = null

function makeStartVoice(
  onStateChange: (state: string) => void,
  onTranscript: (entry: VoiceTranscriptEntry) => void,
  onError: (error: string) => void
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    // Clean up any previous session listeners.
    if (voiceCleanup) {
      voiceCleanup()
      voiceCleanup = null
    }

    function stateHandler(_e: unknown, data: { state: string }): void {
      onStateChange(data.state)
    }
    function transcriptHandler(_e: unknown, data: VoiceTranscriptEntry): void {
      onTranscript(data)
    }
    function errorHandler(_e: unknown, data: { error: string }): void {
      onError(data.error)
    }

    function cleanup(): void {
      ipcRenderer.removeListener('rex:voiceState', stateHandler)
      ipcRenderer.removeListener('rex:voiceTranscript', transcriptHandler)
      ipcRenderer.removeListener('rex:voiceError', errorHandler)
      voiceCleanup = null
    }

    voiceCleanup = cleanup

    ipcRenderer.on('rex:voiceState', stateHandler)
    ipcRenderer.on('rex:voiceTranscript', transcriptHandler)
    ipcRenderer.on('rex:voiceError', errorHandler)

    ipcRenderer
      .invoke('rex:startVoice')
      .then((result: { ok: boolean; error?: string }) => {
        if (result.ok) {
          resolve()
        } else {
          cleanup()
          reject(new Error(result.error ?? 'Failed to start voice'))
        }
      })
      .catch((err: unknown) => {
        cleanup()
        reject(err)
      })
  })
}

function stopVoice(): Promise<void> {
  if (voiceCleanup) {
    voiceCleanup()
    voiceCleanup = null
  }
  return ipcRenderer.invoke('rex:stopVoice').then(() => undefined)
}

const rexAPI = {
  sendChat: (message: string) => ipcRenderer.invoke('rex:sendChat', message),
  sendChatStream: makeSendChatStream,
  getStatus: () => ipcRenderer.invoke('rex:getStatus'),
  getSettings: (section: string): Promise<Settings> =>
    ipcRenderer.invoke('rex:getSettings', section),
  setSettings: (section: string, values: Settings): Promise<SetSettingsResponse> =>
    ipcRenderer.invoke('rex:setSettings', section, values),
  startVoice: makeStartVoice,
  stopVoice,
  getTasks: (): Promise<Task[]> => ipcRenderer.invoke('rex:getTasks'),
  saveTask: (task: TaskInput): Promise<Task> => ipcRenderer.invoke('rex:saveTask', task),
  deleteTask: (taskId: string): Promise<void> => ipcRenderer.invoke('rex:deleteTask', taskId),
  setTaskEnabled: (taskId: string, enabled: boolean): Promise<Task> =>
    ipcRenderer.invoke('rex:setTaskEnabled', taskId, enabled),
  getTaskHistory: (taskId: string): Promise<TaskRun[]> =>
    ipcRenderer.invoke('rex:getTaskHistory', taskId),
  getCalendarEvents: (start: string, end: string): Promise<CalendarEvent[]> =>
    ipcRenderer.invoke('rex:getCalendarEvents', start, end),
  createCalendarEvent: (event: CalendarEventInput): Promise<CalendarEvent> =>
    ipcRenderer.invoke('rex:createCalendarEvent', event),
  updateCalendarEvent: (event: CalendarEvent): Promise<CalendarEvent> =>
    ipcRenderer.invoke('rex:updateCalendarEvent', event),
  deleteCalendarEvent: (id: string): Promise<void> =>
    ipcRenderer.invoke('rex:deleteCalendarEvent', id),
  getReminders: (): Promise<Reminder[]> => ipcRenderer.invoke('rex:getReminders'),
  completeReminder: (id: string): Promise<void> =>
    ipcRenderer.invoke('rex:completeReminder', id).then(() => undefined),
  saveReminder: (reminder: ReminderInput): Promise<Reminder> =>
    ipcRenderer.invoke('rex:saveReminder', reminder),
  deleteReminder: (id: string): Promise<void> =>
    ipcRenderer.invoke('rex:deleteReminder', id).then(() => undefined),
  getMemories: (): Promise<Memory[]> => ipcRenderer.invoke('rex:getMemories'),
  addMemory: (data: MemoryUpdateInput): Promise<Memory> =>
    ipcRenderer.invoke('rex:addMemory', data),
  updateMemory: (id: string, data: MemoryUpdateInput): Promise<Memory> =>
    ipcRenderer.invoke('rex:updateMemory', id, data),
  deleteMemory: (id: string): Promise<void> =>
    ipcRenderer.invoke('rex:deleteMemory', id).then(() => undefined),
  getVersionInfo: (): Promise<VersionInfo> => ipcRenderer.invoke('rex:getVersionInfo'),
  testVoice: (settings: VoiceSettings): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('rex:testVoice', settings),
  testIntegration: (type: 'email' | 'calendar' | 'sms'): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('rex:testIntegration', type),
  getPreferenceSuggestions: (): Promise<PreferenceSuggestion[]> =>
    ipcRenderer.invoke('rex:getPreferenceSuggestions'),
  applyPreferenceSuggestion: (field: string, value: string | number): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('rex:applyPreferenceSuggestion', field, value),
  getEmailInbox: (): Promise<EmailMessage[]> => ipcRenderer.invoke('rex:getEmailInbox'),
  generateEmailReply: (id: string): Promise<string> =>
    ipcRenderer.invoke('rex:generateEmailReply', id),
  findMeetingSlots: (params: FindMeetingSlotsParams): Promise<TimeSlot[]> =>
    ipcRenderer.invoke('rex:findMeetingSlots', params),
  getSMSThreads: (): Promise<SMSThread[]> => ipcRenderer.invoke('rex:getSMSThreads'),
  getSMSThread: (threadId: string): Promise<SMSThread> =>
    ipcRenderer.invoke('rex:getSMSThread', threadId),
  sendSMS: (to: string, body: string): Promise<SMSMessage> =>
    ipcRenderer.invoke('rex:sendSMS', to, body),
  getNotifications: (): Promise<GuiNotification[]> =>
    ipcRenderer.invoke('rex:getNotifications'),
  markNotificationRead: (id: string): Promise<void> =>
    ipcRenderer.invoke('rex:markNotificationRead', id).then(() => undefined),
  dismissNotification: (id: string): Promise<void> =>
    ipcRenderer.invoke('rex:dismissNotification', id).then(() => undefined),
  getUnreadNotificationCount: (): Promise<number> =>
    ipcRenderer.invoke('rex:getUnreadNotificationCount'),
  onNewNotification: (cb: (notification: GuiNotification) => void): void => {
    ipcRenderer.on('rex:newNotification', (_event, notification: GuiNotification) =>
      cb(notification)
    )
  }
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('rex', rexAPI)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.rex = rexAPI
}
