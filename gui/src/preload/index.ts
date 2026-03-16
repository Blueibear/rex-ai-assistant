import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type {
  Settings,
  VoiceTranscriptEntry,
  Task,
  TaskInput,
  TaskRun,
  CalendarEvent,
  CalendarEventInput,
  Reminder,
  ReminderInput,
  Memory,
  MemoryUpdateInput
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
  getSettings: () => ipcRenderer.invoke('rex:getSettings'),
  setSettings: (settings: Settings) => ipcRenderer.invoke('rex:setSettings', settings),
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
  updateMemory: (id: string, data: MemoryUpdateInput): Promise<Memory> =>
    ipcRenderer.invoke('rex:updateMemory', id, data)
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
