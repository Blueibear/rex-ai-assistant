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
}
