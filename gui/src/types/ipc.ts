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

export interface RexAPI {
  sendChat: (message: string) => Promise<string>
  sendChatStream: (message: string, onToken: (token: string) => void) => Promise<void>
  getStatus: () => Promise<StatusResponse>
  getSettings: () => Promise<SettingsResponse>
  setSettings: (settings: Settings) => Promise<SetSettingsResponse>
}
