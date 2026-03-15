export interface ChatRequest {
  message: string
}

export interface ChatResponse {
  ok: boolean
  reply?: string
}

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
  sendChat: (req: ChatRequest) => Promise<ChatResponse>
  getStatus: () => Promise<StatusResponse>
  getSettings: () => Promise<SettingsResponse>
  setSettings: (settings: Settings) => Promise<SetSettingsResponse>
}
