export type RoutingFallbackMode = 'none' | 'named' | 'ask'

export interface OutputQuietHours {
  enabled: boolean
  start_local_time: string
  end_local_time: string
  days_of_week: number[]
}

export interface OutputRoutingRule {
  output_kind: 'spoken_response' | 'timer' | 'alarm' | 'media'
  target_id: string
  days_of_week: number[]
  start_local_time: string | null
  end_local_time: string | null
  target_volume: number | null
  fallback_mode: RoutingFallbackMode | null
  fallback_target_id: string | null
}

export interface OutputRoutingPolicy {
  spoken_response_target_id: string | null
  timer_target_id: string | null
  alarm_target_id: string | null
  media_target_id: string | null
  spoken_response_fallback: RoutingFallbackMode
  timer_fallback: RoutingFallbackMode
  alarm_fallback: RoutingFallbackMode
  media_fallback: RoutingFallbackMode
  spoken_response_fallback_target_id: string | null
  timer_fallback_target_id: string | null
  alarm_fallback_target_id: string | null
  media_fallback_target_id: string | null
  spoken_response_volume: number | null
  timer_volume: number | null
  alarm_volume: number | null
  media_volume: number | null
  prefer_media_request_origin: boolean
  default_media_provider: string | null
  default_media_account_id: string | null
  quiet_hours: OutputQuietHours
  rules: OutputRoutingRule[]
}

export interface MediaAccountInfo {
  provider: string
  account_id: string
  display_name: string
}

export interface OutputRoutingResponse {
  ok: boolean
  policy?: OutputRoutingPolicy
  accounts?: MediaAccountInfo[]
  target_id?: string
  error?: string
}

declare module './ipc' {
  interface RexAPI {
    getOutputRoutingPolicy: () => Promise<OutputRoutingResponse>
    updateOutputRoutingPolicy: (policy: OutputRoutingPolicy) => Promise<OutputRoutingResponse>
    listMediaAccounts: () => Promise<OutputRoutingResponse>
    testOutputRoutingTarget: (targetId: string) => Promise<OutputRoutingResponse>
  }
}
