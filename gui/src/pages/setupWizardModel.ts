import type { SetupCompletePayload } from '../types/ipc'

export interface SetupFormData {
  username: string
  password: string
  llmProvider: string
  llmApiKey: string
  ttsProvider: string
  haBaseUrl: string
  haToken: string
}

export interface SetupSubmissionOptions {
  deferHomeAssistant?: boolean
}

export function buildSetupSubmission(
  data: SetupFormData,
  options?: SetupSubmissionOptions
): SetupCompletePayload {
  const deferHA = options?.deferHomeAssistant ?? false

  const submission: SetupCompletePayload = {
    username: data.username,
    password: data.password,
    llm_provider: data.llmProvider,
    tts_provider: data.ttsProvider,
    ha_base_url: deferHA ? '' : data.haBaseUrl,
    ha_token: deferHA ? '' : data.haToken
  }

  if (data.llmApiKey) {
    submission.llm_api_key = data.llmApiKey
  }

  if (deferHA || options?.deferHomeAssistant === false) {
    submission.defer_home_assistant = deferHA
  }

  return submission
}
