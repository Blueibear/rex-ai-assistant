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

export interface SetupSubmission {
  username: string
  password: string
  llm_provider: string
  llm_api_key?: string
  tts_provider: string
  ha_base_url: string
  ha_token: string
  defer_home_assistant?: boolean
}

export function buildSetupSubmission(
  data: SetupFormData,
  options?: SetupSubmissionOptions
): SetupSubmission {
  const deferHA = options?.deferHomeAssistant ?? false

  const submission: SetupSubmission = {
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
