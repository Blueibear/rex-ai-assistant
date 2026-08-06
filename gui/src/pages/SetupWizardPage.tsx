import React, { useState } from 'react'
import { buildSetupSubmission } from './setupWizardModel'

interface SetupData {
  username: string
  password: string
  llmProvider: string
  llmApiKey: string
  ttsProvider: string
  haBaseUrl: string
  haToken: string
}

interface StepProps {
  data: SetupData
  onChange: (field: keyof SetupData, value: string) => void
}

function StepAccount({ data, onChange }: StepProps): React.ReactElement {
  return (
    <div className="space-y-4">
      <p className="text-text-secondary text-sm">
        Create your Rex account. This account is stored locally — no cloud sign-up required.
      </p>
      <div>
        <label className="block text-sm font-medium text-text-primary mb-1">Username</label>
        <input
          type="text"
          autoComplete="username"
          value={data.username}
          onChange={(e) => onChange('username', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
          placeholder="e.g. james"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-text-primary mb-1">Password</label>
        <input
          type="password"
          autoComplete="new-password"
          value={data.password}
          onChange={(e) => onChange('password', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
          placeholder="Choose a strong password"
        />
      </div>
    </div>
  )
}

function StepLLM({ data, onChange }: StepProps): React.ReactElement {
  return (
    <div className="space-y-4">
      <p className="text-text-secondary text-sm">
        Choose how Rex generates responses. You can change this later in Settings.
      </p>
      <div>
        <label className="block text-sm font-medium text-text-primary mb-1">LLM Provider</label>
        <select
          value={data.llmProvider}
          onChange={(e) => onChange('llmProvider', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
        >
          <option value="local">Local (Transformers / Ollama)</option>
          <option value="openai">OpenAI</option>
          <option value="openrouter">OpenRouter</option>
          <option value="anthropic">Anthropic</option>
          <option value="ollama">Ollama (custom URL)</option>
        </select>
      </div>
      {(data.llmProvider === 'openai' || data.llmProvider === 'openrouter' || data.llmProvider === 'anthropic') && (
        <div>
          <label className="block text-sm font-medium text-text-primary mb-1">API Key</label>
          <input
            type="password"
            autoComplete="off"
            value={data.llmApiKey}
            onChange={(e) => onChange('llmApiKey', e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
            placeholder={data.llmProvider === 'anthropic' ? 'sk-ant-...' : 'Enter API key'}
          />
          <p className="text-text-muted text-xs mt-1">Stored in the Windows credential vault.</p>
        </div>
      )}
    </div>
  )
}

function StepTTS({ data, onChange }: StepProps): React.ReactElement {
  return (
    <div className="space-y-4">
      <p className="text-text-secondary text-sm">
        Choose a text-to-speech engine for Rex&apos;s voice. You can disable TTS if you prefer
        silent operation.
      </p>
      <div>
        <label className="block text-sm font-medium text-text-primary mb-1">TTS Provider</label>
        <select
          value={data.ttsProvider}
          onChange={(e) => onChange('ttsProvider', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
        >
          <option value="none">None (text only)</option>
          <option value="edge">Edge TTS (free, requires internet)</option>
          <option value="pyttsx3">pyttsx3 (offline, system voices)</option>
          <option value="xtts">Coqui XTTS (offline, high quality)</option>
        </select>
      </div>
    </div>
  )
}

function StepHomeAssistant({ data, onChange }: StepProps): React.ReactElement {
  return (
    <div className="space-y-4">
      <p className="text-text-secondary text-sm">
        Connect to Home Assistant to control smart home devices. Skip this step if you don&apos;t
        use Home Assistant.
      </p>
      <div>
        <label className="block text-sm font-medium text-text-primary mb-1">
          Home Assistant URL
        </label>
        <input
          type="url"
          value={data.haBaseUrl}
          onChange={(e) => onChange('haBaseUrl', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
          placeholder="http://homeassistant.local:8123"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-text-primary mb-1">
          Long-Lived Access Token
        </label>
        <input
          type="password"
          autoComplete="off"
          value={data.haToken}
          onChange={(e) => onChange('haToken', e.target.value)}
          className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
          placeholder="eyJ..."
        />
        <p className="text-text-muted text-xs mt-1">
          Generate in HA → Profile → Long-Lived Access Tokens.
        </p>
      </div>
    </div>
  )
}

function StepDone(): React.ReactElement {
  return (
    <div className="space-y-4 text-center">
      <div className="w-16 h-16 mx-auto rounded-full bg-green-500/15 flex items-center justify-center">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true">
          <path
            d="M6 16l7 7L26 9"
            stroke="#22c55e"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      <h3 className="text-lg font-semibold text-text-primary">Rex is ready!</h3>
      <p className="text-text-secondary text-sm">
        Your account has been created and Rex has been configured. Click Finish to open the
        dashboard.
      </p>
    </div>
  )
}

const STEPS = ['Account', 'LLM', 'Voice', 'Home Assistant', 'Done']

interface SetupWizardPageProps {
  onComplete: () => void
}

export function SetupWizardPage({ onComplete }: SetupWizardPageProps): React.ReactElement {
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [data, setData] = useState<SetupData>({
    username: '',
    password: '',
    llmProvider: 'local',
    llmApiKey: '',
    ttsProvider: 'none',
    haBaseUrl: '',
    haToken: ''
  })

  const handleChange = (field: keyof SetupData, value: string): void => {
    setData((prev) => ({ ...prev, [field]: value }))
    setError('')
  }

  const validate = (): string => {
    if (step === 0) {
      if (!data.username.trim()) return 'Username is required.'
      if (data.password.length < 8) return 'Password must be at least 8 characters.'
    }
    return ''
  }

  const handleNext = (): void => {
    const err = validate()
    if (err) {
      setError(err)
      return
    }
    if (step < STEPS.length - 2) {
      setStep((s) => s + 1)
    } else {
      void handleSubmit()
    }
  }

  const handleSubmit = async (deferHomeAssistant = false): Promise<void> => {
    setSubmitting(true)
    setError('')
    try {
      const submission = buildSetupSubmission(data, { deferHomeAssistant })
      const result = await window.rex.completeSetup(submission)
      if (!result.ok) {
        setError(result.error ?? 'Setup failed.')
        setSubmitting(false)
        return
      }
      setStep(STEPS.length - 1) // Done step
    } catch {
      setError('Setup failed. Please try again.')
      setSubmitting(false)
    }
  }

  const renderStep = (): React.ReactElement => {
    switch (step) {
      case 0:
        return <StepAccount data={data} onChange={handleChange} />
      case 1:
        return <StepLLM data={data} onChange={handleChange} />
      case 2:
        return <StepTTS data={data} onChange={handleChange} />
      case 3:
        return <StepHomeAssistant data={data} onChange={handleChange} />
      default:
        return <StepDone />
    }
  }

  const isDone = step === STEPS.length - 1

  return (
    <div className="flex items-center justify-center min-h-screen bg-bg p-4">
      <div className="w-full max-w-md">
        {/* Progress bar */}
        <div className="flex items-center gap-1 mb-8">
          {STEPS.slice(0, -1).map((label, i) => (
            <React.Fragment key={label}>
              <div
                className={`flex-1 h-1 rounded-full transition-colors ${
                  i <= step ? 'bg-accent' : 'bg-surface-raised'
                }`}
              />
            </React.Fragment>
          ))}
        </div>

        <div className="bg-surface border border-border rounded-2xl p-8 space-y-6">
          {/* Header */}
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
                Step {isDone ? STEPS.length : step + 1} of {STEPS.length}
              </span>
            </div>
            <h2 className="text-xl font-semibold text-text-primary">
              {isDone ? 'All done!' : `Set up ${STEPS[step]}`}
            </h2>
          </div>

          {/* Step content */}
          {renderStep()}

          {/* Error */}
          {error && <p className="text-red-400 text-sm">{error}</p>}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            {!isDone && step > 0 ? (
              <button
                type="button"
                onClick={() => setStep((s) => s - 1)}
                className="text-sm text-text-secondary hover:text-text-primary transition-colors"
              >
                ← Back
              </button>
            ) : (
              <span />
            )}

            {isDone ? (
              <button
                type="button"
                onClick={onComplete}
                className="px-6 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors"
              >
                Open Dashboard
              </button>
            ) : (
              <div className="flex items-center gap-2">
                {step === 3 && (
                  <button
                    type="button"
                    onClick={() => {
                      void handleSubmit(true)
                    }}
                    disabled={submitting}
                    className="text-sm text-text-secondary hover:text-text-primary transition-colors"
                  >
                    Do this later
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleNext}
                  disabled={submitting}
                  className="px-6 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors disabled:opacity-50"
                >
                  {submitting ? 'Setting up…' : step === STEPS.length - 2 ? 'Finish' : 'Next →'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
