import { describe, it, expect } from 'vitest'
import { buildSetupSubmission } from '../src/pages/setupWizardModel'

describe('setupWizardModel', () => {
  describe('buildSetupSubmission', () => {
    it('returns basic submission without options', () => {
      const result = buildSetupSubmission({
        username: 'alice',
        password: 'pass1234',
        llmProvider: 'local',
        llmApiKey: '',
        ttsProvider: 'none',
        haBaseUrl: 'http://ha.local:8123',
        haToken: 'secret-token'
      })

      expect(result.username).toBe('alice')
      expect(result.password).toBe('pass1234')
      expect(result.llm_provider).toBe('local')
      expect(result.tts_provider).toBe('none')
      expect(result.ha_base_url).toBe('http://ha.local:8123')
      expect(result.ha_token).toBe('secret-token')
    })

    it('returns empty HA fields when deferHomeAssistant is true', () => {
      const result = buildSetupSubmission(
        {
          username: 'bob',
          password: 'pass1234',
          llmProvider: 'openai',
          llmApiKey: 'sk-test',
          ttsProvider: 'edge',
          haBaseUrl: 'http://ha.local:8123',
          haToken: 'secret-token'
        },
        { deferHomeAssistant: true }
      )

      expect(result.username).toBe('bob')
      expect(result.ha_base_url).toBe('')
      expect(result.ha_token).toBe('')
      expect(result.defer_home_assistant).toBe(true)
    })

    it('includes llm_api_key when provided', () => {
      const result = buildSetupSubmission({
        username: 'carol',
        password: 'pass1234',
        llmProvider: 'openai',
        llmApiKey: 'sk-test-key',
        ttsProvider: 'none',
        haBaseUrl: '',
        haToken: ''
      })

      expect(result.llm_api_key).toBe('sk-test-key')
    })

    it('does not include llm_api_key when empty', () => {
      const result = buildSetupSubmission({
        username: 'dave',
        password: 'pass1234',
        llmProvider: 'local',
        llmApiKey: '',
        ttsProvider: 'none',
        haBaseUrl: '',
        haToken: ''
      })

      expect(result.llm_api_key).toBeUndefined()
    })

    it('respects deferHomeAssistant=false explicitly', () => {
      const result = buildSetupSubmission(
        {
          username: 'eve',
          password: 'pass1234',
          llmProvider: 'local',
          llmApiKey: '',
          ttsProvider: 'none',
          haBaseUrl: 'http://ha.local:8123',
          haToken: 'token'
        },
        { deferHomeAssistant: false }
      )

      expect(result.ha_base_url).toBe('http://ha.local:8123')
      expect(result.ha_token).toBe('token')
      expect(result.defer_home_assistant).toBe(false)
    })
  })
})
