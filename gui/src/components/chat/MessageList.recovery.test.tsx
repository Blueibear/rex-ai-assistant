import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { MessageList } from './MessageList'


describe('chat recovery actions (US-078)', () => {
  it('renders structured recovery guidance without implying completion', () => {
    const html = renderToStaticMarkup(
      <MessageList
        messages={[
          {
            id: 'rex-1',
            role: 'rex',
            content: 'Weather is not configured.',
            timestamp: new Date('2026-08-16T00:00:00Z'),
            recoveryActions: [
              {
                kind: 'enable_capability',
                label: 'Configure weather_lookup',
                detail: 'Set the required configuration: weather_api_key.',
                source: 'local',
                settings_route: '/settings?section=integrations',
                requires_confirmation: true
              }
            ]
          }
        ]}
      />
    )

    expect(html).toContain('Recovery actions')
    expect(html).toContain('Configure weather_lookup')
    expect(html).toContain('weather_api_key')
    expect(html).toContain('Requires your confirmation')
    expect(html).toContain('Open settings')
    expect(html).not.toContain('enabled successfully')
  })
})
