import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { MessageList } from './MessageList'


describe('chat model failure status', () => {
  it('labels a model failure separately from a normal assistant refusal', () => {
    const failed = renderToStaticMarkup(
      React.createElement(MessageList, {
        messages: [
          {
            id: 'failed',
            role: 'rex',
            content: "I couldn't produce a reliable response from the selected model. Please try again.",
            timestamp: new Date(0),
            status: 'model_failure',
          },
        ],
      })
    )
    const refused = renderToStaticMarkup(
      React.createElement(MessageList, {
        messages: [
          {
            id: 'refused',
            role: 'rex',
            content: "I can't help with that request.",
            timestamp: new Date(0),
          },
        ],
      })
    )
    expect(failed).toContain('Model failure')
    expect(failed).toContain('role="status"')
    expect(refused).not.toContain('Model failure')
  })
})
