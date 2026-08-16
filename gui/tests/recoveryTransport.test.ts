import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const root = join(__dirname, '..')
const bridge = readFileSync(join(root, '../bridge/rex_chat_stream_bridge.py'), 'utf8')
const handler = readFileSync(join(root, 'src/main/handlers/chat.ts'), 'utf8')
const preload = readFileSync(join(root, 'src/preload/index.ts'), 'utf8')
const chatPage = readFileSync(join(root, 'src/pages/ChatPage.tsx'), 'utf8')

describe('structured chat recovery transport (US-078)', () => {
  it('carries recovery metadata from Python through Electron to the renderer', () => {
    expect(bridge).toContain('"type": "recovery"')
    expect(handler).toContain("event.sender.send('rex:chatRecovery'")
    expect(handler).toContain("obj.type === 'recovery'")
    expect(preload).toContain("ipcRenderer.on('rex:chatRecovery'")
    expect(preload).toContain("removeListener('rex:chatRecovery'")
    expect(chatPage).toContain('recoveryActions: recovery.actions')
  })
})
