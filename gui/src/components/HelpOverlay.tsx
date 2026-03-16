import React from 'react'
import { Modal } from './ui/Modal'

interface ShortcutRow {
  keys: string
  description: string
}

const shortcuts: ShortcutRow[] = [
  { keys: 'Ctrl+K', description: 'Focus chat input / go to Chat' },
  { keys: 'Ctrl+Shift+V', description: 'Toggle voice mode' },
  { keys: 'Ctrl+,', description: 'Open Settings' },
  { keys: 'Ctrl+N', description: 'New task / reminder / event (context-sensitive)' },
  { keys: '?', description: 'Toggle this help overlay' }
]

export interface HelpOverlayProps {
  onClose: () => void
}

export const HelpOverlay: React.FC<HelpOverlayProps> = ({ onClose }) => {
  return (
    <Modal title="Keyboard shortcuts" onClose={onClose}>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-text-muted uppercase tracking-wide text-xs border-b border-border">
            <th className="pb-2 pr-6 text-left font-medium">Shortcut</th>
            <th className="pb-2 text-left font-medium">Action</th>
          </tr>
        </thead>
        <tbody>
          {shortcuts.map((row) => (
            <tr key={row.keys} className="border-b border-border/50 last:border-0">
              <td className="py-2 pr-6">
                <kbd className="inline-flex items-center px-1.5 py-0.5 rounded bg-surface-raised text-text-primary font-mono text-xs border border-border">
                  {row.keys}
                </kbd>
              </td>
              <td className="py-2 text-text-secondary">{row.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Modal>
  )
}

export default HelpOverlay
