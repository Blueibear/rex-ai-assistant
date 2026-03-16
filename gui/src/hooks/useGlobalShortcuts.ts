import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'

/** Returns true if the event target is a text input element */
function isTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName.toLowerCase()
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true
  if (target.isContentEditable) return true
  return false
}

export interface UseGlobalShortcutsResult {
  isHelpOpen: boolean
  closeHelp: () => void
}

export function useGlobalShortcuts(): UseGlobalShortcutsResult {
  const [isHelpOpen, setIsHelpOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const closeHelp = useCallback(() => setIsHelpOpen(false), [])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent): void => {
      const ctrl = e.ctrlKey || e.metaKey

      // Ctrl+ chords fire regardless of text input focus
      if (ctrl) {
        if (e.key === 'k' || e.key === 'K') {
          e.preventDefault()
          if (location.pathname === '/chat') {
            window.dispatchEvent(new CustomEvent('rex:focus-chat'))
          } else {
            navigate('/chat')
            window.dispatchEvent(new CustomEvent('rex:focus-chat'))
          }
          return
        }

        if (e.shiftKey && (e.key === 'v' || e.key === 'V')) {
          e.preventDefault()
          if (location.pathname === '/voice') {
            window.dispatchEvent(new CustomEvent('rex:toggle-voice'))
          } else {
            navigate('/voice')
            window.dispatchEvent(new CustomEvent('rex:toggle-voice'))
          }
          return
        }

        if (e.key === ',') {
          e.preventDefault()
          navigate('/settings')
          return
        }

        if (e.key === 'n' || e.key === 'N') {
          e.preventDefault()
          const path = location.pathname
          if (path === '/tasks') {
            window.dispatchEvent(new CustomEvent('rex:new-task'))
          } else if (path === '/reminders') {
            window.dispatchEvent(new CustomEvent('rex:new-reminder'))
          } else if (path === '/calendar') {
            window.dispatchEvent(new CustomEvent('rex:new-event'))
          } else {
            navigate('/tasks')
          }
          return
        }
      }

      // Non-ctrl shortcuts: skip when focus is in a text input
      if (isTextInput(e.target)) return

      if (e.key === '?') {
        e.preventDefault()
        setIsHelpOpen((prev) => !prev)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [navigate, location.pathname])

  return { isHelpOpen, closeHelp }
}
