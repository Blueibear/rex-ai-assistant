import React, { useState, useRef, useCallback } from 'react'

export interface TooltipProps {
  text: string
  position?: 'top' | 'bottom' | 'left' | 'right'
  children: React.ReactNode
}

const positionClasses: Record<NonNullable<TooltipProps['position']>, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2'
}

export const Tooltip: React.FC<TooltipProps> = ({
  text,
  position = 'top',
  children
}) => {
  const [visible, setVisible] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const show = useCallback(() => {
    timerRef.current = setTimeout(() => setVisible(true), 300)
  }, [])

  const hide = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    setVisible(false)
  }, [])

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {visible && (
        <span
          role="tooltip"
          className={[
            'absolute z-50 whitespace-nowrap rounded px-2 py-1 text-xs',
            'bg-surface-raised text-text-primary border border-border shadow-md pointer-events-none',
            positionClasses[position]
          ].join(' ')}
        >
          {text}
        </span>
      )}
    </span>
  )
}

export default Tooltip
