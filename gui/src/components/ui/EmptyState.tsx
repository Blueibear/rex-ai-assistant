import React from 'react'
import { Button } from './Button'

export interface EmptyStateProps {
  icon?: React.ReactNode
  heading: string
  subtext?: string
  action?: {
    label: string
    onClick: () => void
  }
  className?: string
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  heading,
  subtext,
  action,
  className = ''
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-3 py-12 text-center ${className}`}
    >
      {icon && (
        <div className="text-text-secondary text-4xl" aria-hidden="true">
          {icon}
        </div>
      )}
      <h3 className="text-text-primary text-base font-semibold">{heading}</h3>
      {subtext && <p className="text-text-secondary text-sm max-w-xs">{subtext}</p>}
      {action && (
        <Button variant="secondary" size="sm" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  )
}

export default EmptyState
