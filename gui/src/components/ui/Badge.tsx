import React from 'react'

export interface BadgeProps {
  variant?: 'default' | 'accent' | 'success' | 'warning' | 'danger'
  children: React.ReactNode
  className?: string
}

const variantClasses: Record<NonNullable<BadgeProps['variant']>, string> = {
  default: 'bg-surface-raised text-text-secondary border border-border',
  accent: 'bg-accent/20 text-accent border border-accent/30',
  success: 'bg-success/20 text-success border border-success/30',
  warning: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
  danger: 'bg-danger/20 text-danger border border-danger/30'
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  children,
  className = ''
}) => {
  return (
    <span
      className={[
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        variantClasses[variant],
        className
      ].join(' ')}
    >
      {children}
    </span>
  )
}

export default Badge
