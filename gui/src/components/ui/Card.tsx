import React from 'react'

export interface CardProps {
  children: React.ReactNode
  header?: React.ReactNode
  padding?: 'sm' | 'md' | 'lg' | 'none'
  hoverable?: boolean
  className?: string
}

const paddingClasses: Record<NonNullable<CardProps['padding']>, string> = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6'
}

export const Card: React.FC<CardProps> = ({
  children,
  header,
  padding = 'md',
  hoverable = false,
  className = ''
}) => {
  return (
    <div
      className={[
        'bg-surface border border-border rounded-lg',
        hoverable
          ? 'cursor-pointer motion-safe:transition-all motion-safe:duration-150 motion-safe:hover:-translate-y-0.5 motion-safe:hover:shadow-lg'
          : '',
        className
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {header && (
        <div className="px-4 py-3 border-b border-border font-medium text-text-primary">
          {header}
        </div>
      )}
      <div className={paddingClasses[padding]}>{children}</div>
    </div>
  )
}

export default Card
