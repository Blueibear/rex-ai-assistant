import React, { useEffect, useState } from 'react'
import { SkeletonLine } from './SkeletonLine'

interface PageLoadingFallbackProps {
  lines?: number
  className?: string
}

export function PageLoadingFallback({
  lines = 5,
  className = ''
}: PageLoadingFallbackProps): React.ReactElement {
  const [slow, setSlow] = useState(false)

  useEffect(() => {
    const id = setTimeout(() => setSlow(true), 5000)
    return () => clearTimeout(id)
  }, [])

  const widths = ['100%', '100%', '60%', '100%', '80%', '100%', '45%', '100%', '70%']

  return (
    <div className={`p-6 space-y-3 ${className}`}>
      {Array.from({ length: lines }, (_, i) => (
        <SkeletonLine
          key={i}
          width={widths[i % widths.length]}
          height={i === 0 ? '1.25rem' : '1rem'}
        />
      ))}
      {slow && (
        <p className="text-xs text-text-secondary mt-2 animate-pulse">
          Taking longer than expected…
        </p>
      )}
    </div>
  )
}

export default PageLoadingFallback
