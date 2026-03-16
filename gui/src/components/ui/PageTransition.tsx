import React from 'react'
import { useLocation } from 'react-router-dom'

interface PageTransitionProps {
  children: React.ReactNode
}

/**
 * Wraps page content with a fade + upward-slide entrance animation on route change.
 * Uses CSS class `page-enter` defined in index.css.
 * The `key` on the div causes React to recreate the element on each path change,
 * which re-triggers the CSS animation.
 */
export function PageTransition({ children }: PageTransitionProps): React.ReactElement {
  const { pathname } = useLocation()
  return (
    <div key={pathname} className="page-enter h-full">
      {children}
    </div>
  )
}
