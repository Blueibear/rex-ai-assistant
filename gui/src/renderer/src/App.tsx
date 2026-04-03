import React, { useEffect, useState } from 'react'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Spinner } from '../../components/ui/Spinner'
import { SkeletonLine } from '../../components/ui/SkeletonLine'
import { EmptyState } from '../../components/ui/EmptyState'
import { useToast } from '../../components/ui/Toast'
import type { ToastType } from '../../components/ui/Toast'
import type { NotificationPriority } from '../../types/ipc'
import { AppLayout } from '../../layouts/AppLayout'
import { PageTransition } from '../../components/ui/PageTransition'
import { ChatPage } from '../../pages/ChatPage'
import { VoicePage } from '../../pages/VoicePage'
import { TasksPage } from '../../pages/TasksPage'
import { CalendarPage } from '../../pages/CalendarPage'
import { RemindersPage } from '../../pages/RemindersPage'
import { MemoriesPage } from '../../pages/MemoriesPage'
import { EmailPage } from '../../pages/EmailPage'
import { SmsPage } from '../../pages/SmsPage'
import { NotificationsPage } from '../../pages/NotificationsPage'
import { SettingsPage } from '../../pages/SettingsPage'
import { ShoppingListPage } from '../../pages/ShoppingListPage'
import { LogsPage } from '../../pages/LogsPage'
import { ErrorBoundary } from '../../components/ErrorBoundary'

const PRIORITY_TOAST_TYPE: Record<NotificationPriority, ToastType> = {
  critical: 'error',
  high: 'warning',
  medium: 'info',
  low: 'info'
}

function AppShell(): React.ReactElement {
  const [status, setStatus] = useState<string>('loading…')
  const addToast = useToast()

  useEffect(() => {
    window.rex
      .getStatus()
      .then((res) => {
        const s = res.status ?? 'unknown'
        setStatus(s)
        addToast(`Rex status: ${s}`, 'info')
      })
      .catch(() => {
        setStatus('error')
        addToast('Could not reach Rex backend', 'error')
      })
  }, [addToast])

  // Listen for push notifications forwarded by the main process (e.g. from the
  // Rex Python backend). Only critical and high are forwarded (see notifications.ts).
  useEffect(() => {
    window.rex.onNewNotification((notification) => {
      const toastType = PRIORITY_TOAST_TYPE[notification.priority]
      addToast(notification.title, toastType)
    })
  }, [addToast])

  if (status === 'loading…') {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-bg gap-4">
        <Spinner size="lg" />
        <SkeletonLine width="200px" height="16px" />
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="flex items-center justify-center h-screen bg-bg">
        <EmptyState
          icon="⚠"
          heading="Rex backend unavailable"
          subtext="Start the Rex Python process and relaunch."
        />
      </div>
    )
  }

  return (
    <AppLayout>
      <PageTransition>
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ErrorBoundary><ChatPage /></ErrorBoundary>} />
          <Route path="/voice" element={<ErrorBoundary><VoicePage /></ErrorBoundary>} />
          <Route path="/tasks" element={<ErrorBoundary><TasksPage /></ErrorBoundary>} />
          <Route path="/calendar" element={<ErrorBoundary><CalendarPage /></ErrorBoundary>} />
          <Route path="/reminders" element={<ErrorBoundary><RemindersPage /></ErrorBoundary>} />
          <Route path="/memories" element={<ErrorBoundary><MemoriesPage /></ErrorBoundary>} />
          <Route path="/email" element={<ErrorBoundary><EmailPage /></ErrorBoundary>} />
          <Route path="/sms" element={<ErrorBoundary><SmsPage /></ErrorBoundary>} />
          <Route path="/notifications" element={<ErrorBoundary><NotificationsPage /></ErrorBoundary>} />
          <Route path="/shopping" element={<ErrorBoundary><ShoppingListPage /></ErrorBoundary>} />
          <Route path="/logs" element={<ErrorBoundary><LogsPage /></ErrorBoundary>} />
          <Route path="/settings" element={<ErrorBoundary><SettingsPage /></ErrorBoundary>} />
        </Routes>
      </PageTransition>
    </AppLayout>
  )
}

function App(): React.ReactElement {
  return (
    <HashRouter>
      <AppShell />
    </HashRouter>
  )
}

export default App
