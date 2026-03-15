import React, { useEffect, useState } from 'react'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Spinner } from '../../components/ui/Spinner'
import { SkeletonLine } from '../../components/ui/SkeletonLine'
import { EmptyState } from '../../components/ui/EmptyState'
import { useToast } from '../../components/ui/Toast'
import { AppLayout } from '../../layouts/AppLayout'
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
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/voice" element={<VoicePage />} />
        <Route path="/tasks" element={<TasksPage />} />
        <Route path="/calendar" element={<CalendarPage />} />
        <Route path="/reminders" element={<RemindersPage />} />
        <Route path="/memories" element={<MemoriesPage />} />
        <Route path="/email" element={<EmailPage />} />
        <Route path="/sms" element={<SmsPage />} />
        <Route path="/notifications" element={<NotificationsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
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
