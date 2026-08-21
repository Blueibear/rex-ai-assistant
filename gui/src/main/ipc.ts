import type { BrowserWindow } from 'electron'
import { registerChatHandlers } from './handlers/chat'
import { registerVoiceHandlers } from './handlers/voice'
import { registerTaskHandlers } from './handlers/tasks'
import { registerCalendarHandlers } from './handlers/calendar'
import { registerRemindersHandlers } from './handlers/reminders'
import { registerMemoriesHandlers } from './handlers/memories'
import { registerEmailHandlers } from './handlers/email'
import { registerSMSHandlers } from './handlers/sms'
import { registerNotificationHandlers } from './handlers/notifications'
import { registerSpeakerHandlers } from './handlers/speakers'
import { registerOutputRoutingHandlers } from './handlers/outputRouting'
import { registerFileHandlers } from './handlers/files'
import { registerShoppingHandlers } from './handlers/shopping'
import { registerLogsHandlers } from './handlers/logs'
import { registerUsageHandlers } from './handlers/usage'
import { registerSettingsHandlers } from './handlers/settings'
import { registerIntegrationsHandlers } from './handlers/integrations'
import { registerSystemHandlers } from './handlers/system'
import { registerHistoryHandlers } from './handlers/history'
import { registerDevicesHandlers } from './handlers/devices'
import { registerQuickActionsHandlers } from './handlers/quickActions'
import { registerSetupHandlers } from './handlers/setup'
import { registerPairingHandlers } from './handlers/pairing'
import { registerProfileHandlers } from './handlers/profile'
import type { ElectronSessionIdentity } from './sessionIdentity'

export function registerIpcHandlers(mainWindow: BrowserWindow | null, session: ElectronSessionIdentity): void {
  registerChatHandlers(session)
  registerVoiceHandlers(session)
  registerTaskHandlers(session)
  registerCalendarHandlers(session)
  registerRemindersHandlers(session)
  registerMemoriesHandlers(session)
  registerEmailHandlers(session)
  registerSMSHandlers(session)
  registerNotificationHandlers(mainWindow)
  registerSpeakerHandlers(session)
  registerOutputRoutingHandlers(session)
  registerFileHandlers(session)
  registerShoppingHandlers(session)
  registerLogsHandlers()
  registerUsageHandlers()
  registerSettingsHandlers(session)
  registerIntegrationsHandlers(session)
  registerSystemHandlers()
  registerHistoryHandlers(session)
  registerDevicesHandlers(session)
  registerQuickActionsHandlers(session)
  registerSetupHandlers()
  registerPairingHandlers(session)
  registerProfileHandlers(session)
}
