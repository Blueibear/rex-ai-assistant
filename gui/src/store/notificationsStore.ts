import { create } from 'zustand'

interface NotificationsState {
  unreadCount: number
  setUnreadCount: (count: number) => void
  increment: () => void
  clear: () => void
  fetchUnreadCount: () => Promise<void>
}

export const useNotificationsStore = create<NotificationsState>((set) => ({
  unreadCount: 0,
  setUnreadCount: (count) => set({ unreadCount: count }),
  increment: () => set((state) => ({ unreadCount: state.unreadCount + 1 })),
  clear: () => set({ unreadCount: 0 }),
  fetchUnreadCount: async () => {
    try {
      const count = await window.rex.getUnreadNotificationCount()
      set({ unreadCount: count })
    } catch {
      // Keep previous count on error
    }
  }
}))
