import { ElectronAPI } from '@electron-toolkit/preload'
import type { RexAPI } from '../types/ipc'

declare global {
  interface Window {
    electron: ElectronAPI
    rex: RexAPI
  }
}
