/**
 * Centralized bridge path resolver for Electron → Python bridge scripts.
 *
 * All Electron spawn/exec calls for bridge scripts must route through
 * resolveBridgePath() so paths are never hardcoded inline.
 *
 * Call validateBridges() at app launch to surface missing scripts early,
 * before a user action triggers a confusing "bridge exited" error.
 */

import { app } from 'electron'
import { join } from 'path'
import { existsSync } from 'fs'

// ---------------------------------------------------------------------------
// Bridge registry — maps bridge name to repo-relative script filename
// ---------------------------------------------------------------------------

const BRIDGE_REGISTRY: Record<string, string> = {
  rex_tasks_bridge: 'rex_tasks_bridge.py',
  rex_reminders_bridge: 'rex_reminders_bridge.py',
  rex_shopping_list_bridge: 'rex_shopping_list_bridge.py',
  rex_speaker_bridge: 'rex_speaker_bridge.py',
  rex_chat_bridge: 'rex_chat_bridge.py',
  rex_chat_stream_bridge: 'rex_chat_stream_bridge.py',
  rex_voices_bridge: 'rex_voices_bridge.py',
  rex_voice_enrollment_bridge: 'rex_voice_enrollment_bridge.py',
  rex_voice_sample_bridge: 'rex_voice_sample_bridge.py',
  rex_voice_upload_bridge: 'rex_voice_upload_bridge.py',
  rex_wakeword_list_bridge: 'rex_wakeword_list_bridge.py',
  rex_wakeword_train_bridge: 'rex_wakeword_train_bridge.py',
  rex_wakeword_sample_bridge: 'rex_wakeword_sample_bridge.py',
  rex_stt_bridge: 'rex_stt_bridge.py',
  rex_memories_bridge: 'rex_memories_bridge.py',
  rex_file_extract_bridge: 'rex_file_extract_bridge.py',
  rex_voice_bridge: 'rex_voice_bridge.py',
}

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/** Return the absolute path to a bridge script given its filename (e.g. `"rex_tasks_bridge.py"`). */
export function resolveBridgePath(scriptFilename: string): string {
  return join(app.getAppPath(), '..', scriptFilename)
}

// ---------------------------------------------------------------------------
// Python executable resolution
// ---------------------------------------------------------------------------

/** Return the Python executable path: bundled venv if present, else system `python`. */
export function resolvePythonCommand(): string {
  const bundledVenvPython = join(app.getAppPath(), '..', '.venv', 'Scripts', 'python.exe')
  return existsSync(bundledVenvPython) ? bundledVenvPython : 'python'
}

// ---------------------------------------------------------------------------
// Startup validation
// ---------------------------------------------------------------------------

/**
 * Validate that all registered bridge scripts exist on disk.
 * Call once at app launch (inside `app.whenReady()`).
 * Logs an error for each missing script so the developer knows exactly
 * which bridge is absent and where it should be.
 */
export function validateBridges(): void {
  let allPresent = true
  for (const [name, filename] of Object.entries(BRIDGE_REGISTRY)) {
    const resolvedPath = resolveBridgePath(filename)
    if (!existsSync(resolvedPath)) {
      console.error(
        `[bridgeResolver] Missing bridge script "${name}": expected at ${resolvedPath}`
      )
      allPresent = false
    }
  }
  if (allPresent) {
    console.log('[bridgeResolver] All bridge scripts validated successfully.')
  }
}
