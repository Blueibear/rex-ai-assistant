import { ipcMain, app } from 'electron'
import { existsSync, readFileSync } from 'fs'
import { join } from 'path'

interface UsageBucket {
  requests: number
  tokens: number
}

interface PeriodSplit {
  local: UsageBucket
  cloud: UsageBucket
}

interface UsageSummary {
  ok: boolean
  local: UsageBucket
  cloud: UsageBucket
  by_period: {
    today: PeriodSplit
    week: PeriodSplit
    month: PeriodSplit
  }
  error?: string
}

interface UsageRecord {
  model?: string
  prompt_tokens?: number
  completion_tokens?: number
  timestamp?: string
}

const CLOUD_PREFIXES = [
  'gpt-', 'text-', 'o1', 'o3', 'babbage', 'davinci', 'curie', 'ada',
  'claude-', 'gemini-', 'mistral-'
]

function isCloudModel(model: string): boolean {
  const lower = model.toLowerCase()
  return CLOUD_PREFIXES.some(p => lower.startsWith(p))
}

function emptyBucket(): UsageBucket {
  return { requests: 0, tokens: 0 }
}

function emptyPeriod(): PeriodSplit {
  return { local: emptyBucket(), cloud: emptyBucket() }
}

function periodStart(now: Date, period: 'today' | 'week' | 'month'): Date {
  const d = new Date(now)
  if (period === 'today') {
    d.setHours(0, 0, 0, 0)
    return d
  }
  if (period === 'week') {
    const day = d.getDay() // 0=Sun
    d.setDate(d.getDate() - day)
    d.setHours(0, 0, 0, 0)
    return d
  }
  // month
  d.setDate(1)
  d.setHours(0, 0, 0, 0)
  return d
}

function computeUsageSummary(records: UsageRecord[]): Omit<UsageSummary, 'ok'> {
  const now = new Date()
  const starts = {
    today: periodStart(now, 'today'),
    week: periodStart(now, 'week'),
    month: periodStart(now, 'month')
  }

  const totals = { local: emptyBucket(), cloud: emptyBucket() }
  const byPeriod = {
    today: emptyPeriod(),
    week: emptyPeriod(),
    month: emptyPeriod()
  }

  for (const rec of records) {
    const model = rec.model ?? ''
    const bucket = isCloudModel(model) ? 'cloud' : 'local'
    const tokens = (rec.prompt_tokens ?? 0) + (rec.completion_tokens ?? 0)

    totals[bucket].requests += 1
    totals[bucket].tokens += tokens

    let ts: Date | null = null
    if (rec.timestamp) {
      const parsed = new Date(rec.timestamp)
      if (!isNaN(parsed.getTime())) {
        ts = parsed
      }
    }

    if (ts !== null) {
      for (const period of ['today', 'week', 'month'] as const) {
        if (ts >= starts[period]) {
          byPeriod[period][bucket].requests += 1
          byPeriod[period][bucket].tokens += tokens
        }
      }
    }
  }

  return { local: totals.local, cloud: totals.cloud, by_period: byPeriod }
}

function resolveUsageFile(): string {
  return join(app.getAppPath(), '..', 'data', 'llm_usage.json')
}

function loadUsageRecords(filePath: string): UsageRecord[] {
  if (!existsSync(filePath)) return []
  try {
    const content = readFileSync(filePath, 'utf-8')
    const records: UsageRecord[] = []
    for (const line of content.split('\n')) {
      const trimmed = line.trim()
      if (trimmed) {
        try {
          records.push(JSON.parse(trimmed) as UsageRecord)
        } catch {
          // skip malformed lines
        }
      }
    }
    return records
  } catch {
    return []
  }
}

export function registerUsageHandlers(): void {
  ipcMain.handle('rex:getUsage', async (): Promise<UsageSummary> => {
    try {
      const filePath = resolveUsageFile()
      const records = loadUsageRecords(filePath)
      const summary = computeUsageSummary(records)
      return { ok: true, ...summary }
    } catch (err) {
      return {
        ok: false,
        local: emptyBucket(),
        cloud: emptyBucket(),
        by_period: { today: emptyPeriod(), week: emptyPeriod(), month: emptyPeriod() },
        error: String(err)
      }
    }
  })
}
