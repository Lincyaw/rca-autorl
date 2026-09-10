/**
 * What `autorl.web` returns. One declaration per endpoint payload, kept in the
 * shape the Python side builds it — `Episode.summary()` for the list rows and
 * `web._detail` for everything else — so a drift shows up as a type error here
 * rather than as an empty panel.
 */

export interface Run {
  run: string
  episodes: number
}

export interface Step {
  index: number
  reasoning: string
  text: string
  tool: string
  arguments: string
  result_head: string
  result_error: boolean
  input_tokens: number
}

export interface Compaction {
  at_step: number
  shadowed_tokens: number
  summary: string
  error: string
  landed: boolean
}

/** The three axes of the graph score, as sets of subjects. */
export interface Axis {
  found: string[]
  missed: string[]
  invented: string[]
}

export interface EpisodeRow {
  session_id: string
  case: string
  finish: string
  error: string
  score: number
  submitted: boolean
  steps: number
  tools: Record<string, number>
  peak_input_tokens: number
  compactions: number
  compaction_failures: number
}

export interface Episode extends Omit<EpisodeRow, 'steps' | 'compactions'> {
  steps: Step[]
  compactions: Compaction[]
  submission: Record<string, unknown> | null
  axes?: Record<string, Axis>
}

export interface CaseStats {
  cases: number
  episodes: number
  submission_rate: number
  pass_1: number
  pass_k: number
  varied_fraction: number
  per_case: {
    case: string
    k: number
    pass_1: number
    pass_k: number
    spread: number
    submitted: number
  }[]
}

export interface CaseGroup {
  case: string
  stats: CaseStats
  episodes: Episode[]
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) throw new Error(`${response.status} ${path}`)
  return response.json() as Promise<T>
}

export const api = {
  runs: () => get<Run[]>('/api/runs'),
  episodes: (run: string) => get<EpisodeRow[]>(`/api/runs/${run}/episodes`),
  episode: (run: string, id: string) => get<Episode>(`/api/runs/${run}/episodes/${id}`),
  group: (run: string, name: string) => get<CaseGroup>(`/api/runs/${run}/cases/${name}`),
  async report(run: string): Promise<string> {
    const response = await fetch(`/api/runs/${run}/report`)
    if (!response.ok) throw new Error(`${response.status} report`)
    return response.text()
  },
}

/** 0.6 x 32768: where compaction fires, per `agent/rca-harness/cordis.patch.yml`. */
export const COMPACTION_THRESHOLD = 19661
/** The serving window a request cannot cross, per `sglang.context_length`. */
export const CONTEXT_WINDOW = 32768
