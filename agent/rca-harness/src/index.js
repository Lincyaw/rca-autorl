import { readFileSync } from 'node:fs'
import z from '@deepseek-ai/schemastery'
import { registerNotebook } from './notebook.js'
import { seedUnnoted } from './note-ledger.js'
import { parseTemperature, registerSampling } from './sampling.js'
import { registerNotePolicy } from './note-policy.js'
import { trace } from './debug.js'
import { registerSqlTool } from './sql-tool.js'
import { registerSubmitResult } from './submit-result.js'

/** Stable Cordis plugin name. */
export const name = 'rca-harness'

/** Services this row waits for; `sdk-minimal` provides `tools`. */
export const inject = ['tools', 'llm']

const positiveInteger = z.number().step(1).min(1).required()

/** Validated by the loader before `apply`; the bundle's own patch sets every limit. */
export const Config = z.object({
  snapshot: z.string().required(),
  resultRoot: z.string().required(),
  maxRows: positiveInteger,
  maxChars: positiveInteger,
  maxCellChars: positiveInteger,
  noteEvery: positiveInteger,
  noteLimit: positiveInteger,
  // The trainer's temperature as the environment carries it, and the fork
  // prefix file; both optional.
  temperature: z.string(),
  fork: z.string(),
})

/**
 * The RCA harness: the episode's whole action space. `sdk-minimal`'s shell and
 * editor rows are disabled by this bundle's patch, so the model sees exactly
 * three tools — `sql` to read the snapshot, `take_note` to record findings
 * in a persistent notebook, and `submit_result` to answer.
 */
export function apply(ctx, config) {
  const { snapshot, resultRoot, maxRows, maxChars, maxCellChars, noteEvery, noteLimit } = config
  const limits = { maxRows, maxChars, maxCellChars }
  if (noteLimit < noteEvery) {
    throw new Error(`rca-harness: noteLimit (${noteLimit}) must not be below noteEvery (${noteEvery})`)
  }
  // A forked episode starts with the parent's history, notebook and note debt
  // (spec §4); the temperature is the trainer's.
  const fork = readFork(config.fork)
  seedUnnoted(fork.unnoted)
  const temperature = parseTemperature(config.temperature)

  // Episode state, keyed by the calling agent so one runtime can serve several
  // sessions. Everything here lives and dies with this plugin's fiber.
  const state = {
    snapshot,
    resultRoot,
    submitted: new Set(),
    notes: fork.notes,
    key: exec => exec.agent?.id ?? 'root',
  }

  trace('apply', { snapshot, resultRoot, ...limits, noteEvery, noteLimit, temperature, fork: fork.messages.length })
  registerSampling(ctx, temperature, fork.messages)
  registerSqlTool(ctx, state, limits)
  registerNotebook(ctx, state)
  registerNotePolicy(ctx, state, noteEvery, noteLimit)
  registerSubmitResult(ctx, state)
}

/** The prefix file `autorl.fork.fork_prefix` writes, or an empty one. */
function readFork(path) {
  if (path === undefined || path === null || String(path).trim() === '') {
    return { messages: [], notes: [], unnoted: 0 }
  }
  const parsed = JSON.parse(readFileSync(String(path), 'utf8'))
  if (!Array.isArray(parsed.messages)) throw new Error('rca-harness: fork prefix has no messages')
  return {
    messages: parsed.messages,
    notes: Array.isArray(parsed.notes) ? parsed.notes.map(String) : [],
    unnoted: Number.isInteger(parsed.unnoted) && parsed.unnoted > 0 ? parsed.unnoted : 0,
  }
}
