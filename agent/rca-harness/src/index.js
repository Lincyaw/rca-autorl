import { readFileSync } from 'node:fs'
import { registerNotebook } from './notebook.js'
import { seedUnnoted } from './note-ledger.js'
import { registerNotePolicy } from './note-policy.js'
import { trace } from './debug.js'
import { registerSqlTool } from './sql-tool.js'
import { registerSubmitResult } from './submit-result.js'

/** Stable Cordis plugin name. */
export const name = 'rca-harness'

/** Services this row waits for; `sdk-minimal` provides `tools`. */
export const inject = ['tools']

const DEFAULTS = { maxRows: 200, maxChars: 4000, maxCellChars: 200, noteEvery: 10, noteLimit: 20 }

/**
 * The RCA harness: the episode's whole action space. `sdk-minimal`'s shell and
 * editor rows are disabled by this bundle's patch, so the model sees exactly
 * three tools — `sql` to read the snapshot, `take_note` to record findings
 * in a persistent notebook, and `submit_result` to answer.
 */
export function apply(ctx, config = {}) {
  const limits = {
    maxRows: positiveInteger(config.maxRows, DEFAULTS.maxRows, 'maxRows'),
    maxChars: positiveInteger(config.maxChars, DEFAULTS.maxChars, 'maxChars'),
    maxCellChars: positiveInteger(config.maxCellChars, DEFAULTS.maxCellChars, 'maxCellChars'),
  }
  const noteEvery = positiveInteger(config.noteEvery, DEFAULTS.noteEvery, 'noteEvery')
  const noteLimit = positiveInteger(config.noteLimit, DEFAULTS.noteLimit, 'noteLimit')
  if (noteLimit < noteEvery) {
    throw new Error(`rca-harness: noteLimit (${noteLimit}) must not be below noteEvery (${noteEvery})`)
  }
  const snapshot = String(config.snapshot ?? '').trim()
  if (snapshot.length === 0) throw new Error('rca-harness: snapshot must be a non-empty path')
  const resultRoot = String(config.resultRoot ?? '').trim()
  if (resultRoot.length === 0) throw new Error('rca-harness: resultRoot must be a non-empty path')
  // A forked episode starts with the parent's notebook and note debt (spec §4);
  // the history itself is spliced by the sampling route.
  const fork = readFork(config.fork)
  seedUnnoted(fork.unnoted)

  // Episode state, keyed by the calling agent so one runtime can serve several
  // sessions. Everything here lives and dies with this plugin's fiber.
  const state = {
    snapshot,
    resultRoot,
    snapshots: new Map(),
    submitted: new Set(),
    notes: fork.notes,
    key: exec => exec.agent?.id ?? 'root',
  }

  trace('apply', { snapshot, resultRoot, ...limits, noteEvery, noteLimit, notePolicy: config.notePolicy !== false })
  registerSqlTool(ctx, state, limits)
  registerNotebook(ctx, state)
  if (config.notePolicy !== false) registerNotePolicy(ctx, state, noteEvery, noteLimit)
  registerSubmitResult(ctx, state)
}

function readFork(path) {
  if (path === undefined || path === null || String(path).trim() === '') return { notes: [], unnoted: 0 }
  const parsed = JSON.parse(readFileSync(String(path), 'utf8'))
  return {
    notes: Array.isArray(parsed.notes) ? parsed.notes.map(String) : [],
    unnoted: Number.isInteger(parsed.unnoted) && parsed.unnoted > 0 ? parsed.unnoted : 0,
  }
}

function positiveInteger(value, fallback, field) {
  const resolved = value ?? fallback
  if (!Number.isInteger(resolved) || resolved < 1) {
    throw new Error(`rca-harness: ${field} must be a positive integer, got ${resolved}`)
  }
  return resolved
}
